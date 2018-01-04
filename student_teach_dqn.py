import os, logging, gym, uuid, argparse, cv2, six, h5py, random

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque

from six.moves import queue

import keras.layers as layers
from keras.utils import np_utils
from keras.models import Model as KModel
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint

os.environ['TENSORPACK_TRAIN_API'] = 'v2'   # will become default soon

from tensorpack import *
from tensorpack.utils.concurrency import ensure_proc_terminate, start_proc_mask_signal
from tensorpack.utils.serialize import dumps
from tensorpack.tfutils import symbolic_functions as symbf
from tensorpack.tfutils.gradproc import MapGradient, SummaryGradient
from tensorpack.utils.gpu import get_nr_gpu

from simulator import SimulatorProcess, SimulatorMaster, TransitionExperience
from common import Evaluator, eval_model_multithread, play_n_episodes, play_n_episodes_recording
from atari_wrapper import MapState, FrameStack, FireResetEnv, LimitLength

if six.PY3:
    from concurrent import futures
    CancelledError = futures.CancelledError
else:
    CancelledError = Exception

IMAGE_SIZE = (84, 84)
FRAME_HISTORY = 4
GAMMA = 0.99
CHANNEL = FRAME_HISTORY * 3
IMAGE_SHAPE3 = IMAGE_SIZE + (CHANNEL,)

LOCAL_TIME_MAX = 5
STEPS_PER_EPOCH = 6000
EVAL_EPISODE = 50
BATCH_SIZE = 128
PREDICT_BATCH_SIZE = 15     # batch for efficient forward
SIMULATOR_PROC = 50
PREDICTOR_THREAD_PER_GPU = 3
PREDICTOR_THREAD = None

NUM_ACTIONS = None
ENV_NAME = None


def get_player(train=False, dumpdir=None):
    env = gym.make(ENV_NAME)
    if dumpdir:
        env = gym.wrappers.Monitor(env, dumpdir, force=True)
    env = FireResetEnv(env)
    env = MapState(env, lambda im: cv2.resize(im, IMAGE_SIZE))
    env = FrameStack(env, 4)
    if train:
        env = LimitLength(env, 60000)
    return env

class MySimulatorWorker(SimulatorProcess):
    def _build_player(self):
        return get_player(train=True)

class Model(ModelDesc):
    def _get_inputs(self):
        assert NUM_ACTIONS is not None
        return [InputDesc(tf.uint8, (None,) + IMAGE_SHAPE3, 'state'),
                InputDesc(tf.int64, (None,), 'action'),
                InputDesc(tf.float32, (None,), 'futurereward'),
                InputDesc(tf.float32, (None,), 'action_prob'),
                ]

    def _get_NN_prediction(self, image):
        image = tf.cast(image, tf.float32) / 255.0
        with argscope(Conv2D, nl=tf.nn.relu):
            l = Conv2D('conv0', image, out_channel=32, kernel_shape=5)
            l = MaxPooling('pool0', l, 2)
            l = Conv2D('conv1', l, out_channel=32, kernel_shape=5)
            l = MaxPooling('pool1', l, 2)
            l = Conv2D('conv2', l, out_channel=64, kernel_shape=4)
            l = MaxPooling('pool2', l, 2)
            l = Conv2D('conv3', l, out_channel=64, kernel_shape=3)

        l = FullyConnected('fc0', l, 512, nl=tf.identity)
        l = PReLU('prelu', l)
        logits = FullyConnected('fc-pi', l, out_dim=NUM_ACTIONS, nl=tf.identity)    # unnormalized policy
        value = FullyConnected('fc-v', l, 1, nl=tf.identity)
        return logits, value

    def _build_graph(self, inputs):
        state, action, futurereward, action_prob = inputs
        logits, value = self._get_NN_prediction(state)
        value = tf.squeeze(value, [1], name='pred_value')  # (B,)
        policy = tf.nn.softmax(logits, name='policy')
        is_training = get_current_tower_context().is_training
        if not is_training:
            return
        log_probs = tf.log(policy + 1e-6)

        log_pi_a_given_s = tf.reduce_sum(
            log_probs * tf.one_hot(action, NUM_ACTIONS), 1)
        advantage = tf.subtract(tf.stop_gradient(value), futurereward, name='advantage')

        pi_a_given_s = tf.reduce_sum(policy * tf.one_hot(action, NUM_ACTIONS), 1)  # (B,)
        importance = tf.stop_gradient(tf.clip_by_value(pi_a_given_s / (action_prob + 1e-8), 0, 10))

        policy_loss = tf.reduce_sum(log_pi_a_given_s * advantage * importance, name='policy_loss')
        xentropy_loss = tf.reduce_sum(policy * log_probs, name='xentropy_loss')
        value_loss = tf.nn.l2_loss(value - futurereward, name='value_loss')

        pred_reward = tf.reduce_mean(value, name='predict_reward')
        advantage = symbf.rms(advantage, name='rms_advantage')
        entropy_beta = tf.get_variable('entropy_beta', shape=[],
                                       initializer=tf.constant_initializer(0.01), trainable=False)
        self.cost = tf.add_n([policy_loss, xentropy_loss * entropy_beta, value_loss])
        self.cost = tf.truediv(self.cost,
                               tf.cast(tf.shape(futurereward)[0], tf.float32),
                               name='cost')
        summary.add_moving_summary(policy_loss, xentropy_loss,
                                   value_loss, pred_reward, advantage,
                                   self.cost, tf.reduce_mean(importance, name='importance'))

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.001, trainable=False)
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)

        gradprocs = [MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.1)),
                     SummaryGradient()]
        opt = optimizer.apply_grad_processors(opt, gradprocs)
        return opt

class MySimulatorMaster(SimulatorMaster, Callback):
    def __init__(self, pipe_c2s, pipe_s2c, model, gpus):
        super(MySimulatorMaster, self).__init__(pipe_c2s, pipe_s2c)
        self.M = model
        self.queue = queue.Queue(maxsize=BATCH_SIZE * 8 * 2)
        self._gpus = gpus

    def _setup_graph(self):
        # create predictors on the available predictor GPUs.
        nr_gpu = len(self._gpus)
        predictors = [self.trainer.get_predictor(
            ['state'], ['policy', 'pred_value'],
            self._gpus[k % nr_gpu])
            for k in range(PREDICTOR_THREAD)]
        self.async_predictor = MultiThreadAsyncPredictor(
            predictors, batch_size=PREDICT_BATCH_SIZE)

    def _before_train(self):
        self.async_predictor.start()

    def _on_state(self, state, ident):
        def cb(outputs):
            try:
                distrib, value = outputs.result()
            except CancelledError:
                logger.info("Client {} cancelled.".format(ident))
                return
            assert np.all(np.isfinite(distrib)), distrib
            action = np.random.choice(len(distrib), p=distrib)
            client = self.clients[ident]
            client.memory.append(TransitionExperience(
                state, action, reward=None, value=value, prob=distrib[action]))
            self.send_queue.put([ident, dumps(action)])
        self.async_predictor.put_task([state], cb)

    def _on_episode_over(self, ident):
        self._parse_memory(0, ident, True)

    def _on_datapoint(self, ident):
        client = self.clients[ident]
        if len(client.memory) == LOCAL_TIME_MAX + 1:
            R = client.memory[-1].value
            self._parse_memory(R, ident, False)

    def _parse_memory(self, init_r, ident, isOver):
        client = self.clients[ident]
        mem = client.memory
        if not isOver:
            last = mem[-1]
            mem = mem[:-1]

        mem.reverse()
        R = float(init_r)
        for idx, k in enumerate(mem):
            R = np.clip(k.reward, -1, 1) + GAMMA * R
            self.queue.put([k.state, k.action, R, k.prob])

        if not isOver:
            client.memory = [last]
        else:
            client.memory = []

def get_config():
    nr_gpu = get_nr_gpu()
    global PREDICTOR_THREAD
    if nr_gpu > 0:
        if nr_gpu > 1:
            # use half gpus for inference
            predict_tower = list(range(nr_gpu))[-nr_gpu // 2:]
        else:
            predict_tower = [0]
        PREDICTOR_THREAD = len(predict_tower) * PREDICTOR_THREAD_PER_GPU
        train_tower = list(range(nr_gpu))[:-nr_gpu // 2] or [0]
        logger.info("[Batch-A3C] Train on gpu {} and infer on gpu {}".format(
            ','.join(map(str, train_tower)), ','.join(map(str, predict_tower))))
    else:
        logger.warn("Without GPU this model will never learn! CPU is only useful for debug.")
        PREDICTOR_THREAD = 1
        predict_tower, train_tower = [0], [0]

    # setup simulator processes
    name_base = str(uuid.uuid1())[:6]
    PIPE_DIR = os.environ.get('TENSORPACK_PIPEDIR', '.').rstrip('/')
    namec2s = 'ipc://{}/sim-c2s-{}'.format(PIPE_DIR, name_base)
    names2c = 'ipc://{}/sim-s2c-{}'.format(PIPE_DIR, name_base)
    procs = [MySimulatorWorker(k, namec2s, names2c) for k in range(SIMULATOR_PROC)]
    ensure_proc_terminate(procs)
    start_proc_mask_signal(procs)

    M = Model()
    master = MySimulatorMaster(namec2s, names2c, M, predict_tower)
    dataflow = BatchData(DataFromQueue(master.queue), BATCH_SIZE)
    return TrainConfig(
        model=M,
        dataflow=dataflow,
        callbacks=[
            ModelSaver(),
            ScheduledHyperParamSetter('learning_rate', [(20, 0.0003), (120, 0.0001)]),
            ScheduledHyperParamSetter('entropy_beta', [(80, 0.005)]),
            HumanHyperParamSetter('learning_rate'),
            HumanHyperParamSetter('entropy_beta'),
            master,
            StartProcOrThread(master),
            PeriodicTrigger(Evaluator(
                EVAL_EPISODE, ['state'], ['policy'], get_player),
                every_k_epochs=3),
        ],
        session_creator=sesscreate.NewSessionCreator(
            config=get_default_sess_config(0.5)),
        steps_per_epoch=STEPS_PER_EPOCH,
        max_epoch=1000,
        tower=train_tower
    )

class student_dqn:
    
    def __init__(self,
                 env,
                 teacher_size=(),
                 state_size=(84,84,3,),# Size of the "image" 
                 frustration=20,       # Number of no reward steps before frustrated
                 confusion=0.0013,     # Minimum difference b/w top choices before confused
                 teacher=None):        # Teacher Model
        self.env = env
        self.state_size = state_size
        self.action_size = env.action_space.n
        self.memory = deque(maxlen=35000)
        self.dr = 0.95                 # Discount Rate 
        self.epsilon = 1.0             # Probability of Random Move
        self.min_epsilon = 0.01        # Smallest Epsilon Value
        self.epsilon_decay = 0.99      # Decay Speed of Epsilon
        self.model = self.build_prediction_network(trainable=True)
        self.target_model = self.build_prediction_network()
        
        # Specify Parameters for Teaching
        self.rebound = .5 # How likely a student is to become less frustrated 
        self.frustration = frustration 
        self.confusion = confusion
        self.frustration_max = frustration
        
        self.teaching = False
        self.teach_for = 20 # How many steps (max) should the teacher take
        self.has_taught = 0
        
        self.teacher = teacher

        self.model.compile(optimizer='adam', loss='mse')
        
    def build_prediction_network(self, trainable=True):
        # Input Layer for Observation
        obs_input = layers.Input(shape=self.state_size)

        # Convolutional Layers
        conv1 = layers.Conv2D(32, 5, padding='same', activation='relu', trainable=trainable)(obs_input)
        conv1 = layers.MaxPooling2D(2, padding='same', trainable=trainable)(conv1)
        conv1 = layers.Dropout(0.2, trainable=trainable)(conv1)

        conv1 = layers.Conv2D(64, 5, padding='same', activation='relu', trainable=trainable)(conv1)
        conv1 = layers.MaxPooling2D(2, padding='same', trainable=trainable)(conv1)
        conv1 = layers.Dropout(0.2, trainable=trainable)(conv1)

        conv1 = layers.Conv2D(128, 5, padding='same', activation='relu', trainable=trainable)(conv1)
        conv1 = layers.MaxPooling2D(2, padding='same', trainable=trainable)(conv1)
        conv1 = layers.Dropout(0.2, trainable=trainable)(conv1)
        
        conv_o = layers.Flatten()(conv1)

        # Dense Layers
        dense2 = layers.Dense(128, activation='relu', trainable=trainable)(conv_o)
        dense2 = layers.Dropout(0.3, trainable=trainable)(dense2)
        dense2 = layers.Dense(256, activation='relu', trainable=trainable)(conv_o)
        dense2 = layers.Dropout(0.3, trainable=trainable)(dense2)

        # Output
        prediction = layers.Dense(self.action_size, activation='linear', trainable=trainable)(dense2)

        # Theory Block Compile
        value_predict = KModel(inputs=obs_input, outputs=prediction)

        return value_predict
    
    def build_memory(self, ob, act, reward, next_state, done):
        self.memory.append((ob,act,reward,next_state,done))
    
    def act(self, ob):
        if self.teaching:
            # The teacher is currently playing
            return self.teacher(ob[None, :, :, :])[0][0].argmax()
        
        if np.random.rand() <= self.epsilon:
            # Random "Exploration Action"
            return env.action_space.sample()
        
        predicts = self.model.predict(ob[None, :, :, 9:]/255.)[0]
        ord_predicts = np.argsort(predicts, axis=-1, kind='quicksort', order=None)
        confidence_d = predicts[ord_predicts[-1]]-predicts[ord_predicts[-2]]
        
        if confidence_d <= self.confusion:
            # Need the Teachers Help
            return self.teacher(ob[None, :, :, :])[0][0].argmax()
        
        return ord_predicts[-1]
    
    def update_target(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)
    
    def replay(self, batch_size=32):
        minibatch = random.sample(self.memory, batch_size)
        
        for ob, action, reward, next_state, done in minibatch:
            ob_train = ob[None, :, :, 9:]/255.
            ns_train = next_state[None, :, :, 9:]/255.
            target = reward
            if not done:
                target += self.dr * np.amax(self.target_model.predict(ns_train)[0])
                
            target_f = self.model.predict(ob_train)
            target_f[0][action] = target
            self.model.fit(ob_train, target_f, epochs=1, verbose=0)
        
        self.teaching = False
        self.frustration = self.frustration_max
        self.has_taught = 0
        
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

if __name__ == "__main__":
    ENV_NAME = 'MsPacman-v0'
    NUM_ACTIONS = get_player().action_space.n
    env = gym.make(ENV_NAME)
    
    env = FireResetEnv(env)
    env = MapState(env, lambda im: cv2.resize(im, (84,84)))
    env = FrameStack(env, 4)

    pred = OfflinePredictor(PredictConfig(
                model=Model(),
                session_init=get_model_loader("models/MsPacman-v0.tfmodel"),
                input_names=['state'],
                output_names=['policy']))

    student = student_dqn(env, teacher=pred)
    episodes = 1000000
    scores = []
    teacher_step_nums = []
    step_nums = []
    ep_avgs = [0]
    ep_avg = []

    for ee in range(episodes):
        ob = env.reset()
        score_ = 0
        steps = 0
        t_steps = 0
        
        while True:
            steps += 1
            action = student.act(ob)
        
            next_ob, reward, done, _ = env.step(action)
            score_ += reward
            student.build_memory(ob, action, reward, next_ob, done)
            
            ob = next_ob
                    
            if done:
                if (ee+1) % 20 == 0 or ee == 0:
                    student.model.save_weights("teacher_student_weights.h5")
                    if ee == 0:
                        print "Completed {}/{} with a score of {}.".format(ee+1, episodes, score_)
                    else:
                        ep_avgs.append(np.mean(ep_avg))
                        ep_avg = []
                        print "Completed {}/{} with a score of {}. Average over the last 20 epochs was {} ({}).".format(ee+1, episodes, score_, ep_avgs[-1], ep_avgs[-1]-ep_avgs[-2])
                
                scores.append(score_)
                step_nums.append(steps)
                teacher_step_nums.append(t_steps)
                ep_avg.append(score_)
                
                if (ee+1) % 200 == 0 or ee == 0:
                    print "Saving Checkpoint Data."

                    np.save("data/scores_"+str(ee), scores)
                    np.save("data/teacher_step_nums_"+str(ee), teacher_step_nums)
                    np.save("data/step_nums_"+str(ee), step_nums)
                    np.save("data/ep_avgs_"+str(ee), ep_avgs)
                break
            
            # Check if the Teacher is Player
            if student.teaching:
                t_steps += 1
                student.has_taught += 1
                if student.has_taught == student.teach_for:
                    # Stop the Teacher
                    student.has_taught = 0
                    student.teaching = False
                    student.frustration = student.frustration_max
            else:
                # Update Model Frustration
                if reward == 0:
                    student.frustration -= 1
                    if student.frustration == 0:
                        student.teaching = True
                else:
                    student.frustration = min(student.frustration_max, student.frustration + int(np.random.rand()<student.rebound))
        
        student.replay()
        student.update_target()
        
    student.model.save_weights("final_weights_student.h5")    
    np.save("scores", scores)
    np.save("teacher_step_nums", teacher_step_nums)
    np.save("step_nums", step_nums)
    np.save("ep_avgs", ep_avgs)
