import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
from scipy.stats import pearsonr

# Build Model w/ Keras
# Needed Keras Libraries
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D, Merge
from keras.utils import np_utils
from keras.models import Model
from keras import backend as K

from keras.layers.merge import Concatenate

import h5py

from keras.callbacks import EarlyStopping

# Input Layer for Observation
obs_input = Input(shape=(84,84,12,))

# Convolutional Layers
conv1 = Conv2D(32, 5, padding='same', activation='relu')(obs_input)
conv1 = MaxPooling2D(2, padding='same')(conv1)
conv1 = Conv2D(32, 5, padding='same', activation='relu')(conv1)
conv1 = MaxPooling2D(2, padding='same')(conv1)

conv_o = Flatten()(conv1)

# Dense Layers
dense2 = Dense(512, activation='relu')(conv_o)

# Output
prediction = Dense(9, activation='softmax')(dense2)

# Theory Block Compile
student = Model(inputs=obs_input, outputs=prediction)
student.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
student.summary()

# The typical imports
import gym
import numpy as np
import matplotlib.pyplot as plt

import os, logging, gym
import tensorflow as tf

import numpy as np
import os
import uuid
import argparse

import cv2
import tensorflow as tf
import six
from six.moves import queue

os.environ['TENSORPACK_TRAIN_API'] = 'v2'   # will become default soon
from tensorpack import *
from tensorpack.utils.concurrency import ensure_proc_terminate, start_proc_mask_signal
from tensorpack.utils.serialize import dumps
from tensorpack.tfutils import symbolic_functions as symbf
from tensorpack.tfutils.gradproc import MapGradient, SummaryGradient
from tensorpack.utils.gpu import get_nr_gpu


import gym
from simulator import SimulatorProcess, SimulatorMaster, TransitionExperience
from common import Evaluator, eval_model_multithread, play_n_episodes, play_n_episodes_recording
from atari_wrapper import MapState, FrameStack, FireResetEnv, LimitLength

env = gym.make('MsPacman-v0')
env = FireResetEnv(env)
env = MapState(env, lambda im: cv2.resize(im, (84,84)))
env = FrameStack(env, 4)
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
NUM_ACTIONS = env.action_space.n
IMAGE_SIZE = (84, 84)
FRAME_HISTORY = 4
GAMMA = 0.99
CHANNEL = FRAME_HISTORY * 3
IMAGE_SHAPE3 = IMAGE_SIZE + (CHANNEL,)
pred = OfflinePredictor(PredictConfig(
            model=Model(),
            session_init=get_model_loader('models/MsPacman-v0.tfmodel'),
            input_names=['state'],
            output_names=['policy']))

observations_ = []
actions_ = []
teacher_actions = []
student.load_weights("Post-Long_Train.h5")
max_score = 0

for i in range(5000):
    print "===== {} =====".format(str(i))
    ob = env.reset()
    sum_r = 0

    while True:
        act_t = pred(ob[None, :, :, :])[0][0].argmax() # Record Teacher Action for Training
        act_s = student.predict(np.array(ob[None, :, :, :])/255.)[0].argmax() # Students Action
        
        if act_t != act_s:
            observations_.append(ob/255.) # Append Observation
            teacher_actions.append(act_t)
        
        ob, r, isOver, info = env.step(act_s) # Take Step    
        sum_r += r
        
        if isOver:
            print "Total Score \t {}".format(str(sum_r))
            break
    
    scrambled_idx = np.random.choice(len(np.array(observations_)),size=(len(np.array(observations_)),),replace=False)
    print "Training Count \t {}".format(str(len(scrambled_idx)))
    student.fit(np.array(observations_)[scrambled_idx], 
                np_utils.to_categorical(teacher_actions,num_classes=9)[scrambled_idx], 
                epochs=5,
                batch_size=8,
                verbose=False)
    print "Trained Accuracy \t {}".format(str(student.evaluate(np.array(observations_)[scrambled_idx], 
                np_utils.to_categorical(teacher_actions,num_classes=9)[scrambled_idx])[1]))


    if sum_r > max_score:
        print "New Max Score!"
        max_score = sum_r
        student.save_weights("Student_Max_score="+str(sum_r)+".h5")
    
    if (len(observations_)) >= 10000:
        break

np.save("diff_obs",np.array(observations_),allow_pickle=False)
student.save_weights("Pre_Adjustments.h5")
student.fit(np.array(observations_)[scrambled_idx], 
            np_utils.to_categorical(teacher_actions,num_classes=9)[scrambled_idx], 
            epochs=100,
            batch_size=8,
            verbose=True)

student.save_weights("Post_Adjustments.h5")
