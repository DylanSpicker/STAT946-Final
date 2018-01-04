import os, logging, gym, uuid, argparse, cv2, six, h5py, random

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from collections import deque

import tensorflow as tf

import keras.layers as layers
from keras.utils import np_utils
from keras.models import Model
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint

from atari_wrapper import MapState, FrameStack, FireResetEnv, LimitLength

class student_dqn:
    def __init__(self,
                 env,
                 state_size=(84,84,6,), # Size of the "image" 
                 frustration=20,       # Number of no reward steps before frustrated
                 confusion=0.15,       # Minimum threshold before confused
                 teacher=None):        # Teacher Model
        self.env = env
        self.state_size = state_size
        self.action_size = env.action_space.n
        self.memory = deque(maxlen=15000)
        self.dr = 0.99                 # Discount Rate 
        self.epsilon = 1.0             # Probability of Random Move
        self.min_epsilon = 0.01        # Smallest Epsilon Value
        self.epsilon_decay = 0.9995       # Decay Speed of Epsilon
        self.frustration = frustration 
        self.confusion = confusion
        self.model = self.build_prediction_network(trainable=True)
        self.target_model = self.build_prediction_network()
        self.teacher = teacher

        self.model.compile(optimizer='adam', loss='mse')
        
    def build_prediction_network(self, trainable=True):
        # Input Layer for Observation
        obs_input = layers.Input(shape=self.state_size)

        # Convolutional Layers
        conv1 = layers.Conv2D(32, 5, padding='same', activation='relu', trainable=trainable)(obs_input)
        conv1 = layers.MaxPooling2D(2, padding='same', trainable=trainable)(conv1)
        conv1 = layers.Dropout(0.2, trainable=trainable)(conv1)

        conv1 = layers.Conv2D(32, 5, padding='same', activation='relu', trainable=trainable)(conv1)
        conv1 = layers.MaxPooling2D(2, padding='same', trainable=trainable)(conv1)
        conv1 = layers.Dropout(0.2, trainable=trainable)(conv1)

        conv_o = layers.Flatten()(conv1)

        # Dense Layers
        dense2 = layers.Dense(256, activation='relu', trainable=trainable)(conv_o)
        dense2 = layers.Dropout(0.3, trainable=trainable)(dense2)
        
        # Output
        prediction = layers.Dense(self.action_size, activation='linear', trainable=trainable)(dense2)

        # Theory Block Compile
        value_predict = Model(inputs=obs_input, outputs=prediction)

        return value_predict
    
    def build_memory(self, ob, act, reward, next_state, done):
        self.memory.append((ob,act,reward,next_state,done))
    
    def act(self, ob):
        if np.random.rand() <= self.epsilon:
            # Random "Exploration Action"
            return env.action_space.sample()
        #TODO: Implement Teacher Assistance
        return self.model.predict(ob)[0].argmax()
    
    def update_target(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)
    
    def replay(self, batch_size=32):
        minibatch = random.sample(self.memory, batch_size)
        
        for ob, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.dr * np.amax(self.target_model.predict(next_state)[0])
                
            target_f = self.model.predict(ob)
            target_f[0][action] = target
            self.model.fit(ob, target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

print __name__

env = gym.make('MsPacman-v0')
env = FireResetEnv(env)
env = MapState(env, lambda im: cv2.resize(im, (84,84)))
env = FrameStack(env, 4)

student = student_dqn(env)

episodes = 10000000
save_every = 1000
record_scores = 50

scores = []
mini_score = []

for ee in range(episodes):
    ob_big = env.reset()
    ob = ob_big[None,:,:,6:]/255.
    score_ = 0
    while True:
        action = student.act(ob)
        next_ob_big, reward, done, _ = env.step(action)
        next_ob = next_ob_big[None,:,:,6:]/255.
        score_ += reward
        student.build_memory(ob, action, reward, next_ob, done)

        ob = next_ob

        if done:
            mini_score.append(score_)

            if (ee+1) % record_scores == 0 or ee == 0:
                scores.append(np.mean(mini_score))
                print "Adding average score ({}).".format(np.mean(mini_score))
                mini_score = []

            if (ee+1) % save_every == 0 or ee == 0:
                student.model.save("lt_student_weights.h5")
                np.save("student_scores", np.array(scores))
                print "Saving after {} steps.".format(ee+1)
            break
            
    student.replay()
    student.update_target()

