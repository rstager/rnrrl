import os

import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

import gym
from rnr.gym import rnrenvs

rnrenvs()
env = gym.make('NServoArm-v6')
actor = load_model(os.path.join('experiments/latest', 'actor.h5'))
plt.ion()
for i_episode in range(10):
    observation = env.reset()
    for t in range(2000):
        env.render()
        action = actor.predict(np.expand_dims(observation, axis=0))[0]
        observation, reward, done, info = env.step(action)
        if done:
            break
