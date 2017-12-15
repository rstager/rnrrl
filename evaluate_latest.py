import functools
import os

import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
import tensorflow as tf
from keras.utils.training_utils import multi_gpu_model

from callbacks import ActorCriticRollout, PltObservation, PltQEval2
from rl import EnvRollout
from x7 import make_env,ComplexSliderEnv

import gym
from rnr.gym import rnrenvs
import pickle

# kwargs = {'ndim': 3,'minx':-0.5,'maxx':0.5,'wrap':False}
rnrenvs()
# gym.envs.register(
#     id='Test-v0',
#     entry_point=functools.partial(make_env, ComplexSliderEnv, **kwargs),
#     max_episode_steps=200,
#     kwargs=kwargs
# )

#cluster = EnvRollout('Test-v0', 1)
#with open('envspec.p','wb') as f:
#   pickle.dump(cluster.env.spec,f)
os.chdir('experiments/latest')
with open('envspec.p', 'rb') as f:
    spec=pickle.load(f)

gym.envs.registry.env_specs['Test-v0']=spec

cluster = EnvRollout('Test-v0',1)

with tf.device("/cpu:0"):
    actor = load_model( 'actor.h5')
    critic = load_model('critic.h5')
    #model = multi_gpu_model(actor, gpus=1) #could be multiple gpus
plt.ion()
gamma = 0.9 # should come from experiments

rollout = ActorCriticRollout(cluster, actor, critic,
                             [PltObservation(cluster.env, fignum=2),
                              PltQEval2(cluster.env, gamma, fignum=3)],visualize=True)
for i_episode in range(100):
    with tf.device("/cpu:0"):
        rollout.on_epoch_end(i_episode,{})
