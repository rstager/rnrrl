'''
experiment : DDPGAgent validation test for slider.
Do not change.
Should converge in 30-50 epochs
'''

import os
import pickle
from math import log

import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from keras import Input, Model, regularizers
from keras.layers import Dense, concatenate


from callbacks import ActorCriticEval, PlotDist, PltQEval
from rl import EnvRollout, PrioritizedMemory, TD_q
from rlagents import DDPGAgent
from rnr.gym import rnrenvs
from rnr.movieplot import MoviePlot
from rnr.util import kwargs

import matplotlib.pyplot as plt

def make_models(env,reg=1e-4):
    ain = Input(shape=env.action_space.shape, name='action')
    oin = Input(shape=env.observation_space.shape, name='observeration')  # full observation
    common_args = kwargs(kernel_initializer='glorot_normal',
                         activation='relu',
                         #bias_initializer = 'zeros',
                         )
    x = oin
    x = Dense(64, **common_args)(x)
    x = Dense(128, **common_args, kernel_regularizer=regularizers.l2(reg))(x)
    x = Dense(128, **common_args, kernel_regularizer=regularizers.l2(reg))(x)
    x = Dense(env.action_space.shape[0], activation='tanh',kernel_initializer='glorot_normal')(x)
    actor = Model(oin, x, name='actor')

    x = oin
    x = concatenate([x, ain], name='sensor_goal_action')
    x = Dense(32, **common_args)(x)
    x = Dense(128, kernel_regularizer=regularizers.l2(reg),**common_args)(x)
    x = Dense(1, activation='linear', name='Q')(x)
    critic = Model([oin, ain], x, name='critic')
    return actor,critic

def objective(kwargs):
    cluster = kwargs.pop('cluster')
    gamma=kwargs.get('gamma')
    reg=kwargs.pop('reg')
    epochs=kwargs.pop('epochs')
    clip_tdq=kwargs.get('clip_tdq',False)

    actor, critic = make_models(cluster.env, reg=reg)
    agent = DDPGAgent(cluster, actor, critic, mode=2,**kwargs)

    def qpriority(obs0, a0, r0, obs1, done):
        tdq = TD_q(agent.target_actor, agent.target_critic, agent.gamma, obs1, r0, done)
        q0 = agent.target_critic.predict([obs0,a0])
        if clip_tdq is not None:
            tdq = np.clip(tdq, clip_tdq / (1 - gamma), 0)
        epsilon=0.00001
        priority=np.abs((q0-tdq)/(tdq+epsilon))
        priority=np.clip(priority,0.0001,1)
        return priority.squeeze(axis=-1)

    eval=ActorCriticEval(cluster,agent.target_actor,agent.target_critic,gamma)
    callbacks=[]
    callbacks.append(eval)
    callbacks.append(PltQEval(cluster, gamma, [('target', agent.target_actor, agent.target_critic),
                                               ('ddpg', agent.actor, agent.critic)], title="RL eval",fignum=1))
    callbacks.append(PlotDist(cluster, eval.hist, title="actor/critic training trends",fignum=2))
    memory = PrioritizedMemory(sz=1000000,updater=qpriority)
    agent.train(memory=memory,epochs=epochs, fignum=1, visualize=False,callbacks=callbacks)

    reward=eval.hist['reward']
    r1=np.array(reward)
    n=min(int(r1.shape[0]*0.2),20) # last 20% or 20 epochs
    if not np.isnan(r1).any():
        loss= -np.median(r1[-n:]) # median reward of the last nepochs
    else:
        loss=-np.inf
    print("loss={loss:.0f} gamma={gamma:.3f} tau={tau:.3e} lr={lr:.3e} clr={clr:.3e} decay={decay:.3e}".format(loss=loss,**kwargs))
    return {
        'loss': loss,
        'status': STATUS_OK,
        }

def run():
    os.chdir('experiments/latest')
    rnrenvs()

    trials = Trials()
    cluster = EnvRollout('Slider-v3', 64)

    space = {
            'cluster':cluster,
            'gamma':0.9,
            'epochs': 200,
            'tau': 1e-2,
            'reg':1e-4,
            'lr':1e-3,
            'clr':1e-2,
            'decay':1e-6,
            'clip_tdq': -10,
            'end_gamma': False,
            'critic_training_cycles': 40,
            'batch_size': 32,
            'nfrac':0.1,
         }


    movie = MoviePlot("RL", path='experiments/latest')
    movie.grab_on_pause(plt)
    best = fmin(objective,
        space=space,
        algo=tpe.suggest,
        max_evals=30,
        trials=trials)
    movie.finish()
    print(best)

if __name__ == "__main__":
    run()