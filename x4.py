'''
experiment : improve pendulum solution performance and stability
'''

import os
import pickle
from math import log

import numpy as np
import objgraph
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from keras import Input, Model, regularizers
from keras.layers import Dense, concatenate, Lambda

import gym
from callbacks import ActorCriticEval, PlotDist, PltQEval, PltObservation
from hindsight import HEREnvRollout
from rl import EnvRollout
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
    #x = Lambda(lambda x: x * 10)(x)
    critic = Model([oin, ain], x, name='critic')
    return actor,critic

def objective(kwargs):
    cluster = kwargs.pop('cluster')
    gamma=kwargs.get('gamma')
    reg=kwargs.pop('reg')
    epochs=kwargs.pop('epochs')

    actor, critic = make_models(cluster.env, reg=reg)
    agent = DDPGAgent(cluster, actor, critic, mode=2,**kwargs)

    # objgraph.show_growth(1)
    # m=cluster.rollout(policy=agent.target_actor, nepisodes=1)
    # objgraph.show_growth(100)

    eval=ActorCriticEval(cluster,agent.target_actor,agent.target_critic,gamma)
    callbacks=[]
    callbacks.append(eval)
    callbacks.append(PltQEval(cluster, gamma, [('target', agent.target_actor, agent.target_critic),
                                               ('ddpg', agent.actor, agent.critic)], title="RL eval",fignum=1))
    callbacks.append(PlotDist(cluster, eval.hist, title="actor/critic training trends",fignum=2))
    #callbacks.append(PltObservation(cluster, agent.target_actor, fignum=3))
    agent.train(epochs=epochs, fignum=1, visualize=True,callbacks=callbacks)

    reward=eval.hist['reward']
    r1=np.array(reward)
    n=min(int(r1.shape[0]*0.2),20) # last 20% or 20 epochs
    loss= -np.median(r1[-n:]) # median reward of the last nepochs
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
            'tau': hp.lognormal('tau', log(1e-2), 1),
            'reg':hp.lognormal('reg',log(1e-4),1),
            'lr':hp.lognormal('lr',log(1e-3),1),
            'clr': hp.lognormal('clr', log(1e-2), 1),
            'decay':hp.lognormal('decay',log(1e-6),1),
            'clip_tdq': hp.choice('clip_tdq',[None,-10]),
            'end_gamma': False,
            'critic_training_cycles': hp.randint('critic_training_cycles',10)+1,
            'batch_size': hp.randint('batch_size', 8)*32+32,
         }
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
            'critic_training_cycles': 10,
            'batch_size': 32,
            'nfrac':0.1,
         }
    # space = { #baseline parameters
    #         #'actor':actor,
    #         #'critic':critic,
    #         'cluster':cluster,
    #         'gamma':0.98,
    #         'epochs': 100,
    #         'tau': 0.02,
    #         'reg':1e-4,
    #         'lr':0.005,
    #         'clr': 0.05,
    #         'decay':0.0005,
    #     }
    # space = {
    #         #'actor':actor,
    #         #'critic':critic,
    #         'cluster':cluster,
    #         'gamma':0.98,
    #         'epochs': 5000,
    #         'tau': 0.002,
    #         'reg':1e-4,
    #         'lr':0.0005,
    #         'clr': 0.005,
    #         'decay':0.0005,
    #         'clip_tdq':True,
    #         'end_gamma':False,
    #         'critic_training_cycles':40,
    #         'batch_size':128,
    #     }
    # space = {  # camigord parameter settings
    #         #'actor':actor,
    #         #'critic':critic,
    #         'cluster':cluster,
    #         'gamma':0.98,
    #         'epochs': 100,
    #         'tau': 0.001,
    #         'reg':1e-4,
    #         'lr':0.0001,
    #         'clr': 0.001,
    #         'decay':0.0005,
    #     }


    movie = MoviePlot("RL", path='experiments/latest')
    movie.grab_on_pause(plt)
    best = fmin(objective,
        space=space,
        algo=tpe.suggest,
        max_evals=30,
        trials=trials)
    movie.finish()
    print(best)
    with open('trials.p','wb') as f:
        pickle.dump(trials,f)

if __name__ == "__main__":
    run()