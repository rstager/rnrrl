'''
Test HER implementation
'''

import os
import pickle
from math import log

import numpy as np
import objgraph
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from keras import Input, Model, regularizers
from keras.layers import Dense, concatenate

import gym
from callbacks import ActorCriticEval, PlotDist, PltQEval, PltObservation
from hindsight import HEREnvRollout
from rl import EnvRollout
from rlagents import DDPGAgent
from rnr.util import kwargs


def make_models(env,reg=1e-4):
    ain = Input(shape=env.action_space.shape, name='action')
    oin = Input(shape=env.observation_space.shape, name='observeration')  # full observation
    common_args = kwargs(kernel_initializer='glorot_normal',
                         activation='relu',
                         kernel_regularizer=regularizers.l2(reg),
                         #bias_initializer = 'zeros',
                         )
    x = oin
    x = Dense(64, **common_args)(x)
    x = Dense(32, **common_args)(x)
    x = Dense(env.action_space.shape[0], activation='tanh')(x)
    actor = Model(oin, x, name='actor')

    x = oin
    x = concatenate([x, ain], name='sensor_goal_action')
    x = Dense(32, **common_args)(x)
    x = Dense(128, **common_args)(x)
    x = Dense(1, activation='linear', name='Q')(x)
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
    callbacks.append(PltObservation(cluster, agent.target_actor, fignum=3))
    agent.train(epochs=epochs, fignum=1, visualize=False,callbacks=callbacks)

    reward=np.array(eval.hist['reward'])
    n=min(int(reward.shape[0]*0.2),20) # last 20% or 20 epochs
    loss= -np.median(reward[-n:,:]) # median reward of the last nepochs
    print("loss={loss:.0f} gamma={gamma:.3f} tau={tau:.3e} lr={lr:.3e} clr={clr:.3e} decay={decay:.3e}".format(loss=loss,**kwargs))
    return {
        'loss': loss,
        'status': STATUS_OK,
        }

def run():
    os.chdir('experiments/latest')
    # register the environment here to make sure it is the same
    gym.envs.register(
        id='Pendulum-v100',
        entry_point='rnr.wrappedenvs:RestartablePendulumEnv',
        max_episode_steps=200,
    )
    gym.envs.register(
        id='PendulumHER-v0',
        entry_point='hindsight:PendulumHEREnv',
        max_episode_steps=200,
        kwargs={}
    )
    gym.envs.register(
        id='PendulumHER-v1',
        entry_point='hindsight:PendulumHEREnv',
        max_episode_steps=200,
        kwargs={'shapedreward':True}
    )

    trials = Trials()
    #cluster = EnvRollout('PendulumHER-v0', 64)
    cluster = HEREnvRollout('PendulumHER-v0', 64)
    actor,critic=make_models(cluster.env)
    # space = {
    #         'cluster':cluster,
    #         'gamma':0.98,
    #         'epochs': 1000,
    #         'tau': hp.lognormal('tau', log(1e-3), 2),
    #         'reg':hp.lognormal('reg',log(1e-4),2),
    #         'lr':hp.lognormal('lr',log(1e-4),2),
    #         'clr': hp.lognormal('clr', log(1e-3), 2),
    #         'decay':hp.lognormal('decay',log(1e-6),2),
    #     }
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
    space = {
            #'actor':actor,
            #'critic':critic,
            'cluster':cluster,
            'gamma':0.98,
            'epochs': 5000,
            'tau': 0.002,
            'reg':1e-4,
            'lr':0.0005,
            'clr': 0.005,
            'decay':0.0005,
            'clip_tdq':True,
            'end_gamma':False,
            'critic_training_cycles':40,
            'batch_size':128,
        }
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

    #movie = MoviePlot({3:"NRL"}, path='experiments/hoptest')
    #movie.grab_on_pause(plt)

    best = fmin(objective,
        space=space,
        algo=tpe.suggest,
        max_evals=30,
        trials=trials)

    print(best)
    with open('trials.p','wb') as f:
        pickle.dump(trials,f)

if __name__ == "__main__":
    run()