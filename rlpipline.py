import os

import time
from keras import regularizers
from keras.layers import Activation, Dense, Input, concatenate
from keras.models import Model, load_model
from matplotlib import pyplot as plt

from callbacks import ActorCriticEval, PlotDist, PltQEval, SaveModel
from rl import EnvRollout, ExperienceMemory
from rlagents import DDPGAgent
from rnr.gym import rnrenvs
from rnr.keras import OrnstienUhlenbeckExplorer
from rnr.movieplot import MoviePlot
from rnr.util import kwargs

import numpy as np

def make_models(env,reg=1e-5):

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
    x = Dense(env.action_space.shape[0], activation='linear')(x)
    actor = Model(oin, x, name='actor')

    x = oin
    x = concatenate([x, ain], name='sensor_goal_action')
    x = Dense(32, **common_args)(x)
    x = Dense(128, **common_args)(x)
    x = Dense(1, activation='linear', name='Q')(x)
    critic = Model([oin, ain], x, name='critic')
    return actor,critic


def run(cluster, gamma=0.995, tau=0.001, epochs=1000,lr=0.01,clr=0.01,decay=1e-6,reg=1e-4,
        nepisodes=None,nsteps=None,prefill=10000,actor=None,critic=None,
        make_model=make_models,agent=None):

    reload=False

    if actor is None or critic is None:
        actor,critic = make_model(cluster.env,reg=reg)
    if agent is None:
        agent=DDPGAgent(cluster,actor,critic,tau=tau,gamma=gamma,mode=2,lr=lr,clr=clr,decay=decay)

    if reload and os.path.exists('actor.h5'):
        print("Load actor")
        actor = load_model('actor.h5')

    if reload and os.path.exists('critic.h5'):
        print("load critic")
        critic = load_model('critic.h5')

    if reload and os.path.exists('memory.p'):
        print("Load data")
        memory = ExperienceMemory.load('memory.p')
    else:
        print("random policy envs")
        memory = ExperienceMemory(sz=1000000)
        explorers = [OrnstienUhlenbeckExplorer(cluster.env.action_space, theta=.5, mu=0., dt=1.0, nfrac=0.03) for i in range(cluster.nenv)]
        #cluster.rollout(actor, nsteps=prefill, memory=memory, exploration=explorers, visualize=False)
        print("Save memory")
        memory.save('memory.p')

    eval=ActorCriticEval(cluster,agent.target_actor,agent.target_critic,gamma)
    callbacks=[]
    callbacks.append(eval)
    callbacks.append(PltQEval(cluster, gamma, [('target', agent.target_actor, agent.target_critic),
                                               ('ddpg', agent.actor, agent.critic)], title="RL eval",fignum=1))
    callbacks.append(PlotDist(cluster, eval.hist, title="actor/critic training trends",fignum=2,skip=10))
    #callbacks.append(PltObservation(cluster, agent.target_actor))
    callbacks.append(SaveModel(agent.target_actor, agent.target_critic))
    memory=None
    agent.train(memory, epochs=epochs, fignum=1, visualize=False,callbacks=callbacks,nepisodes=nepisodes,nsteps=nsteps,
                minsteps=prefill)

    return eval.hist



if __name__ == "__main__":
    os.chdir('experiments/latest')
    rnrenvs()
    # envname='Slider-v0'
    #envname = 'Pendulum-v100'
    #envname = 'PendulumHER-v100'
    # envname='ContinuousMountainCart-v100'
    #cluster = HERRollout(envname, 1)
    #movie = MoviePlot("RL", path='experiments/latest')
    #movie.grab_on_pause(plt)
    if True:
        # this should converge in about 40 epochs
        envname='Pendulum-v100'
        cluster=EnvRollout(envname,64)
        #run(cluster, prefill=10000, gamma=0.95, tau=0.02,lr=0.01,clr=0.01,decay=1e-6,reg=1e-4,epochs=200)
        hist=run(cluster, prefill=10000, gamma=0.98, tau=0.02,lr=0.005,clr=0.05,decay=0.0005,reg=1e-4,epochs=200)
        reward = np.array(hist['reward'])
        loss = -np.median(reward[-min(int(reward.shape[0] * 0.2), 20):, :])#
        print ("loss = {}".format(loss))#  last 20% or 20 epochs
        time.sleep(10000)
    elif True:
        envname = 'NServoArm-v6'
        cluster = EnvRollout(envname, 64)
        run(cluster, prefill=50000, gamma=0.95, tau=0.02,epochs=100)
    #movie.finish()

