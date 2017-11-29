'''
Test slider with a complex reward function
'''
import functools

import gym
from gym import spaces
from rnr.slider import SliderEnv

'''
experiment : DDPGAgent validation test for slider.
Do not change.
Should converge in 30-50 epochs
'''

import os
import pickle
from math import log, cos, sin, sqrt

import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from keras import Input, Model, regularizers
from keras.layers import Dense, concatenate


from callbacks import ActorCriticEval, PlotDist, PltQEval, SaveModel
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
    callbacks.append(SaveModel(agent.target_actor,agent.target_critic,skip=10))
    memory = PrioritizedMemory(sz=100000,env=cluster.env,updater=qpriority)
    agent.train(memory=memory,epochs=epochs, fignum=1, visualize=False,callbacks=callbacks,nsteps=1000)

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

class ComplexSliderEnv(SliderEnv):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.ndim=kwargs.get('ndim',1)
        self.links=[1.0,0.7,0.3]
        self.observation_space= spaces.Box(low=np.array([0,0,-1,-1,-1,-1]),high=np.array([1,1,1,1,1,1]))

    def _angles_to_coord(self,angles):
        # simulate nservo reward
        xs = [0]
        ys = [0]
        ts = [np.pi / 2]  # zero is straight up
        for i, l in enumerate(self.links[:self.ndim]):
            ts.append(ts[-1] + angles[i])
            xs.append(xs[-1] + l * cos(ts[-1]))
            ys.append(ys[-1] + l * sin(ts[-1]))
        del ts[0]
        return [xs[-1],ys[-1],ts[-1]]

    def _step(self,u):
        obs,reward,done,info=super()._step(u)
        tip=self._angles_to_coord(obs[:self.ndim]*np.pi)
        newgoal=self._angles_to_coord(obs[2*self.ndim:3*self.ndim]*np.pi)
        assert self.ndim==2
        newobs=self._get_obs()
        #newreward=-2*(obs[0]-obs[4])**2-2*(obs[1]-obs[5])**2
        newreward= -2*((obs[4]*sin(obs[5])-obs[0]*sin(obs[1]))**2+(obs[4]*cos(obs[5])-obs[0]*cos(obs[1]))**2)
        newreward+=-0.1*(obs[2]**2 + obs[3]**2) -0.01*(u[0]**2+u[1]**2)-1
        #newreward=-2*(obs[0]-obs[4])**2-2*(obs[1]-obs[5])**2 -0.1*(obs[2]**2 + obs[3]**2) -0.01*(u[0]**2+u[1]**2)-1
        #newreward=reward
        #reward= -np.sum(error**2+0.1*self.v**2+0.01*u[0]**2) - 1.0


        #print("x/cos {} g {} {} reward {}".format(obs[0],obs[2],cos(obs[0]-obs[2]),reward))
        return newobs,newreward,done,info # original solution
        d2 = np.sum(np.square(np.array(tip[:self.ndim])-np.array(newgoal[:self.ndim])))
        new_reward = -d2 - 1
        new_reward /= 4*sum(self.links[:self.ndim])**2
        obs[2*self.ndim:3*self.ndim]=newgoal[:self.ndim]
        #reward = -((angles[0]-goalx[0])**2 +(angles[1]-goalx[1])**2 +(angles[2]-goalx[2])**2)
        #reward = -(sin(angles[0] - goalx[0])**2+sin(angles[1] - goalx[1]) ** 2 + sin(angles[2] - goalx[2]) ** 2)-1

        return obs,new_reward,done,info
    def _reset(self):
        super()._reset()
        return self._get_obs()

    def set_state(self,state):
        obs=super().set_state(state)
        return self._get_obs()

    def _get_obs(self):
        obs=super()._get_obs()
        newobs=np.array([obs[0],obs[1],obs[2],obs[3],obs[4]*sin(obs[5]),obs[4]*cos(obs[5])])
        return newobs

def make_env(cls,*args,**kwargs):
    return cls(*args,**kwargs)

def run():
    os.chdir('experiments/latest')
    rnrenvs()

    trials = Trials()
    kwargs={'ndim':2,'minx':0} #,'image_goal':[40,40]}

    gym.envs.register(
        id='Test-v0',
        entry_point=functools.partial(make_env,ComplexSliderEnv,**kwargs),
        max_episode_steps=200,
        kwargs=kwargs
    )
    cluster = EnvRollout('Test-v0', 1)

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