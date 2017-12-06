'''
Test slider with a complex reward function
'''
import functools

from tensorflow.python.layers.normalization import BatchNorm

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
from keras.layers import Dense, concatenate, Activation, BatchNormalization

from callbacks import ActorCriticEval, PlotDist, PltQEval, SaveModel, PltObservation, ActorCriticRollout, PltQEval2
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
                         #activation='relu',
                         #bias_initializer = 'zeros',
                         )
    x = oin
    x = Dense(64, **common_args)(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(128, **common_args, kernel_regularizer=regularizers.l2(reg))(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(128, **common_args, kernel_regularizer=regularizers.l2(reg))(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(env.action_space.shape[0], activation='tanh',kernel_initializer='glorot_normal')(x)
    actor = Model(oin, x, name='actor')

    x = oin
    x = concatenate([x, ain], name='sensor_goal_action')
    x = Dense(32, **common_args)(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(32, **common_args)(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(128, kernel_regularizer=regularizers.l2(reg),**common_args)(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
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
    # callbacks.append(PltQEval(cluster, gamma, [('target', agent.target_actor, agent.target_critic),
    #                                            ('ddpg', agent.actor, agent.critic)], title="RL eval",fignum=1))
    callbacks.append(PlotDist(cluster, eval.hist, title="actor/critic training trends",fignum=1))
    callbacks.append(ActorCriticRollout(cluster,agent.target_actor,agent.target_critic,
                                        [PltObservation(cluster.env, fignum=2),
                                         PltQEval2(cluster.env,gamma,fignum=3)]))
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
        ospace=2 if self.ndim!=3 else 3
        self.observation_space = spaces.Box(low=np.array([0]*self.ndim+[-1]*self.ndim+[-self.ndim]*ospace),
                                           high=np.array([1]*self.ndim+[1]*self.ndim+[self.ndim]*ospace))

    def _angles_to_coord(self,angles):
        # simulate nservo reward
        xs = [0]
        ys = [0]
        ts = [np.pi / 2]  # zero is straight up
        for i, l in enumerate(self.links[:self.ndim]):
            ts.append(ts[-1] + angles[i]*np.pi)
            xs.append(xs[-1] + l * cos(ts[-1]))
            ys.append(ys[-1] + l * sin(ts[-1]))
        del ts[0]
        if self.ndim==3:
            return np.array([xs[-1],ys[-1],ts[-1]])
        else:
            return np.array([xs[-1], ys[-1]])

    def _step(self,u):
        obs,reward,done,info=super()._step(u)
        obs=super()._get_obs()
        newobs=self._get_obs()
        #tip=self._angles_to_coord(obs[:self.ndim]*np.pi)
        #newgoal=self._angles_to_coord(obs[2*self.ndim:3*self.ndim]*np.pi)
        tip=self._angles_to_coord(obs[0:self.ndim])
        goal=self._angles_to_coord(obs[2 * self.ndim:3 * self.ndim])
        d=np.sqrt(np.sum(np.square(tip[:2]-goal[:2])))
        newreward= -0.5*(d**2)
        if self.ndim==3:
            newreward+= -0.1*(sin((tip[-1]-goal[-1])*np.pi)**2)
        # if self.hit_limit:
        #     newreward-=5
        if d<0.1: #override the done. There may be multiple solutions that get close enough to the goal.
            done=True
        if done:
            print("tip={}, goal={}, d={} obs {} newobs {}".format(tip,goal,newreward,obs,newobs))
        newreward+=-0.1*(np.sum(np.square(np.array(obs[self.ndim:self.ndim*2])))) -0.1*(np.sum(np.square(u)))-1
        return newobs,newreward,done,info # original solution

    def _reset(self):
        super()._reset()
        return self._get_obs()

    def set_state(self,state):
        super().set_state(state)
        return self._get_obs()

    def _get_obs(self):
        obs=super()._get_obs()
        newobs=np.array(list(obs[0:2*self.ndim])+list(self._angles_to_coord(obs[2*self.ndim:])))
        return newobs

    def _render(self, mode='human', close=False):
        if self.viewer is None:
            ret=super()._render(mode,close)
            sz=2
            self.viewer.set_bounds(-sz, sz, -sz, sz)
        if not close:
            obs = super()._get_obs()
            tip = self._angles_to_coord(obs[0:self.ndim])
            goal = self._angles_to_coord(obs[2 * self.ndim:3 * self.ndim])

            if self.ndim ==1:
                self.goal_transform.set_translation(goal[0],0)
                self.puck_transform.set_translation(tip[0],0)
            elif self.ndim >1:
                self.goal_transform.set_translation(goal[0], goal[1])
                self.puck_transform.set_translation(tip[0], tip[1])
            if self.ndim == 3:
                self.goal_transform.set_rotation(goal[2]*np.pi)
                self.puck_transform.set_rotation(tip[2]*np.pi)

            if self.done:
                self.goal.set_color(0,1, 0)
            else:
                self.goal.set_color(1,0, 0)

            return self.viewer.render(return_rgb_array = mode=='rgb_array')
        else:
            return ret


def make_env(cls,*args,**kwargs):
    return cls(*args,**kwargs)

def run():
    os.chdir('experiments/latest')
    rnrenvs()

    trials = Trials()
    kwargs={'ndim':2,'minx':0,'maxx':1,'wrap':False} #,'image_goal':[40,40]}

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
            'epochs': 2000,
            'tau': 1e-2,
            'reg':1e-4,
            'lr':1e-3,
            'clr':1e-2,
            'decay':1e-6,
            'clip_tdq': -90,
            'end_gamma': False,
            'critic_training_cycles':5,
            'batch_size': 256,
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