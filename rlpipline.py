import os
import random
import time
from collections import namedtuple

import gym
import numpy as np
from keras import regularizers
from keras.callbacks import Callback
from keras.layers import Dense, Input, concatenate
from keras.models import Model, clone_model, load_model
from keras.optimizers import Adam
import keras.backend as K
from matplotlib import pyplot as plt
from rnr.keras import DDPGof,OrnstienUhlenbeckExplorer
from rnr.movieplot import MoviePlot
from rnr.util import kwargs
import rnr.gym

saveit=plt.pause
def pausewrapper(self,*args,**kwargs):
    global saveit
    print("before pause")
    saveit(self,*args,**kwargs)
    print("after pause")
plt.pause=pausewrapper

Experience = namedtuple('Experience', 'obs, action, reward, done')
def run():
    envname='Slider-v0'
    nenvs=64
    nsteps=1000000
    memsz=100000
    aepochs,cepochs,repochs=200,10,10000
    gamma=0.995
    tau=0.001
    envs=[gym.make(envname) for i_env in range(nenvs)]
    reload=False

    movie=MoviePlot({1:'RL'})
    for i,e in enumerate(envs):
        e.reset()

    actor,critic = make_models(envs[0])

    if reload and os.path.exists('actor.h5'):
        print("Load actor")
        actor = load_model('actor.h5')

    if reload and os.path.exists('critic.h5'):
        print("load critic")
        critic = load_model('critic.h5')

    if reload and os.path.exists('memory.p'):
        print("Load data")
        memory = Rmemory.load('memory.p')
    else:
        memory=Rmemory(sz=100000)
        if hasattr(envs[0],'controller'):
            print("pretrain actor using controller")
            actor_pretrain(envs,memory,actor,movie=movie,fignum=1,epochs=aepochs)
            actor.save('actor.h5')
        else:
            print("random policy rollout")
            explorers = [OrnstienUhlenbeckExplorer(e.action_space, theta=.5, mu=0., dt=1.0, ) for e in envs]
            batch_rollout(envs,actor,nsteps=memsz/nenvs,memory=memory,exploration=explorers,visualize=True)
        print("Save memory")
        memory.save('memory.p')

    if reload and os.path.exists('critic.h5'):
        print("load critic")
        critic = load_model('critic.h5')
    else:
        pretrain_critic(envs,actor,critic,gamma,cepochs,memory,movie=movie,fignum=1)
        print("Saving critic")
        critic.save('critic.h5')


    rltraining(envs, memory, actor, critic, gamma, tau, epochs=repochs,movie=movie,fignum=1,visualize=True)
    movie.finish()

def make_models(env):

    ain = Input(shape=env.action_space.shape, name='action')
    oin = Input(shape=env.observation_space.shape, name='observeration')  # full observation
    common_args = kwargs(kernel_initializer='glorot_normal',
                         activation='relu',
                         kernel_regularizer=regularizers.l2(1e-8)
                         # bias_initializer=Constant(value=0.1),
                         )
    x = oin
    #x = Dense(128, **common_args)(x)
    #x = Dense(128, **common_args)(x)
    #x = Dense(128, **common_args)(x)
    x = Dense(64, **common_args)(x)
    #x = Dense(32, **common_args)(x)
    #x = Dense(32, **common_args)(x)
    #x = Dense(32, **common_args)(x)
    x = Dense(32, **common_args)(x)
    x = Dense(env.action_space.shape[0], activation='linear')(x)
    actor = Model(oin, x, name='actor')

    common_args=kwargs( activation='relu',
                        kernel_initializer='glorot_normal',
                        kernel_regularizer=regularizers.l2(1e-6),
                        )
    x = concatenate([oin, ain], name='sensor_goal_action')
    x = Dense(128, **common_args)(x)
    #x = Dense(128, **common_args)(x)
    #x = Dense(128, **common_args)(x)
    x = Dense(64, **common_args)(x)
    #x = Dense(32, **common_args)(x)
    #x = Dense(32,**common_args)(x)
    #x = Dense(32, **common_args)(x)
    #x = Dense(32, **common_args)(x)
    x = Dense(32, **common_args)(x)
    x = Dense(32, **common_args)(x)
    x = Dense(1, activation='linear', name='Q')(x)
    critic = Model([oin, ain], x, name='critic')
    return actor,critic


def singlestep(e, a):
    obs, reward, done, info = e.step(a)
    if done:
        e.reset()
    return obs, reward, done, info

def rollout(e,policy=None,nsteps=1000,memory=None):
    obs = e.reset()
    onpolicy=(policy == None)
    if memory is None:
        memory=Rmemory()
    for i_env in range(nsteps):
        if policy is None:
            action=e.env.controller()
        else:
            action = policy.predict(np.expand_dims(obs, axis=0))[0]
        tobs, reward, done, _ = singlestep(e, action)
        memory.record(Experience(obs, action, reward, done), onpolicy=onpolicy)
        if done:
            break
        obs=tobs
    return memory

def batch_rollout(envs, policy=None, nsteps=1000, memory=None,exploration=None,visualize=False):
    bobs = []
    eidx=[]
    if memory is None:
        memory=Rmemory()
    for i_env, e in enumerate(envs):
        bobs.append(e.reset())
    for i_step in range(int(nsteps)):
        if policy is None:
            acts=[e.env.controller() for e in envs]
        else:
            acts = policy.predict(np.array(bobs))
        if exploration is not None:
            noises=[e.sample(a) for e,a in zip(exploration,acts)]
        else:
            noises=[None]*len(envs)

        for i_env, (action, e,noise) in enumerate(zip(acts, envs,noises)):
            #if i_env == 0: e.render()
            tobs,reward,done,_=singlestep(e,action)
            if visualize and i_env==0:
                e.render()
            if noise is not None:
                action += noise
            memory.record(Experience(bobs[i_env], action, reward, done), i_env,onpolicy=noise is None)
            if done:
                e.reset()
                if exploration is not None:
                    exploration[i_env].reset()
            bobs[i_env]=tobs
    return memory

MemEntry=namedtuple('MemEntry','data,metadata')
MetaData=namedtuple('MetaData','i_env,episode,step,prev')
class Rmemory():

    def __init__(self,sz=10000):
        self.sz=sz
        self.didx=0
        self.buffer=[]
        self.envstate={}
        self.i_env=0
        self.i_episode=0
        self.episodes=[]

    def record(self,e,i_env=0,onpolicy=None):
        if i_env in self.envstate:
            md=self.envstate[i_env]
        else:
            md= MetaData(i_env, self.i_episode, 0, -1) #env,episode,step,prev
            self.episodes.append(self.didx)
            self.i_episode+=1
        self.episodes[md.episode]=self.didx #update last entry in episode
        if self.didx==len(self.buffer):
            self.buffer.append(MemEntry(e, md))
        else:
            self.buffer[self.didx]=MemEntry(e, md)

        if e.done:
            if i_env in self.envstate:
                del self.envstate[i_env]
        else:
            self.envstate[i_env] = MetaData(i_env, md.episode, md.step + 1, self.didx)
        self.didx += 1
        if self.didx >= self.sz:
            self.didx = 0
        return md

    def reset(self,i_env):
        del self.envstate[i_env]

    def getk(self, idx,k=1):
        return self.buffer[idx]

    def sample(self,batchsz=1,k=1):
        while True:
            for i in range(batchsz):
                idx=random.randrange(0, len(self.buffer))

                for j in range(k):
                    pass

            sample=random.choice(range(len(self.buffer)), batchsz)

            Sobs = np.empty((batchsz,) + obs.shape)
            Sobs1 = np.empty((batchsz,) + obs.shape)
            Sreward = np.empty((batchsz,))
            Sdone = np.empty((batchsz,))
            for b in range(nsamples):
                samples = random.sample(range(memsz), batchsz)
                for idx, s in enumerate(samples):
                    Sobs[idx] = np.array(experience[s].state0)
                    Sobs1[idx] = np.array(experience[s].state1)
                    Sreward[idx] = np.array(experience[s].reward)
                    Sdone[idx] = np.array(experience[s].terminal1)
            for i in range(len(self.buffer)):
                yield self.buffer[idx]

    def sample_episode(self,episode_idx=None):
        if episode_idx is None:
            episode_idx=random.randrange(len(self.episodes))
        assert episode_idx<len(self.episodes)
        buffer_idx=self.episodes[episode_idx]
        episode=[buffer_idx]
        while True:
            prev=self.buffer[buffer_idx].metadata.prev
            if  prev == -1:
                break
            if buffer_idx>self.didx and prev<self.didx:
                break # spans fill point
            buffer_idx=prev
            episode.insert(0,buffer_idx)
        return self.np_experience(episode)

    def np_experience(self,idxs):
        batchsz=len(idxs)
        Sobs = np.empty((batchsz,) + self.buffer[0].data.obs.shape)
        Saction=np.empty((batchsz,) + self.buffer[0].data.action.shape)
        Sreward=np.empty((batchsz,1))
        Sdone=np.empty((batchsz,1))
        for idx, s in enumerate(idxs):
            Sobs[idx]=self.buffer[s].data.obs
            Saction[idx]=self.buffer[s].data.action
            Sreward[idx]=self.buffer[s].data.reward
            Sdone[idx]=self.buffer[s].data.done
        return Sobs,Saction,Sreward,Sdone

    def obsact(self):
        obs0=[]
        a0=[]
        for r in self.buffer:
            obs0.append(r.data.obs)
            a0.append(r.data.action)
        return np.array(obs0),np.array(a0)

    def obs1act(self):
        obs0=[]
        a0=[]
        r0=[]
        obs1=[]
        done=[]
        for idx,r in enumerate(self.buffer):
            if idx>=self.didx and r.metadata.prev < self.didx:
                continue # skip events that span the fill point
            # because it is easier to keep a previous pointer,
            # the last experience is sampled twice, the first is not sampled at all
            if r.data.done: # if it is the end, use this sample
                a0.append(r.data.action)
                r0.append(r.data.reward)
                obs0.append(r.data.obs)
                obs1.append(np.zeros_like(r.data.obs))
                done.append(r.data.done)
            if r.metadata.prev != -1: #sample is first in episode
                obs1.append(r.data.obs)
                pr=self.buffer[r.metadata.prev]
                a0.append(pr.data.action)
                r0.append(pr.data.reward)
                obs0.append(pr.data.obs)
                done.append(pr.data.done)
        return np.array(obs0),np.array(a0),np.array(r0),np.array(obs1),np.array(done)

    def save(self,filename):
        import pickle
        pickle.dump(self, open(filename, "wb"))

    def load(filename):
        import pickle
        return pickle.load( open(filename, "rb"))


class pltreplay_vs_policy(Callback):
    def __init__(self,memory,policy,title="",fignum=None):
        self.title=title
        self.fignum=plt.figure(fignum).number
        self.policy=policy
        self.memory=memory
    def on_epoch_end(self,epoch,logs):
        obs, act, reward, done = self.memory.sample_episode()
        pact = self.policy.predict(obs)
        plt.figure(self.fignum)
        plt.clf()
        plt.suptitle(self.title)
        plt.plot(act[:, 0], label='prior')
        plt.plot(pact[:, 0], label='current')
        plt.legend()
        plt.plot(act[:, 1])
        plt.plot(pact[:, 1])
        plt.legend()
        plt.pause(0.1)

class pltqd(Callback):
    def __init__(self,env,actor,critic,gamma,title="",fignum=None):
        self.title=title
        self.fignum=plt.figure(fignum).number
        self.actor=actor
        self.critic=critic
        self.gamma=gamma
        self.env=env
    def on_epoch_end(self,epoch,logs):
        memory=rollout(self.env,self.actor)
        obs, act, reward, done = memory.sample_episode()
        #aq=np.copy(reward)
        #last = 0
        #for i in reversed(range(aq.shape[0])):
            #aq[i] += gamma * last
            #last = aq[i]
        q = self.critic.predict([obs,act])
        plt.figure(self.fignum)
        plt.clf()
        plt.suptitle(self.title)
        plt.plot(reward[:, 0], label='reward')
        plt.plot(q[:-1, 0]-self.gamma*q[1:,0], label='critic delta')
        plt.legend()
        plt.pause(0.1)

class pltq(Callback):
    def __init__(self,env,actor,critic,gamma,title="",fignum=None):
        self.title=title
        self.fignum=plt.figure(fignum).number
        self.critic=critic
        self.gamma=gamma
        self.env=env
        self.actor=actor
        self.ag1=AutoGrowAxes()
        self.ag2=AutoGrowAxes()
        self.epoch=0


    def on_epoch_end(self,epoch,logs):
        memory=rollout(self.env,self.actor)
        obs, act, reward, done = memory.sample_episode()
        aq=np.copy(reward)
        last = 0
        for i in reversed(range(aq.shape[0])):
            aq[i] += self.gamma * last
            last = aq[i]
        q = self.critic.predict([obs,act])
        plt.figure(self.fignum)
        plt.clf()
        plt.suptitle(self.title + " epoch {}".format(self.epoch))
        self.epoch+=1

        plt.subplot(2,1,1)
        plt.xlim(0, self.env.spec.max_episode_steps)
        plt.ylim(*self.ag1.lim(aq[:, 0]))
        plt.plot(aq[:, 0], label='discounted reward')
        plt.plot(q[:, 0], label='critic')
        plt.legend(loc=1, fontsize='xx-small')
        plt.subplot(2,1,2)
        plt.xlim(0, self.env.spec.max_episode_steps)
        plt.ylim(*self.ag2.lim(reward[:, 0]))
        plt.plot(reward[:, 0], label='reward')
        plt.plot(q[:-1, 0]-self.gamma*q[1:,0], label='critic delta')
        plt.legend(loc=1, fontsize='xx-small')
        plt.pause(0.1)
class MovieSnap(Callback):
    def __init__(self,movie):
        super(MovieSnap,self).__init__()
        self.movie=movie
    def on_epoch_end(self, epoch, logs):
        if self.movie:
            self.movie.grab_frames()



class plteval(Callback):
    def __init__(self,env,series,title="",fignum=None,skip=10):
        self.title=title
        self.fignum=plt.figure(fignum).number
        self.env=env
        self.series = series
        self.epoch=0
        self.ag1=AutoGrowAxes()
        self.ag2=AutoGrowAxes()
        self.skip=skip



    def on_epoch_end(self,epoch,logs):
        self.epoch+=1
        if self.epoch % self.skip !=0: return
        plt.figure(self.fignum)
        plt.clf()
        plt.suptitle(self.title + " epoch {}".format(self.epoch))


        self.env.reset()
        self.env.env.locked=True # hack to use same starting point and goal for all rollouts
        for name,actor in self.series:
            memory=rollout(self.env,actor)
            obs, act, reward, done = memory.sample_episode()
            ar=np.copy(reward)
            for i in range(1,ar.shape[0]):
                ar[i] += ar[i-1]
            plt.subplot(2, 1, 1)
            plt.plot(ar[:, 0], label=name)
            plt.xlim(0,self.env.spec.max_episode_steps)
            plt.legend(loc=1,fontsize='xx-small')
            plt.subplot(2,1,2)
            plt.xlim(0,self.env.spec.max_episode_steps)
            for i in range(act.shape[1]):
                plt.plot(act[:, i], label=name)
            plt.legend(loc=1, fontsize='xx-small')
        self.env.env.locked = False
        plt.pause(0.1)

class pltqmap(Callback):
    def __init__(self,env,gamma,series,title="",fignum=None):
        self.title=title
        self.fignum=plt.figure(fignum).number
        self.series=series
        self.gamma=gamma
        self.env=env
        self.ag1=AutoGrowAxes()
        self.ag2=AutoGrowAxes()
        self.ag3=AutoGrowAxes()
        self.epoch=0

    def plot(self,obs0):
        fig = plt.figure(4)
        ax = plt.gca()
        plt.clf()
        fig.suptitle("Actions for obs{}, Episode {}".format(Config.viz_idx,i_episode))
        plt.axis(extent)
        sp = (1,nadim)
        for i in range(nadim):
            plt.subplot(*sp, i + 1)
            avmax = env.action_space.high[i]
            avmin = env.action_space.low[i]
            im = plt.imshow(A[:, :, i], cmap=plt.cm.RdBu_r, vmin=avmin, vmax=avmax, extent=extent)
            im.set_interpolation('bilinear')
            plt.scatter(x=-replay_buffer.obs[episodes[-1], 1], y=replay_buffer.obs[episodes[-1], 0], c='k',
                                vmin=avmin, vmax=avmax, s=0.5)
            plt.scatter(x=-replay_buffer.obs[episodes[-1][-1], 1], y=replay_buffer.obs[episodes[-1][-1], 0], c='green', s=6)
            cb = fig.colorbar(im)

class plt3(Callback):
    def __init__(self,env,gamma,series,title="",fignum=None):
        self.title=title
        self.fignum=plt.figure(fignum).number
        self.series=series
        self.gamma=gamma
        self.env=env
        self.ag1=AutoGrowAxes()
        self.ag2=AutoGrowAxes()
        self.ag3=AutoGrowAxes()
        self.epoch=0


    def on_epoch_end(self,epoch,logs):
        plt.figure(self.fignum)
        plt.clf()
        plt.suptitle(self.title+" epoch {}".format(self.epoch))
        self.epoch+=1
        self.env.reset()
        assert hasattr(self.env.env,'get_state') or len(self.series)==1
        if hasattr(self.env.env,'get_state'):
            start_state=self.env.env.get_state()
        for name,actor,critic in self.series:
            if hasattr(self.env.env, 'set_state'):
                self.env.env.set_state(start_state)
            if actor is None and not hasattr(actor,'controller'): continue
            memory=rollout(self.env,actor)
            obs, act, reward, done = memory.sample_episode()
            q=critic.predict([obs, act])
            ar=np.copy(reward)
            for i in range(1,ar.shape[0]):
                ar[i] += ar[i-1]
            aq = np.copy(reward)
            last = 0
            for i in reversed(range(aq.shape[0])):
                aq[i] += self.gamma * last
                last = aq[i]
            plt.subplot(3,1,1)
            plt.xlim(0,self.env.spec.max_episode_steps)
            plt.ylim(*self.ag1.lim(ar[:, 0]))
            plt.plot(ar[:, 0], label=name)
            plt.legend(loc=1,fontsize='xx-small')
            plt.subplot(3,1,2)
            plt.xlim(0,self.env.spec.max_episode_steps)
            plt.ylim(*self.ag2.lim(reward[:, 0]))
            plt.plot(reward[:, 0], label=name+' reward')
            plt.plot(q[:-1, 0] - self.gamma * q[1:, 0], label=name+' q delta')
            plt.legend(loc=1,fontsize='xx-small')
            plt.subplot(3,1,3)
            plt.xlim(0,self.env.spec.max_episode_steps)
            plt.ylim(*self.ag3.lim(aq[:, 0]))
            plt.plot(aq[:, 0], label=name+' dfr')
            plt.plot(q[:, 0], label=name+' q')
            plt.legend(loc=1,fontsize='xx-small')
        self.env.env.locked=False
        plt.pause(0.1)

def actor_pretrain(envs,memory,actor,nsteps=1000,epochs=100,movie=None,fignum=1):
    envname=envs[0].env.spec.id
    actor.compile(optimizer=Adam(lr=0.01, decay=1e-3), loss='mse', metrics=['mae', 'acc'])

    if False: # 6 time slower that batched
        start = time.perf_counter()
        for e in envs:
            obs=e.reset()
            for i_env in range(nsteps):
                a=actor.predict(np.expand_dims(obs,axis=0))[0]
                singlestep(e,a)

        end=time.perf_counter()
        print("Sequential thread {} {}".format(envname,end-start))

    if False: # use this for roll out
        start = time.perf_counter()
        batch_rollout(envs,actor,nsteps=nsteps)
        end=time.perf_counter()
        print("Batched thread {} {}".format(envname,end-start))


    batch_rollout(envs, memory=memory)
    start = time.perf_counter()
    end=time.perf_counter()
    print("Batched pretrain {} {}".format(envname,end-start))

    start = time.perf_counter()
    obs0,a0=memory.obsact()
    end=time.perf_counter()
    print("prepare {} {}".format(envname,end-start))

    ms=MovieSnap(movie)
    p1=plteval(envs[0], [("ctrl",None),('pretrained',actor)], title="Pretrain actor",fignum=fignum) # critic q vs actual dfr)# I want to compare controller roll out vs current policy rollout
    start = time.perf_counter()
    history = actor.fit(obs0, a0, shuffle=True, verbose=0, validation_split=0.1, epochs=epochs,callbacks=[p1,ms])
    end = time.perf_counter()
    print("Train actor {} {} epochs {}".format(envname,end-start,epochs))


def train_critic(critic, target_actor, target_critic, gamma, Rmem, callbacks):
    obs0, a0, r0, obs1,done = Rmem.obs1act()
    #q0 = target_critic.predict([obs0, a0])
    a1 = target_actor.predict([obs1])
    q1 = target_critic.predict([obs1, a1])
    n=gamma*q1*np.expand_dims(np.logical_not(done), axis=-1)
    qt = np.expand_dims(r0, axis=-1) + n
    history = critic.fit([obs0, a0], qt, shuffle=True, verbose=0, validation_split=0.1, epochs=1, callbacks=callbacks)
    return history

def create_target(model):
    # create target networks
    target_model = clone_model(model)
    target_model.compile(optimizer='sgd',loss='mse') # will not use optimizer or loss, but need compile
    target_model.set_weights(model.get_weights())
    return target_model

def update_target(target, model, tau):
    target.set_weights([tau * w + (1 - tau) * wp for wp, w in zip(target.get_weights(), model.get_weights())])

def train_ddpg_actor(actor, memory, callbacks):
    obs0, a0 = memory.obsact()
    actor.fit(obs0, np.zeros_like(a0), shuffle=True, verbose=2, validation_split=0.1, epochs=1, callbacks=callbacks)

def pretrain_critic(envs, actor, critic, gamma, cepochs, memory,movie=None,fignum=None):
    envname=envs[0].env.spec.id
    critic.compile(optimizer=Adam(lr=0.01, decay=1e-3), loss='mse', metrics=['mae', 'acc'])
    p1=pltq(envs[0],actor,critic,gamma,title="pretrain critic",fignum=fignum) # critic q vs actual dfr
    ms=MovieSnap(movie)
    for i in range(cepochs):
        start = time.perf_counter()
        history=train_critic(critic, actor, critic, gamma, memory, [p1,ms])
        end = time.perf_counter()
        print("Train critic {} {} epochs {}/{}".format(envname,end-start,i,cepochs))



def rltraining(envs, memory, actor, critic, gamma, tau, epochs=100,movie=None,fignum=None,visualize=False):
    envname=envs[0].env.spec.id
    mode=2
    if mode==1:
        actor.compile(optimizer=DDPGof(Adam)(critic, actor, lr=0.0000001), loss='mse', metrics=['mae', 'acc'])
    else:
        cgrad = K.gradients(critic.outputs, critic.inputs[1])  # grad of Q wrt actions
        fgrad = K.function(critic.inputs, cgrad)
        actor.compile(optimizer=Adam(lr=0.001), loss='mse', metrics=['mae', 'acc'])

    critic.compile(optimizer=Adam(lr=0.0001, decay=1e-3), loss='mse', metrics=['mae', 'acc'])
    pretrained_actor=create_target(actor)
    pretrained_critic=create_target(critic)
    target_actor=create_target(actor)
    target_critic=create_target(critic)

    # Each environment requires an explorer instance
    explorers = [ OrnstienUhlenbeckExplorer(e.action_space, theta=.5, mu=0., dt=1.0, ) for e in envs]

    p3=plt3(envs[0], gamma, [("ctrl",None,pretrained_critic),
                             ('target',target_actor,target_critic),('ddpg',actor,critic)], title="RL eval",fignum=fignum)

    ms=MovieSnap(movie)

    for i in range(epochs):
        astart = time.perf_counter()
        lr=K.get_value(critic.optimizer.lr)
        print("critic lr={}".format(lr))
        #K.set_value(actor.optimizer.lr,lr*0.99)
        chistory=train_critic(critic, target_actor, target_critic, gamma, memory, [p3,ms])
        aend = time.perf_counter()

        start = time.perf_counter()
        obs0, a0 = memory.obsact()
        if mode==1:
            actor.fit(obs0, np.zeros_like(a0), shuffle=True, verbose=2, validation_split=0.1, epochs=1,
                      callbacks=[p3])
        else:
            # update the actor : critic.grad()*actor.grad()
            actions = actor.predict(obs0)
            grads = fgrad([obs0, actions])[0]
            ya = actions + 0.1 * grads  # nudge action in direction that improves Q
            actor.fit(obs0, np.zeros_like(a0), shuffle=True, verbose=2, validation_split=0.1, epochs=1,
                      callbacks=[p3])

        end = time.perf_counter()
        print("RL Train {} {:0.3f},{:0.3f}  epochs {} of {}".format(envname, end - start, aend-astart, i, epochs))

        update_target(target_actor,actor,tau)
        update_target(target_critic,critic,tau)

        # rollout some data with latest policy
        batch_rollout(envs, policy=target_actor, nsteps=500, memory=memory, exploration=explorers,visualize=visualize)



class AutoGrowAxes():
    def __init__(self):
        self.limits=None

    def lim(self, data):
        ulimit=np.max(data)*1.1
        llimit=np.min(data)*1.1
        if self.limits is None:
            self.limits=(llimit,ulimit)
        elif  llimit<self.limits[0] or ulimit>self.limits[1]:
            self.limits=(min(self.limits[0], llimit), max(self.limits[1], ulimit))
        b=(self.limits[1]-self.limits[0])*0.1
        return self.limits[0]-b,self.limits[1]+b


if __name__ == "__main__":
    run()