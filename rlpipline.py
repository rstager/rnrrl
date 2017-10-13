import gym
import threading
import numpy as np
from collections import deque, namedtuple,defaultdict
import random
import time
from queue import Queue
from util import Counter
from multiprocessing import Process,Pool,Queue
import rnrlgym
from keras.optimizers import Adam
from keras.layers import Dense,Input,concatenate
from keras.models import Model
from keras.callbacks import Callback
from keras.models import load_model,clone_model
from keras import regularizers
from util import kwargs
from keras_util import DDPGof
from matplotlib import pyplot as plt
import os
from collections import Iterable
from movieplot import MoviePlot


Experience = namedtuple('Experience', 'obs, action, reward, done')
def run():
    envname='NServoArm-v0'
    nenvs=64
    nsteps=100000
    aepochs,cepochs,repochs=200,200,10000
    gamma=0.95
    tau=0.0001
    envs=[gym.make(envname) for i_env in range(nenvs)]
    reload=False

    movie=MoviePlot({1:'RL'})
    for i,e in enumerate(envs):
        e.reset()
    if reload and os.path.exists('memory.p'):
        print("Load actor and data")
        actor = load_model('actor.h5')
        pretrain_memory = Rmemory.load('memory.p')
    else:
        actor, critic = make_models(envs[0])
        pretrain_memory=Rmemory()
        actor_pretrain(envs,pretrain_memory,actor,movie=movie,fignum=1,epochs=aepochs)
        print("Save actor")
        actor.save('actor.h5')
        pretrain_memory.save('memory.p')

    if reload and os.path.exists('critic.h5'):
        print("load critic")
        critic = load_model('critic.h5')
    else:
        _, critic = make_models(envs[0])
        pretrain_critic(envs,actor,critic,gamma,cepochs,pretrain_memory,movie=movie,fignum=1)
        print("Saving critic")
        critic.save('critic.h5')


    rltraining(envs, pretrain_memory, actor, critic, gamma, tau, epochs=repochs,movie=movie,fignum=1)

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
                        #kernel_regularizer=regularizers.l2(1e-9),
                        )
    x = concatenate([oin, ain], name='sensor_goal_action')
    #x = Dense(128, **common_args)(x)
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

def batch_rollout(envs, policy=None, nsteps=1000, memory=None):
    bobs = []
    eidx=[]
    if memory is None:
        memory=Rmemory()
    for i_env, e in enumerate(envs):
        bobs.append(e.reset())
    for i_env in range(nsteps):
        if policy is None:
            acts=[e.env.controller() for e in envs]
        else:
            acts = policy.predict(np.array(bobs))
        for i_env, (action, e) in enumerate(zip(acts, envs)):
            #if i_env == 0: e.render()
            tobs,reward,done,_=singlestep(e,action)
            onpolicy=True
            memory.record(Experience(bobs[i_env], action, reward, done), i_env,onpolicy=True)
            if done:
                e.reset()
            bobs[i_env]=tobs
    return memory

MemEntry=namedtuple('MemEntry','data,metadata')
MetaData=namedtuple('MetaData','i_env,episode,step,prev')
class Rmemory():

    def __init__(self,sz=1000):
        self.sz=sz
        self.didx=0
        self.buffer=[]
        self.envstate={}
        self.i_env=0
        self.i_episode=0
        self.episodes=[]

    def record(self,e,i_env=0,onpolicy=None):
        bidx=len(self.buffer)
        if i_env in self.envstate:
            md=self.envstate[i_env]
        else:
            md= MetaData(i_env, self.i_episode, 0, -1) #env,episode,step,prev
            self.episodes.append(bidx)
            self.i_episode+=1
        self.episodes[md.episode]=bidx #update last entry in episode
        self.buffer.append(MemEntry(e, md))
        if e.done:
            del self.envstate[i_env]
        else:
            self.envstate[i_env] = MetaData(i_env, md.episode, md.step + 1, bidx)
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
            buffer_idx=self.buffer[buffer_idx].metadata.prev
            if  buffer_idx == -1:
                break
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
        for r in self.buffer:
            prev=r.metadata.prev
            if prev != -1:
                obs1.append(r.data.obs)
                pr=self.buffer[prev]
                a0.append(pr.data.action)
                r0.append(pr.data.reward)
                obs0.append(pr.data.obs)
        return np.array(obs0),np.array(a0),np.array(r0),np.array(obs1)

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
        plt.suptitle(self.title)
        plt.subplot(2,1,1)
        plt.plot(aq[:, 0], label='discounted reward')
        plt.plot(q[:, 0], label='critic')
        plt.legend(loc=1, fontsize='xx-small')
        plt.subplot(2,1,2)
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
    #def on_traing_end(self, epoch, logs):
        #self.movie.finish()

class plteval(Callback):
    def __init__(self,env,series,title="",fignum=None):
        self.title=title
        self.fignum=plt.figure(fignum).number
        self.env=env
        self.series = series


    def on_epoch_end(self,epoch,logs):
        plt.figure(self.fignum)
        plt.clf()
        plt.suptitle(self.title)

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
            plt.legend(loc=1,fontsize='xx-small')
            plt.subplot(2,1,2)
            for i in range(act.shape[1]):
                plt.plot(act[:, i], label=name)
            plt.legend(loc=1, fontsize='xx-small')
        self.env.env.locked = False
        plt.pause(0.1)

class plt3(Callback):
    def __init__(self,env,gamma,series,title="",fignum=None):
        self.title=title
        self.fignum=plt.figure(fignum).number
        self.series=series
        self.gamma=gamma
        self.env=env

    def on_epoch_end(self,epoch,logs):
        plt.figure(self.fignum)
        plt.clf()
        plt.suptitle(self.title)
        self.env.reset()
        self.env.env.locked=True # hack to use same starting point and goal for all rollouts
        for name,actor,critic in self.series:
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
            plt.plot(ar[:, 0], label=name)
            plt.legend(loc=1,fontsize='xx-small')
            plt.subplot(3,1,2)
            plt.plot(reward[:, 0], label=name+' reward')
            plt.plot(q[:-1, 0] - self.gamma * q[1:, 0], label=name+' q delta')
            plt.legend(loc=1,fontsize='xx-small')
            plt.subplot(3,1,3)
            plt.plot(aq[:, 0], label=name+' dfr')
            plt.plot(q[:, 0], label=name+' q')
            plt.legend(loc=1,fontsize='xx-small')
        self.env.env.locked=False
        plt.pause(0.1)

def actor_pretrain(envs,memory,actor,nsteps=1000,epochs=100,movie=None,fignum=1):
    envname=envs[0].env.spec.id
    actor.compile(optimizer=Adam(lr=0.001, decay=1e-3), loss='mse', metrics=['mae', 'acc'])

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
    obs0, a0, r0, obs1 = Rmem.obs1act()
    q0 = target_critic.predict([obs0, a0])
    a1 = target_actor.predict([obs1])
    q1 = target_critic.predict([obs1, a1])
    qt = np.expand_dims(r0, axis=-1) + gamma * q1
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
    critic.compile(optimizer=Adam(lr=0.001, decay=1e-3), loss='mse', metrics=['mae', 'acc'])
    p1=pltq(envs[0],actor,critic,gamma,title="pretrain critic",fignum=fignum) # critic q vs actual dfr
    ms=MovieSnap(movie)
    for i in range(cepochs):
        start = time.perf_counter()
        history=train_critic(critic, actor, critic, gamma, memory, [p1,ms])
        end = time.perf_counter()
        print("Train critic {} {} epochs {}".format(envname,end-start,cepochs))



def rltraining(envs, memory, actor, critic, gamma, tau, epochs=100,movie=None,fignum=None):
    envname=envs[0].env.spec.id
    actor.compile(optimizer=DDPGof(Adam)(critic, actor, lr=0.0000001), loss='mse', metrics=['mae', 'acc'])
    critic.compile(optimizer=Adam(lr=0.0001, decay=1e-3), loss='mse', metrics=['mae', 'acc'])
    pretrained_actor=create_target(actor)
    pretrained_critic=create_target(critic)
    target_actor=create_target(actor)
    target_critic=create_target(critic)

    p3=plt3(envs[0], gamma, [("ctrl",None,pretrained_critic),('pretrained',pretrained_actor,pretrained_critic)
                             ,('target',target_actor,target_critic),('ddpg',actor,critic)], title="RL eval",fignum=fignum)

    ms=MovieSnap(movie)

    for i in range(epochs):
        astart = time.perf_counter()
        chistory=train_critic(critic, target_actor, target_critic, gamma, memory, [p3,ms])
        aend = time.perf_counter()

        start = time.perf_counter()
        ahistory=train_ddpg_actor(actor, memory, [p3])
        end = time.perf_counter()
        print("RL Train {} {:0.3f},{:0.3f}  epochs {} of {}".format(envname, end - start, aend-astart, i, epochs))

        update_target(target_actor,actor,tau)
        update_target(target_critic,critic,tau)
    movie.finish()
if __name__ == "__main__":
    run()