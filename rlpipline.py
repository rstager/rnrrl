import os
import time

import keras.backend as K
import numpy as np
from keras import regularizers
from keras.callbacks import Callback
from keras.layers import Dense, Input, concatenate, BatchNormalization
from keras.models import Model, load_model
from keras.optimizers import Adam
from matplotlib import pyplot as plt

from callbacks import ActorCriticEval, PlotDist
from plotutils import AutoGrowAxes
from rl import EnvRollout, ExperienceMemory, TD_q
from rlagents import DDPGAgent
from rnr.keras import OrnstienUhlenbeckExplorer
from rnr.movieplot import MoviePlot
from rnr.util import kwargs
from rnr.gym import rnrenvs


def run(cluster, gamma=0.995, tau=0.001, prefill=10000, aepochs=0, cepochs=0, repochs=1000):

    reload=False
    xdir='experiments/latest'

    movie=MoviePlot("RL",path=xdir)
    movie.grab_on_pause(plt)

    actor,critic = make_models(cluster.env)

    if reload and os.path.exists(os.path.join(xdir,'actor.h5')):
        print("Load actor")
        actor = load_model(os.path.join(xdir,'actor.h5'))

    if reload and os.path.exists(os.path.join(xdir,'critic.h5')):
        print("load critic")
        critic = load_model(os.path.join(xdir,'critic.h5'))

    if reload and os.path.exists(os.path.join(xdir,'memory.p')):
        print("Load data")
        memory = ExperienceMemory.load(os.path.join(xdir,'memory.p'))
    else:
        memory=ExperienceMemory(sz=1000000)
        if hasattr(cluster._instances[0], 'controller'):
            print("pretrain actor using controller")
            actor_pretrain(cluster,memory,actor,fignum=1,epochs=aepochs)
            actor.save(os.path.join(xdir,'actor.h5'))
        else:
            print("random policy envs")
            explorers = [OrnstienUhlenbeckExplorer(cluster.env.action_space, theta=.5, mu=0., dt=1.0, ) for i in range(cluster.nenv)]
            cluster.rollout(actor, nsteps=prefill, memory=memory, exploration=explorers, visualize=False)
        print("Save memory")
        memory.save(os.path.join(xdir,'memory.p'))

    if reload and os.path.exists('critic.h5'):
        print("load critic")
        critic = load_model(os.path.join(xdir,'critic.h5'))
    else:
        pretrain_critic(cluster,actor,critic,gamma,cepochs,memory,fignum=1)
        print("Saving critic")
        critic.save(os.path.join(xdir,'critic.h5'))
    agent=DDPGAgent(cluster,actor,critic,tau,gamma,mode=2,lr=0.01,decay=1e-4)
    eval=ActorCriticEval(cluster,agent.target_actor,agent.target_critic,gamma)
    callbacks=[]
    callbacks.append(eval)
    callbacks.append(PltQEval(cluster, gamma, [('target', agent.target_actor, agent.target_critic),
                                               ('ddpg', agent.actor, agent.critic)], title="RL eval",fignum=1))
    callbacks.append(PlotDist(cluster, eval.hist, title="actor/critic training trends"))

    agent.train(memory, epochs=repochs, fignum=1, visualize=False,callbacks=callbacks)
    movie.finish()

def make_models(env):

    ain = Input(shape=env.action_space.shape, name='action')
    oin = Input(shape=env.observation_space.shape, name='observeration')  # full observation
    common_args = kwargs(kernel_initializer='glorot_normal',
                         activation='relu',
                         kernel_regularizer=regularizers.l2(1e-5)
                         # bias_initializer=Constant(value=0.1),
                         )
    x = oin
    x = Dense(64, **common_args)(x)
    x = BatchNormalization()(x)
    x = Dense(32, **common_args)(x)
    x = Dense(env.action_space.shape[0], activation='linear')(x)
    actor = Model(oin, x, name='actor')

    common_args=kwargs( activation='relu',
                        kernel_initializer='glorot_normal',
                        kernel_regularizer=regularizers.l2(1e-6),
                        )
    x = oin
    x = concatenate([x, ain], name='sensor_goal_action')
    x = Dense(32, **common_args)(x)
    #x = BatchNormalization()(x)
    x = Dense(128, **common_args)(x)
    x = Dense(1, activation='linear', name='Q')(x)
    critic = Model([oin, ain], x, name='critic')
    return actor,critic







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
    def __init__(self,cluster,actor,critic,gamma,title="",fignum=None):
        self.title=title
        self.fignum=plt.figure(fignum).number
        self.actor=actor
        self.critic=critic
        self.gamma=gamma
        self.cluster=cluster
    def on_epoch_end(self,epoch,logs):
        memory=self.cluster.rollout(self.actor,nepisodes=1)
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
    def __init__(self,cluster,actor,critic,gamma,title="",fignum=None):
        self.title=title
        self.fignum=plt.figure(fignum).number
        self.critic=critic
        self.gamma=gamma
        self.cluster=cluster
        self.actor=actor
        self.ag1=AutoGrowAxes()
        self.ag2=AutoGrowAxes()

    def on_epoch_end(self,epoch,logs):
        memory=self.cluster.rollout(self.actor,nepisodes=1)
        obs, act, reward, done = memory.sample_episode()
        aq=dfr(reward,self.gamma)
        q = self.critic.predict([obs,act])
        plt.figure(self.fignum)
        plt.clf()
        plt.suptitle(self.title + " epoch {}".format(epoch))


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


class plteval(Callback):
    def __init__(self,cluster,series,title="",fignum=None,skip=10):
        self.title=title
        self.fignum=plt.figure(fignum).number
        self.cluster=cluster
        self.series = series
        self.ag1=AutoGrowAxes()
        self.ag2=AutoGrowAxes()
        self.skip=skip

    def on_epoch_end(self,epoch,logs):
        plt.figure(self.fignum)
        plt.clf()
        plt.suptitle(self.title + " epoch {}".format(epoch))

        self.cluster.env.reset()
        state=self.cluster.env.get_state()
        for name,actor in self.series:
            self.cluster.env.set_state(state)
            memory = self.cluster.rollout(self.actor, nepisodes=1)
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
        plt.pause(0.1)



class PltQEval(Callback):
    def __init__(self,cluster,gamma,series,title="",fignum=None):
        self.title=title
        self.fignum=plt.figure(fignum).number
        self.series=series
        self.gamma=gamma
        self.cluster=cluster
        self.ag1=AutoGrowAxes()
        self.ag2=AutoGrowAxes()
        self.ag3=AutoGrowAxes()

    def on_epoch_end(self,epoch,logs):
        plt.figure(self.fignum)
        plt.clf()
        plt.suptitle(self.title+" epoch {}".format(epoch))
        self.cluster.env.env.reset()
        assert hasattr(self.cluster.env.env,'get_state'),"plt3 requires environment have get/set_state"
        start_state=self.cluster.env.env.get_state()
        for idx,(name,actor,critic) in enumerate(self.series):
            if actor is None and not hasattr(actor,'controller'): continue
            memory = self.cluster.rollout(actor, nepisodes=1,state=start_state)
            obs0, a0, r0, done = memory.sample_episode()
            q=critic.predict([obs0, a0])
            ar=np.copy(r0)
            for i in range(1,ar.shape[0]):
                ar[i] += ar[i-1]
            aq = np.copy(r0)
            last = 0
            for i in reversed(range(aq.shape[0])):
                aq[i] += self.gamma * last
                last = aq[i]
            tdq = TD_q(actor, critic, self.gamma, np.vstack([obs0[1:],np.zeros_like(obs0[0])]), r0, done)

            plt.subplot(3,1,1)
            plt.xlim(0,self.cluster.env.spec.max_episode_steps)
            plt.ylim(*self.ag1.lim(ar[:, 0]))
            #print("plt3 {} len {} ar {}:{}".format(name,r0.shape[0], np.min(ar), np.max(ar)))
            plt.plot(ar[:, 0], label=name+" policy")
            plt.legend(loc=1,fontsize='xx-small')
            if idx==0:
                plt.subplot(3,1,2)
                plt.xlim(0,self.cluster.env.spec.max_episode_steps)
                plt.ylim(*self.ag2.lim(r0[:, 0]))
                plt.plot(r0[:, 0], label=' reward')
                plt.plot(q[:-1, 0] - self.gamma * q[1:, 0], label=' q delta')
                plt.legend(loc=4,fontsize='xx-small')
                plt.subplot(3,1,3)
                plt.xlim(0,self.cluster.env.spec.max_episode_steps)
                plt.ylim(*self.ag3.lim(aq[:, 0]))
                plt.plot(aq[:, 0], label=' dfr')
                plt.plot(q[:, 0], label=' q')
                plt.plot(tdq[:, 0], label=' tdq')
                plt.legend(loc=4,fontsize='xx-small')
        plt.pause(0.1)




def actor_pretrain(cluster,memory,actor,nsteps=1000,epochs=100,fignum=1):
    actor.compile(optimizer=Adam(lr=0.01, decay=1e-3), loss='mse', metrics=['mae', 'acc'])
    cluster.rollout(memory=memory)
    start = time.perf_counter()
    end=time.perf_counter()
    print("Batched pretrain {} {}".format(cluster.envname,end-start))

    start = time.perf_counter()
    obs0,a0=memory.obsact()
    end=time.perf_counter()
    print("prepare {} {}".format(cluster.envname,end-start))


    p1=plteval(cluster, [("ctrl",None),('pretrained',actor)], title="Pretrain actor",fignum=fignum) # critic q vs actual dfr)# I want to compare controller roll out vs current policy rollout
    start = time.perf_counter()
    history = actor.fit(obs0, a0, shuffle=True, verbose=0, validation_split=0.1, epochs=epochs,callbacks=[p1])
    end = time.perf_counter()
    print("Train actor {} {} epochs {}".format(cluster.envname,end-start,epochs))




def train_critic(critic, target_actor, target_critic, gamma, generator, callbacks,batch_size=32):
    obs0, a0, r0, obs1,done = generator.next()
    qt = TD_q(target_actor, target_critic, gamma, obs1, r0, done)
    history = critic.train_on_batch([obs0, a0], qt, verbose=0)
    for c in callbacks:
        c.on_batch_end(epoch,{})
    return history

def train_ddpg_actor(actor, memory, callbacks):
    obs0, a0 = memory.obsact()
    actor.fit(obs0, np.zeros_like(a0), shuffle=True, verbose=2, validation_split=0.1, epochs=1, callbacks=callbacks)

def pretrain_critic(cluster, actor, critic, gamma, cepochs, memory,fignum=None):
    critic.compile(optimizer=Adam(lr=0.01, decay=1e-3), loss='mse', metrics=['mae', 'acc'])
    p1=pltq(cluster,actor,critic,gamma,title="pretrain critic",fignum=fignum) # critic q vs actual dfr
    for i in range(cepochs):
        start = time.perf_counter()
        history=train_critic(critic, actor, critic, gamma, memory, [p1])
        end = time.perf_counter()
        print("Train critic {} {} epochs {}/{}".format(cluster.env.spec.id,end-start,i,cepochs))






#an implementation of
    def possible_decay_schedule(self,explorers):
        if self.mode == 2:
            alr = K.get_value(self.combined.optimizer.lr)
            aiters = K.get_value(self.combined.optimizer.iterations)
        else:
            alr = K.get_value(self.actor.optimizer.lr)
            aiters = K.get_value(self.actor.optimizer.iterations)
        clr = K.get_value(self.critic.optimizer.lr)
        if self.epoch % self.lrdecay_interval == 0:
            alr *= 0.5
            if self.mode == 2:
                K.set_value(self.combined.optimizer.lr, alr)
            else:
                K.set_value(self.actor.optimizer.lr, alr)
            clr *= 0.5
            K.set_value(self.critic.optimizer.lr, clr)
            for x in explorers:
                x.decay(0.5)
        citers = K.get_value(self.critic.optimizer.iterations)
        print("lr={} {} iters= a {} c {}".format(clr,alr,aiters,citers))



        return memory



if __name__ == "__main__":
    rnrenvs()
    # envname='Slider-v0'
    envname = 'Pendulum-v100'
    # envname = 'PendulumHER-v100'
    # envname='NServoArm-v6'
    # envname='ContinuousMountainCart-v100'
    # cluster = HERRollout(envname, nenvs)
    cluster=EnvRollout(envname,64)
    run(cluster, prefill=10000, gamma=0.95, tau=0.02)
