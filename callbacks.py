from collections import OrderedDict
import os

import time
from keras.callbacks import Callback

from plotutils import AutoGrowAxes, plotdict, plotmeanstd
from rl import ExperienceMemory, discounted_future, TD_q
import numpy as np

from rnr.util import rms
from matplotlib import pyplot as plt

class ActorCriticEval(Callback):
    def __init__(self,cluster,actor,critic,gamma,visualize=False,nepisodes=64):
        self.cluster=cluster
        self.actor=actor
        self.critic=critic
        self.gamma=gamma
        self.visualize=visualize
        self.nepisodes=nepisodes
        self.hist=OrderedDict()
        self.hist["reward"]=[]
        self.hist["qvar"]=[]
        self.hist["qbias"]=[]

    def on_epoch_end(self, epoch, logs):
        # roll out some on policy episodes so we can see the results
        evalmemory = ExperienceMemory(sz=100000)
        episodes=[]
        self.cluster.rollout(policy=self.actor, nepisodes=self.nepisodes, memory=evalmemory, exploration=None,
                             visualize=self.visualize,episodes=episodes)
        qvar, qbias = [], []
        episode_rewards = []
        for obs0, a0, r0, done in evalmemory.episodes(episodes):
            episode_rewards.append(np.sum(r0))
            q = self.critic.predict([obs0, a0])
            dfr = discounted_future(r0, self.gamma,done[-1])
            qe=q-dfr
            qbias.append(np.mean(qe))
            qvar.append(np.std(qe))
            pass
        #todo: make this a callback
        self.hist["reward"].append(episode_rewards)
        self.hist["qvar"].append(qvar)
        self.hist["qbias"].append(qbias)
        #print("episode summary er {} qbias {} qvar {}".format(np.mean(episode_rewards),np.mean(qbias),np.mean(qvar)))
        pass

class PlotDist(Callback):
    def __init__(self, cluster, dists, title="", fignum=None, skip=1):
        self.title = title
        self.fignum = plt.figure(fignum).number
        self.dists = dists
        self.skip = skip
        self.stats = OrderedDict()
        for name,value in self.dists.items():
            self.stats[name]=[[],[]]

    def on_epoch_end(self, epoch, logs):
        for name,value in self.dists.items():
            self.stats[name][0].append(np.mean(value[-1]))
            self.stats[name][1].append(np.std(value[-1]))
        if epoch % self.skip != 0:
            return
        plt.figure(self.fignum)
        plt.clf()
        plt.suptitle(self.title + " epoch {}".format(epoch))
        plotdict(plt,self.stats,plotter=plotmeanstd)
        plt.pause(0.001)

class ActorCriticRollout(Callback):
    def __init__(self,cluster,actor=None,critic=None,series=None,callbacks=None,nepisodes=1):
        self.cluster=cluster
        if series is None:
            series=[('',actor,critic)]
        self.series=series
        self.callbacks=callbacks
        self.nepisodes=1
        pass

    def on_epoch_end(self, epoch, logs):
        self.cluster.env.env.reset()
        assert hasattr(self.cluster.env.env, 'get_state'), "SeriesRollout requires environment have get/set_state"
        start_state = self.cluster.env.env.get_state()
        memory = None
        results=[]
        for idx, (name, actor, critic) in enumerate(self.series):
            if actor is None and not hasattr(actor, 'controller'): continue
            episodes = []
            memory = self.cluster.rollout(actor, memory=memory, nepisodes=self.nepisodes, state=start_state, episodes=episodes)
            tmp=[]
            for obs0, a0, r0, done in memory.episodes(episodes):
                q=critic.predict([obs0, a0])
                tmp.append(q)
            results.append((name,actor,critic,episodes,tmp))
        for callback in self.callbacks:
            callback.on_epoch_end(epoch,logs,memory,results)

class PltQEval2(Callback):
    def __init__(self,gamma,series,title="",fignum=None,skip=1):
        self.title=title
        self.fignum=plt.figure(fignum).number
        self.series=series
        self.gamma=gamma
        self.ag1=AutoGrowAxes()
        self.ag2=AutoGrowAxes()
        self.ag3=AutoGrowAxes()
        self.skip=skip
        self.lastt=time.perf_counter()

    def on_epoch_end(self,epoch,logs, memory=None, results=None):
        if epoch % self.skip != 0:
            return

        t=time.perf_counter()
        self.tdiff=t-self.lastt
        self.lastt=t
        plt.figure(self.fignum)
        plt.clf()
        plt.suptitle(self.title+" {} epoch {} in {:.3f} sec".format(self.cluster.env.spec.id, epoch,self.tdiff))

        for idx,(name,actor,critic,episodes,qs) in enumerate(self.results):
            if actor is None and not hasattr(actor,'controller'): continue
            for  (obs0, a0, r0, done),q in zip(memory.episodes(episodes),qs):
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
        plt.pause(0.001)

class PltQEval(Callback):
    def __init__(self,cluster,gamma,series,title="",fignum=None,skip=1):
        self.title=title
        self.fignum=plt.figure(fignum).number
        self.series=series
        self.gamma=gamma
        self.cluster=cluster
        self.ag1=AutoGrowAxes()
        self.ag2=AutoGrowAxes()
        self.ag3=AutoGrowAxes()

        self.skip=skip
        self.lastt=time.perf_counter()

    def on_epoch_end(self,epoch,logs, memory=None, episodes=None):
        if epoch % self.skip != 0:
            return
        if memory is None or episodes is None:
            self.cluster.env.env.reset()
            assert hasattr(self.cluster.env.env, 'get_state'), "plt3 requires environment have get/set_state"
            start_state = self.cluster.env.env.get_state()
            memory=None
            episodes=[]
            for idx, (name, actor, critic) in enumerate(self.series):
                if actor is None and not hasattr(actor, 'controller'): continue
                memory = self.cluster.rollout(actor, memory=memory, nepisodes=1, state=start_state, episodes=episodes)

        t=time.perf_counter()
        self.tdiff=t-self.lastt
        self.lastt=t
        plt.figure(self.fignum)
        plt.clf()
        plt.suptitle(self.title+" {} epoch {} in {:.3f} sec".format(self.cluster.env.spec.id, epoch,self.tdiff))

        for idx,(name,actor,critic) in enumerate(self.series):
            if actor is None and not hasattr(actor,'controller'): continue
            obs0, a0, r0, done,obs1 = next(memory.episodes1(episodes))
            q=critic.predict([obs0, a0])
            ar=np.copy(r0)
            for i in range(1,ar.shape[0]):
                ar[i] += ar[i-1]
            dfr=discounted_future(r0,self.gamma,done[-1])
            aq = np.copy(r0)
            # last = 0
            # for i in reversed(range(aq.shape[0])):
            #     aq[i] += self.gamma * last
            #     last = aq[i]
            tdq = TD_q(actor, critic, self.gamma, obs1, r0, done)

            plt.subplot(3,1,1)
            plt.xlim(0,self.cluster.env.spec.max_episode_steps)
            plt.ylim(*self.ag1.lim(ar[:, 0]))
            #print("plt3 {} len {} ar {}:{}".format(name,r0.shape[0], np.min(ar), np.max(ar)))
            plt.plot(ar[:, 0], label=name+" policy")
            plt.legend(loc=1,fontsize='xx-small')
            if idx==0:
                self.ag1.lim([0])
                self.ag2.lim([0])
                self.ag3.lim([0])
                plt.subplot(3,1,2)
                plt.xlim(0,self.cluster.env.spec.max_episode_steps)
                plt.ylim(*self.ag2.lim(r0[:, 0]))
                plt.plot(r0[:, 0], label=' reward')
                plt.plot(q[:-1, 0] - self.gamma * q[1:, 0], label=' q delta')
                plt.legend(loc=4,fontsize='xx-small')
                plt.subplot(3,1,3)
                plt.xlim(0,self.cluster.env.spec.max_episode_steps)
                plt.ylim(*self.ag3.lim(dfr[:, 0]))
                plt.plot(dfr[:, 0], label=' dfr')
                plt.plot(q[:, 0], label=' q')
                plt.plot(tdq[:, 0], label=' tdq')
                plt.legend(loc=4,fontsize='xx-small')
        plt.pause(0.001)

class PltObservation(Callback):
    def __init__(self,cluster,policy,title="",fignum=None,skip=1):
        self.title=title
        self.fignum=plt.figure(fignum).number
        self.skip=skip
        self.cluster=cluster
        self.policy = policy
        self.env=cluster.env.env
        self.nobs=cluster.env.observation_space.shape[0]
        self.nacts=cluster.env.action_space.shape[0]
        self.nplots=self.nobs+self.nacts+1
        self.ag=[AutoGrowAxes() for i in range(1+self.nobs+self.nacts)]

    def on_epoch_end(self,epoch,logs,memory=None,episodes=None,qs=None):
        if epoch % self.skip != 0:
            return

        env=self.cluster.env
        plt.figure(self.fignum)
        plt.clf()
        plt.suptitle(self.title + " epoch {}".format(epoch))

        for obs, act, reward, done in memory.episodes(episodes):
            for i in range(self.nobs):
                plt.subplot(self.nplots, 1, i+1)
                plt.plot(obs[:, i], label="obs{}".format(i))
                plt.xlim(0,self.env.spec.max_episode_steps)
                plt.legend(loc=1,fontsize='xx-small')

            for i in range(self.nacts):
                plt.subplot(self.nplots, 1, i+1+self.nobs)
                plt.plot(act[:, i], label="act{}".format(i))
                plt.xlim(0,self.env.spec.max_episode_steps)
                plt.legend(loc=1,fontsize='xx-small')

            plt.subplot(self.nplots, 1, 1+self.nobs+self.nacts)
            plt.plot(reward[:,0], label="reward")
            plt.xlim(0,self.env.spec.max_episode_steps)
            plt.legend(loc=1,fontsize='xx-small')

        plt.pause(0.001)

class SaveModel(Callback):
    def __init__(self,actor,critic,xdir=''):
        self.actor=actor
        self.critic=critic
        self.xdir=xdir
    def on_epoch_end(self,epoch,logs):
        self.actor.save(os.path.join(self.xdir, 'actor.h5'))
        self.critic.save(os.path.join(self.xdir,'critic.h5'))