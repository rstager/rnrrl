from collections import OrderedDict

from keras.callbacks import Callback

from plotutils import plotdistdict, AutoGrowAxes
from rl import ExperienceMemory, discounted_future, TD_q
import numpy as np

from rnr.util import rms
from matplotlib import pyplot as plt

class ActorCriticEval(Callback):
    def __init__(self,cluster,actor,critic,gamma,visualize=False):
        self.cluster=cluster
        self.actor=actor
        self.critic=critic
        self.gamma=gamma
        self.visualize=visualize
        self.hist=OrderedDict()
        self.hist["reward"]=[]
        self.hist["qvar"]=[]
        self.hist["qbias"]=[]

    def on_epoch_end(self, epoch, logs):
        # roll out some on policy episodes so we can see the results
        evalmemory = ExperienceMemory(sz=100000)
        self.cluster.rollout(policy=self.actor, nepisodes=10, memory=evalmemory, exploration=None,
                             visualize=self.visualize)
        qvar, qbias = [], []
        episode_rewards = []
        for obs0, a0, r0, done in evalmemory.episodes():
            episode_rewards.append(np.sum(r0))
            q = self.critic.predict([obs0, a0])
            dfr = discounted_future(r0, self.gamma)
            bias = np.mean(q - dfr)
            qbias.append(bias)
            qvar.append(rms(q - dfr - bias))
        #todo: make this a callback
        self.hist["reward"].append(episode_rewards)
        self.hist["qvar"].append(qvar)
        self.hist["qbias"].append(qbias)
        #print("episode summary er {} qbias {} qvar {}".format(np.mean(episode_rewards),np.mean(qbias),np.mean(qvar)))

class PlotDist(Callback):
    def __init__(self,cluster,dists,title=""):
        self.title=title
        self.fignum=plt.figure().number
        self.dists=dists

    def on_epoch_end(self,epoch,logs):
        plt.figure(self.fignum)
        plt.clf()
        plt.suptitle(self.title + " epoch {}".format(epoch))
        plotdistdict(plt,self.dists)
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
