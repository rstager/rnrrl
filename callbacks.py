from collections import OrderedDict

from keras.callbacks import Callback

from plotutils import plotdistdict
from rl import ExperienceMemory, discounted_future
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
        print("episode summary er {} qbias {} qvar {}".format(np.mean(episode_rewards),np.mean(qbias),np.mean(qvar)))

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


