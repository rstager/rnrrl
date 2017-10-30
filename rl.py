from collections import namedtuple
import random
import numpy as np


Experience = namedtuple('Experience', 'obs, action, reward, done')
MemEntry=namedtuple('MemEntry','data,metadata')
MetaData=namedtuple('MetaData','episode,step,prev')
class ExperienceMemory():

    def __init__(self,sz=10000):
        self.sz=sz
        self.didx=0
        self.buffer=[]
        self.i_episode=0
        self._episodes=[]

    def recorder(self):
        class Recorder:
            def __init__(self,memory):
                self.memory=memory
                self.md=None
            def record(self,experience):
                self.md=self.memory._record(experience,self.md)
            def reset(self):
                self.md=None
        return Recorder(self)

    def _record(self, experience, md):
        if md is None:
            md= MetaData(self.i_episode, 0, -1) #env,episode,step,prev
            self._episodes.append(self.didx)
            self.i_episode+=1
        self._episodes[md.episode]=self.didx #update last entry in episode
        if self.didx==len(self.buffer):
            self.buffer.append(MemEntry(experience, md))
        else:
            self.buffer[self.didx]=MemEntry(experience, md)
        md = MetaData(md.episode, md.step + 1, self.didx) if not experience.done else None
        self.didx += 1
        if self.didx >= self.sz:
            self.didx = 0
        return md

    def episodes(self):
        for i in range(len(self._episodes)):
            yield self.sample_episode(i)

    def sample_episode(self, episode_idx=None):
        if episode_idx is None:
            episode_idx=random.randrange(len(self._episodes))
        assert episode_idx<len(self._episodes)
        buffer_idx=self._episodes[episode_idx]
        episode=[buffer_idx]
        while True:
            prev=self.buffer[buffer_idx].metadata.prev
            if  prev == -1:
                break
            if buffer_idx>self.didx and prev<self.didx:
                break # spans fill point
            buffer_idx=prev
            episode.insert(0,buffer_idx)
        return self._np_experience(episode)

    def _np_experience(self,idxs):
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


    def obs1generator(self,batch_size=32):
        while True:
            idxs=np.random.choice(len(self.buffer),size=batch_size)
            obs0 = []
            obs1 = []
            a0=[]
            r0=[]
            done=[]
            for idx in idxs:
                r=self.buffer[idx]
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
            yield np.array(obs0),np.array(a0),np.expand_dims(np.array(r0),axis=-1),np.array(obs1),np.expand_dims(np.array(done),axis=-1)

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

def discounted_future(reward,gamma):
        df=np.copy(reward)
        last = 0
        for i in reversed(range(df.shape[0])):
            df[i] += gamma * last
            last = df[i]
        return df

def TD_q(target_actor, target_critic, gamma, obs1, r0, done):
    assert obs1.shape[0]==r0.shape[0],"Observation and reward sizes must match"
    assert r0.shape[-1]==1 and done.shape[-1]==1,"Reward and done have shape [None,1]"
    a1 = target_actor.predict([obs1])
    q1 = target_critic.predict([obs1, a1])
    n=np.select([np.logical_not(done)], [gamma * np.array(q1)], 0)
    qt = r0 + n
    # print("TD_q obs1 {} r0 {} done {} n {} qt {}".format(obs1.shape,r0.shape,done.shape,n.shape,qt.shape))
    return qt