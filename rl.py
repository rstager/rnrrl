from collections import namedtuple
import random
import numpy as np
import gym

from rnr.segment_tree import SumSegmentTree

Experience = namedtuple('Experience', 'obs, action, reward, done')
MemEntry=namedtuple('MemEntry','action,obs1,reward,done,metadata')
MetaData=namedtuple('MetaData','episode,step,prev')
class ExperienceMemory():
    class Recorder:
        def __init__(self, memory, episodes=None):
            self.memory = memory
            self.md = None
            self.episodes=episodes
            self.eidx=None

        def record_reset(self, obs0):
            self.md,_ = self.memory._record(None, obs0, None, None, self.md)
            self.eidx = None

        def record_step(self,action, obs1, reward, done):
            self.md, ridx = self.memory._record(action, obs1, reward, done, self.md)
            if self.episodes is not None:
                if self.eidx is None:
                    self.eidx = len(self.episodes)
                    self.episodes.append(-1)
                self.episodes[self.eidx]=ridx

    def __init__(self,sz=10000):
        self.sz=sz
        self.didx=-1
        self.buffer=[]
        self.i_episode=0

    def recorder(self,episodes=None):
        return ExperienceMemory.Recorder(self,episodes)

    def _record(self, action, obs1, reward, done, md):
        self.didx += 1
        if self.didx >= self.sz:
            print("recycle memory buffer")
            self.didx = 0
        if md is None:
            md = MetaData(-1, 0, -1)
        # update md if this is a reset
        if action is None:
            md = MetaData(-1,0,-1) # reset
            ridx=None
        else: # record steps
            if md.episode == -1:
                md = MetaData(self.i_episode, md.step, md.prev)
                self.i_episode+=1
            ridx=self.didx
        # add a new memory entry or recycle
        if self.didx==len(self.buffer):
            self.buffer.append(MemEntry(action, obs1, reward, done, md))
        else:
            self.buffer[self.didx]=MemEntry(action, obs1, reward, done, md)
        md = MetaData(md.episode, md.step + 1, self.didx)
        return md, ridx

    def episodes(self,idxs):
        for idx in idxs:
            episode=[]
            i_episode=self.buffer[idx].metadata.episode
            while True:
                prev=self.buffer[idx].metadata.prev
                if  prev == -1:
                    break
                if self.buffer[prev].metadata.episode!=i_episode:
                    break # spans fill point
                episode.insert(0,idx)
                idx=prev
            yield self._np_experience(episode)

    def _np_experience(self,idxs):
        batchsz=len(idxs)
        Sobs = np.empty((batchsz,) + self.buffer[idxs[0]].obs1.shape)
        Saction=np.empty((batchsz,) + self.buffer[idxs[0]].action.shape)
        Sreward=np.empty((batchsz,1))
        Sdone=np.empty((batchsz,1))
        for idx, s in enumerate(idxs):
            assert self.buffer[s].metadata.prev != -1
            Sobs[idx]=self.buffer[self.buffer[s].metadata.prev].obs1
            Saction[idx]=self.buffer[s].action
            Sreward[idx]=self.buffer[s].reward
            Sdone[idx]=self.buffer[s].done
        return Sobs,Saction,Sreward,Sdone


    def obs1generator(self,batch_size=32,oversample_done=None):
        assert len(self.buffer) > 0
        while True:
            if oversample_done is None:
                idxs=np.random.choice(len(self.buffer),size=batch_size)
            else:
                p = np.array([r.data.done * oversample_done + 1 for r in self.buffer])
                p = p/np.sum(p)
                idxs=np.random.choice(p.shape[0],size=batch_size,p=p)

            #idxs=np.random.choice(len(self.buffer),size=batch_size)
            obs0 = []
            obs1 = []
            a0=[]
            r0=[]
            done=[]
            for idx in idxs:
                r=self.buffer[idx]
                if idx >= self.didx and r.metadata.prev < self.didx:
                    continue  # skip events that span the fill point
                if r.metadata.prev != -1: #sample is not the reset
                    pr = self.buffer[r.metadata.prev]
                    a0.append(r.action)
                    obs1.append(r.obs1)
                    r0.append(r.reward)
                    done.append(r.done)
                    obs0.append(pr.obs1)

            yield np.array(obs0),np.array(a0),np.expand_dims(np.array(r0),axis=-1),np.array(obs1),np.expand_dims(np.array(done),axis=-1)

    def _obsact(self):
        obs0=[]
        a0=[]
        for r in self.buffer:
            obs0.append(r.data.obs)
            a0.append(r.data.action)
        return np.array(obs0),np.array(a0)

    def _obs1act(self):
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

class PrioritizedMemory(ExperienceMemory):
    def __init__(self,sz=10000):
        super().__init(self,sz)
        self._it_sum=SumSegmentTree()

    class Recorder(ExperienceMemory.Recorder):
        def __init__(self, memory):
            super().__init__(self,memory)

        def record(self, experience,priority=0):
            idx=super().record(experience, self.md)
            self.memory._it_sum[idx] = priority

    def recorder(self):
        return PrioritizedMemory.Recorder(self)

    def sample_episode(self, episode_idx=None):
        pass

    def sample(self):
        pass

    def append(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx=super().append(*args,**kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            # TODO(szymon): should we ensure no repeats?
            mass = random.random() * self._it_sum.sum(0, self.nb_entries - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res



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
    #
    # print("n mean={}".format(np.mean(n)))
    qt = r0 + n
    # print("TD_q obs1 {} r0 {} done {} n {} qt {}".format(obs1.shape,r0.shape,done.shape,n.shape,qt.shape))
    return qt

# collection of environments for efficient rollout
class EnvRollout:
    def __init__(self,envname,nenv=16):
        self.envname=envname
        self._instances=[gym.make(envname) for x in range(nenv)]
        self.env = self._instances[0]
        self.nenv=nenv

    def rollout(self, policy, memory=None,exploration=None,visualize=False,nepisodes=None,nsteps=None,state=None,episodes=None):
        bobs = []
        recorders=[]
        step_cnt,episode_cnt=0,0

        if memory is None:
            memory=ExperienceMemory()
        n= nepisodes if nepisodes is not None and nepisodes<len(self._instances) else len(self._instances)
        for env in self._instances[:n]:
            recorders.append(memory.recorder(episodes))
            tobs = env.reset()
            recorders[-1].record_reset(tobs)
            if state is not None:
                tobs = env.env.set_state(state)
            bobs.append(tobs)

        while (nepisodes is not None and episode_cnt<nepisodes) or (nsteps is not None and step_cnt<nsteps):
            if policy is None:
                acts=[e.env.controller() for e in self._instances]
            else:
                acts = policy.predict(np.array(bobs))

            if exploration is not None:
                noises=[e.sample(a) for e,a in zip(exploration,acts)]
            else:
                noises=[None]*len(self._instances)

            for i_env, (action, env,recorder,noise) in enumerate(zip(acts, self._instances,recorders, noises)):
                if noise is not None:
                    action += noise
                tobs,reward,done,_=env.step(action)
                recorder.record_step(action,tobs, reward, done)
                if visualize and i_env==0:
                    env.render()
                if done:
                    tobs=env.reset()
                    recorder.record_reset(tobs)
                    if state is not None:
                        tobs=env.env.set_state(state)
                    if exploration is not None:
                        exploration[i_env].reset()
                    episode_cnt+=1
                bobs[i_env]=tobs
                step_cnt+=1
                if nsteps is not None and step_cnt>=nsteps:
                    break
        return memory
