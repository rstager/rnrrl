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

    def __init__(self,sz=100000):
        self.sz=sz
        self.didx=-1
        self.buffer=[]
        self.i_episode=0
        self.writes = 0
        self.reads = 0
        self.wdones = 0
        self.rdones = 0

    def summary(self):
        return "writes {} dones {} sampled reads {} dones {}".format(self.writes, self.wdones, self.reads,
                                                                    self.rdones)
    def recorder(self,episodes=None):
        return ExperienceMemory.Recorder(self,episodes)

    def _record(self, action, obs1, reward, done, md):
        self.didx += 1
        if self.didx >= self.sz:
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
        self.writes+=1
        if done:
            self.wdones+=1
        return md, ridx



    def _episodes(self, idxs,terminus):
        for idx in idxs:
            episode=[]
            i_episode=self.buffer[idx].metadata.episode
            while True:
                prev=self.buffer[idx].metadata.prev
                if  prev == -1:
                    break
                if self.buffer[idx].metadata.episode!=i_episode:
                    break # spans fill point
                episode.insert(0,idx)
                idx=prev
            yield self._np_experience(episode,terminus)

    def episodes(self,idxs): # without obs1
        return self._episodes(idxs,False)

    def episodes1(self,idxs): # return include observation1
        gen=self._episodes(idxs,terminus=True)
        for obs0,a0,r0,done in gen:
            yield obs0[:-1],a0[:-1],r0[:-1],done[:-1],obs0[1:]

    def _np_experience(self,idxs,terminus=False): # terminus includes the observation after the done
        batchsz=len(idxs)+(1 if terminus else 0)
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

        if terminus:
            Sobs[-1]=self.buffer[idxs[-1]].obs1
            Saction[-1]=None
            Sreward[-1]=None
            Sdone[-1]=None
        return Sobs,Saction,Sreward,Sdone

    def obs1generator(self,batch_size=32,showdone=False,sample=None,indexes=[]):
        assert len(self.buffer) > 0
        while True:
            obs0 = []
            obs1 = []
            a0=[]
            r0=[]
            done=[]
            del indexes[:]
            for cnt in range(batch_size):
                while True:
                    idx=sample() if sample is not None else random.randrange(len(self.buffer))
                    if self.buffer[idx].metadata.prev != -1:
                        break
                indexes.append(idx)
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
                    self.reads += 1
                    if r.done:
                        self.rdones+=1

            yield np.array(obs0),np.array(a0),np.expand_dims(np.array(r0),axis=-1),np.array(obs1),np.expand_dims(np.array(done),axis=-1)


    def save(self,filename):
        import pickle
        pickle.dump(self, open(filename, "wb"))

    def load(filename):
        import pickle
        return pickle.load( open(filename, "rb"))

class PrioritizedMemory(ExperienceMemory):
    def __init__(self,max_priority=1,sz=10000,alpha=1,updater=None):
        super().__init__(sz)
        self._max_priority=max_priority
        self._alpha=alpha
        sz2=2**np.math.ceil(np.math.log(sz, 2))
        self._it_sum=SumSegmentTree(sz2)
        self.nb_entries=0
        self.updater=updater

    def summary(self):
        return "{} {}".format(super().summary(),self._it_sum.sum()/self.nb_entries)

    def _record(self, action, obs1, reward, done, md):
        md, ridx= super()._record(action,obs1,reward,done,md)
        if ridx is not None:
            self._it_sum[ridx] = self._max_priority ** self._alpha
            self.nb_entries=max(self.nb_entries,ridx+1)
        return md,ridx

    def obs1generator(self,batch_size=32,showdone=False,):
        def sample():
            return self._sample_index() # self caught in closure
        #return super().obs1generator(batch_size=batch_size,showdone=showdone,sample=sample)
        indexes=[]
        gen=super().obs1generator(batch_size=batch_size,sample=sample,indexes=indexes)
        for obs0, a0, r0, obs1, done in gen:
            yield obs0, a0, r0, obs1, done
            if self.updater is not None:
                priorities=self.updater(obs0,a0,r0,obs1,done)
                if priorities is not None:
                    #assert len(indexes) == len(priorities)
                    for (priority,idx) in zip(priorities,indexes):
                        self._it_sum[idx] = priority ** self._alpha
                    #print("sample {} base {} done {}".format(np.mean(priorities), self._it_sum.sum() / self.nb_entries,np.sum(done)))
    def _sample_index(self):
        mass = random.random() * self._it_sum.sum(0, self.nb_entries - 1)
        idx = self._it_sum.find_prefixsum_idx(mass)
        return idx


def discounted_future(reward,gamma,done=True):
        df=np.copy(reward)
        last = reward[-1]/(1-gamma) if not done else 0
        for i in reversed(range(df.shape[0])):
            df[i] += gamma * last
            last = df[i]
        return df

def TD_q(target_actor, target_critic, gamma, obs1, r0, done,end_gamma=False):
    assert obs1.shape[0]==r0.shape[0],"Observation and reward sizes must match"
    assert r0.shape[-1]==1 and done.shape[-1]==1,"Reward and done have shape [None,1]"
    a1 = target_actor.predict([obs1])
    q1 = target_critic.predict([obs1, a1])
    if end_gamma:
        n=gamma*q1
    else:
        s=np.logical_not(done).astype(bool)
        n=np.select([s], [gamma * np.array(q1)], 0)
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
        self.max_steps=getattr(self.env,'_max_episode_steps',1e6)-1

    def rollout(self, policy, memory=None,exploration=None,visualize=False,nepisodes=None,nsteps=None,state=None,episodes=None):
        bobs = []
        recorders=[]
        step_cnt,episode_cnt=0,0
        env_nsteps=[0]*self.nenv
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

        while (nepisodes is None or episode_cnt<nepisodes) and (nsteps is None or step_cnt<nsteps):
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
                if not (done and  env_nsteps[i_env] == self.max_steps):  # dont record timed out step
                    recorder.record_step(action,tobs, reward, done)
                if visualize and i_env==0:
                    env.render()
                if done:
                    tobs=env.reset()
                    recorder.record_reset(tobs)
                    env_nsteps[i_env]=0
                    if state is not None:
                        tobs=env.env.set_state(state)
                    if exploration is not None:
                        exploration[i_env].reset()
                    episode_cnt+=1
                else:
                    env_nsteps[i_env] += 1
                bobs[i_env]=tobs
                step_cnt+=1
                if nsteps is not None and step_cnt>=nsteps:
                    break
        return memory
