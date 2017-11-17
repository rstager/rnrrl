import random

from gym.envs.classic_control import Continuous_MountainCarEnv,PendulumEnv
import copy
from rl import ExperienceMemory,Experience,EnvRollout
from gym import Env,spaces
import numpy as np

# def __init__(self, env):
#     self.env = env
#     # Merge with the base metadata
#     metadata = self.metadata
#     self.metadata = self.env.metadata.copy()
#     self.metadata.update(metadata)
from rnr.wrappedenvs import RestartablePendulumEnv


class HEREnvRollout(EnvRollout):
    """
    Roll out an environment with 'Hindsight Experience Replay'

    the HERenv must
        1) include the goal in the observation
        2) provide only a sparse reward at the terminal condition
        3) support an extended method:
            env.set_goal(observation,goal_observation)
                - returns a copy of observation with the goal set to the goal_observation
    """
    def __init__(self, *args, **kwargs):
        self.ngoals = kwargs.pop('ngoals',4)
        self.goal_len = 10
        super().__init__(*args, **kwargs)

    def rollout(self, policy, memory=None, *args, **kwargs):
        episodes=kwargs.pop('episodes',None)
        temp_episodes=[]

        # first roll out in the environment
        memory = super().rollout(policy, memory=memory, episodes=temp_episodes, *args, **kwargs)
        recorder = memory.recorder()
        for obs0, a0, r0, done in memory.episodes(temp_episodes,terminus=True):
            episode_length = obs0.shape[0]-1

            # first record original episode with the original goal
            # no longer needed because we record directly into memory, just loop on temp_episodes
            # recorder.record_reset(obs0[0])
            # for idx in range(episode_length):
            #     recorder.record_step( a0[idx], obs0[idx+1],r0[idx,0], done[idx,0])

            # next record additional goals
            g=sorted(random.sample(range(episode_length),self.ngoals-1))
            g+=[obs0.shape[0]-1]
            for gidx in g:
                idx = 0
                newobs0 = self.env.env.set_goal(obs0[idx], obs0[gidx])
                recorder.record_reset(newobs0)
                while idx<gidx:
                    newobs1 = self.env.env.set_goal(obs0[idx+1], obs0[gidx])
                    newr0 = -1 if idx+1 != gidx else 0
                    newdone = 0 if idx+1 != gidx else 1
                    recorder.record_step( a0[idx],newobs1, newr0, newdone)
                    idx+=1
                pass
        if episodes is not None: # if caller requested episodes, the pass along
            episodes += temp_episodes
        return memory

class PendulumHEREnv(RestartablePendulumEnv):
    def __init__(self, *args, **kwargs):
        self.shapedreward=kwargs.pop('shapedreward',False)
        super().__init__(*args, **kwargs)
        self.max_steps=200
        value=self.observation_space
        low=np.hstack([value.low,value.low,np.array([0])])
        high = np.hstack([value.high, value.high,np.array([self.max_steps])])
        self.observation_space= spaces.Box(low,high) # add goal to observation_space

    def set_goal(self, observation, goal):
        newobservation=copy.copy(observation)
        newobservation[3:6]=goal[:3]
        return newobservation

    def _reset(self):
        self.step_count=0
        super()._reset()
        return self._get_obs()

    def _step(self,u):  # change reward to be sparse
        self.step_count+=1
        obs0,reward,done,info=super()._step(u)
        if not self.shapedreward:
            #reward=0 if abs(obs0[0]-obs0[3])<0.1 and  abs(obs0[1]-obs0[4])<0.5 and  abs(obs0[2]-obs0[5])<0.5 else -1
            reward = 0 if abs(obs0[0] - obs0[3]) < 0.1 and abs(obs0[1] - obs0[4]) < 0.5 else -1
            # if reward == 0:
            #     print("reward={} obs={}".format(reward,obs0))
        #print("obs {} {} {}".format(obs0[:2],obs0[3:5],reward))
        return obs0,reward,done,info


    def _get_obs(self): # add goal to observation
        obs=np.hstack([super()._get_obs(),np.array([1.0,0,0]),np.array([self.step_count/self.max_steps])])
        return obs


class ContinuousMountainCartHEREnv(Env):
    def __init__(self, env, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def observation_space(self):
        return super().observation_space

    @property
    def action_space(self):
        return super().action_space

    def set_goal(self, observation, goal):
        return copy.copy(observation)

    def step(self, action):
        return super().step(action)

    def reset(self):
        return super().reset()


if __name__ == "__main__":
    import gym


    gym.envs.register(
        id='PendulumHER-v0',
        entry_point='hindsight:PendulumHEREnv',
        max_episode_steps=200,
    )

    cluster = HERRollout("PendulumHER-v0",1,ngoals=4)
    memory=ExperienceMemory(sz=100000)
    class RandomAgent:
        def __init__(self,env):
            self.env=env
        def predict(self,obs):
            return np.array([self.env.action_space.sample()])

    while True:
        cluster.rollout(policy=RandomAgent(cluster.env),memory=memory,nepisodes=1)
        for obs0,a0,r0,obs1,done in memory.obs1generator(32):
            for obs0,r0,done in zip(obs0,r0,done):
                print("{:.3f} {:.3f} {:.3f} {}".format(obs0[0],obs0[3],r0[0],done[0]))
