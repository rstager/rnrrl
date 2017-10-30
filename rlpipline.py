import os
import random
import time


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
from rnr.util import kwargs, rms, reduce
import rnr.gym
from plotutils import AutoGrowAxes,plotdist
from rl import Experience,ExperienceMemory,discounted_future,TD_q

def run():
    #envname='Slider-v0'
    #envname='Pendulum-v100'
    envname='NServoArm-v6'
    #envname='ContinuousMountainCart-v100'
    nenvs=64
    memsz=100000
    aepochs,cepochs,repochs=0,0,10000
    gamma=0.99
    tau=0.001
    cluster=EnvRollout(envname,nenvs)
    reload=False

    movie=MoviePlot("RL",path='experiments/latest')
    movie.grab_on_pause(plt)

    actor,critic = make_models(cluster.env)

    if reload and os.path.exists('actor.h5'):
        print("Load actor")
        actor = load_model('actor.h5')

    if reload and os.path.exists('critic.h5'):
        print("load critic")
        critic = load_model('critic.h5')

    if reload and os.path.exists('memory.p'):
        print("Load data")
        memory = ExperienceMemory.load('memory.p')
    else:
        memory=ExperienceMemory(sz=100000)
        if hasattr(cluster._instances[0], 'controller'):
            print("pretrain actor using controller")
            actor_pretrain(cluster,memory,actor,fignum=1,epochs=aepochs)
            actor.save('actor.h5')
        else:
            print("random policy envs")
            explorers = [OrnstienUhlenbeckExplorer(cluster.env.action_space, theta=.5, mu=0., dt=1.0, ) for i in range(cluster.nenv)]
            cluster.rollout(actor,nsteps=memsz,memory=memory,exploration=explorers,visualize=False)
        print("Save memory")
        memory.save('memory.p')

    if reload and os.path.exists('critic.h5'):
        print("load critic")
        critic = load_model('critic.h5')
    else:
        pretrain_critic(cluster,actor,critic,gamma,cepochs,memory,fignum=1)
        print("Saving critic")
        critic.save('critic.h5')
    agent=DDPGAgent(cluster,actor,critic,tau,gamma,mode=2)
    callbacks=[]
    callbacks.append(plt3(cluster, gamma, [('target', agent.target_actor, agent.target_critic),
                                            ('ddpg', agent.actor, agent.critic)], title="RL eval",
                              fignum=1))
    callbacks.append(plttrends(cluster,agent.hist,title="actor/critic training trends"))
    if False and hasattr(cluster.env.observation_space, 'shape') and cluster.env.observation_space.shape[0] == 2:
        callbacks.append(pltmaps(cluster,agent))

    agent.train(memory, epochs=repochs, fignum=1, visualize=True,callbacks=callbacks)
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
    x = Dense(64, **common_args)(x)
    x = Dense(32, **common_args)(x)
    x = Dense(32, **common_args)(x)
    x = Dense(1, activation='linear', name='Q')(x)
    critic = Model([oin, ain], x, name='critic')
    return actor,critic

# collection of environments for efficient rollout
class EnvRollout:
    def __init__(self,envname,nenv=16):
        self.envname=envname
        self._instances=[gym.make(envname) for x in range(nenv)]
        self.env = self._instances[0]
        self.nenv=nenv

    def rollout(self, policy, memory=None,exploration=None,visualize=False,nepisodes=None,nsteps=None,state=None):
        bobs = []
        recorders=[]
        step_cnt,episode_cnt=0,0

        if memory is None:
            memory=ExperienceMemory()
        n= nepisodes if nepisodes is not None and nepisodes<len(self._instances) else len(self._instances)
        for env in self._instances[:n]:
            tobs = env.reset()
            if state is not None:
                tobs = env.env.set_state(state)
            bobs.append(tobs)
            recorders.append(memory.recorder())

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
                tobs,reward,done,_=env.step(action)
                if visualize and i_env==0:
                    env.render()
                if noise is not None:
                    action += noise
                recorder.record(Experience(bobs[i_env], action, reward, done))
                if done:
                    tobs=env.reset()
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

def episode_summary(memory,critic,gamma,count=32):
    qvar,qbias=[],[]
    episode_rewards=[]
    for obs0, a0, r0, done in memory.episodes():
        episode_rewards.append(np.sum(r0))
        q=critic.predict([obs0,a0])
        dfr=discounted_future(r0,gamma)
        bias=np.mean(q-dfr)
        qbias.append(bias)
        qvar.append(rms(q-dfr-bias))
        if len(episode_rewards)>=count:
            break
    return episode_rewards,qvar,qbias




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

class pltmap(Callback):
    def __init__(self,cluster,func,title="",fignum=None,gsz=25):
        self.title=title
        self.fignum=plt.figure(fignum).number
        self.func=func
        self.cluster=cluster
        self.ag1=AutoGrowAxes()
        self.gsz=100
        env=self.cluster.env
        self.extent=(env.observation_space.low[0],env.observation_space.high[0],
                     env.observation_space.low[1], env.observation_space.high[1])


    def on_epoch_end(self,epoch,logs):
        X, Y = np.meshgrid(np.linspace(self.extent[0],self.extent[1], self.gsz),
                           np.linspace(self.extent[2], self.extent[3], self.gsz))
        A=self.func([np.vstack([X.flatten(),Y.flatten()]).T])
        A=A[0].reshape((self.gsz,self.gsz))
        fig = plt.figure(self.fignum)
        plt.clf()
        fig.suptitle(self.title.format(**{"epoch":epoch}))
        plt.axis(self.extent)
        avmin,avmax=self.ag1.lim(A)
        vmax=max(abs(avmin),abs(avmax))
        im = plt.imshow(A, cmap=plt.cm.RdBu_r, vmin=-vmax, vmax=vmax, extent=self.extent,origin='lower')
        im.set_interpolation('bilinear')
        cb = fig.colorbar(im)
        plt.pause(0.1)

class plt3(Callback):
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
            plt.plot(ar[:, 0], label=name)
            plt.legend(loc=1,fontsize='xx-small')
            if idx==0:
                plt.subplot(3,1,2)
                plt.xlim(0,self.cluster.env.spec.max_episode_steps)
                plt.ylim(*self.ag2.lim(r0[:, 0]))
                plt.plot(r0[:, 0], label=name+' reward')
                plt.plot(q[:-1, 0] - self.gamma * q[1:, 0], label=name+' q delta')
                plt.legend(loc=4,fontsize='xx-small')
                plt.subplot(3,1,3)
                plt.xlim(0,self.cluster.env.spec.max_episode_steps)
                plt.ylim(*self.ag3.lim(aq[:, 0]))
                plt.plot(aq[:, 0], label=name+' dfr')
                plt.plot(q[:, 0], label=name+' q')
                plt.plot(tdq[:, 0], label=name+' tdq')
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


def create_target(model):
    # create target networks
    target_model = clone_model(model)
    target_model.compile(optimizer='sgd',loss='mse') # will not use optimizer or loss, but need compile
    target_model.set_weights(model.get_weights())
    return target_model

def update_target(target, model, tau):
    target.set_weights([tau * w + (1 - tau) * wp for wp, w in zip(target.get_weights(), model.get_weights())])

class update_target_callback(Callback):
    def __init__(self,target_actor,actor,target_critic,critic,tau):
        self.target_actor=target_actor
        self.actor=actor
        self.target_critic=target_critic
        self.critic=critic
        self.tau=tau
        self.cnt=0

    def on_batch_end(self,epoch,logs):
        update_target(self.target_actor,self.actor,self.tau)
        update_target(self.target_critic,self.critic,self.tau)
        self.cnt +=1
        print("Update_target {}".format(self.cnt))

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

class pltmaps(Callback):
    def __init__(self,cluster,agent,title=""):
        self.title=title
        self.afignum=plt.figure().number
        self.cfignum=plt.figure().number
        self.cgfignum=plt.figure().number
        self.cluster=cluster
        self.agent=agent
        tmp_combined=self.critic([self.actor.input, self.actor.output])
        self.combinedf=K.function([self.target_actor.inputs[0]],[tmp_combined])
        self.actorf=K.function([self.target_actor.inputs[0]],[self.target_actor.outputs[0]])

        combinedgrad=K.gradients(tmp_combined, self.target_actor.outputs[0])
        self.combinedgradf=K.function([self.target_actor.inputs[0]],combinedgrad)

    def on_epoch_end(self,epoch,logs):
        pltmap(self.cluster, self.actorf, title="actor a(o)) {epoch}", fignum=self.afignum)
        pltmap(self.cluster, self.combinedf, title="combined q(o,a(o)) {epoch}", fignum=self.cfignum)
        pltmap(self.cluster, self.combinedgradf, title="combined grad wrt a(o)) {epoch}", fignum=self.cgfignum)

class plttrends(Callback):
    def __init__(self,cluster,hist,title=""):
        self.title=title
        self.fignum=plt.figure().number
        self.hist=hist

    def on_epoch_end(self,epoch,logs):
        plt.figure(self.fignum)
        plt.clf()
        plt.suptitle(self.title + " epoch {}".format(epoch))
        keys=reversed(sorted(self.hist.keys()))
        n=len(self.hist)
        for idx,name in enumerate(keys):
            series=self.hist[name]
            plt.subplot(n, 1, idx+1)
            plotdist(plt,series,label=name)
            plt.axhline(0, color='black')
            plt.legend(loc=1, fontsize='xx-small')
        plt.pause(0.1)

class rollouts(Callback):
    def __init__(self,cluster,policies,visualize=False):
        self.cluster=cluster
        self.policies=policies

    def on_epoch_end(self, epoch, logs):
        # roll out some on policy episodes so we can see the results
        evalmemory = ExperienceMemory(sz=100000)
        memories = [ExperienceMemory(sz=100000) for p in self.policies]
        for policy,memory in zip(self.policies,memories):
            self.cluster.rollout(policy=self.target_actor, nepisodes=10, memory=evalmemory, exploration=None,
                             visualize=self.visualize)
        #todo: make this a callback
        self.hist["reward"].append(er)
        self.hist["qvar"].append(qe2)
        self.hist["qbias"].append(qe2)
        print("episode summary er {} qe2 {}".format(np.mean(er),np.mean(qe2)))
        for c in self.callbacks:
            if hasattr(c,'on_eval_end'):
                c.on_eval_end(self.epoch, evalmemory)

class Agent:
    def __init__(self):
        self.hist={"reward":[]}
        pass
    def train(self,*args,**kwargs):
        self.callbacks=kwargs.pop('callbacks',[])
        self.visualize=kwargs.get('visualize',False)
        self.eval=kwargs.get('visualize',True)
        self.epoch=1
        self.hist["reward"]=[]
        self.hist["qvar"]=[]
        self.hist["qbias"]=[]
        self._train(*args,**kwargs)
        #self.eval = reduce(self.callbacks,lambda v,x: v or hasattr(x,'on_eval_end'),False)

    def _train(self):
        raise NotImplemented("Agent must implement train method")

    def _epoch_end(self,logs):
        if self.eval:
            # roll out some on policy episodes so we can see the results
            evalmemory = ExperienceMemory(sz=100000)
            self.cluster.rollout(policy=self.target_actor, nepisodes=10, memory=evalmemory, exploration=None,
                                 visualize=self.visualize)
            #todo: make this a callback
            er, qvar,qbias = episode_summary(evalmemory, self.target_critic, self.gamma)
            self.hist["reward"].append(er)
            self.hist["qvar"].append(qvar)
            self.hist["qbias"].append(qbias)
            print("episode summary er {} qvar {} qbias {}".format(np.mean(er),np.mean(qvar),np.mean(qbias)))
        for c in self.callbacks:
            c.on_epoch_end(self.epoch, {})
        self.epoch+=1

class DDPGAgent(Agent):
    def __init__(self,cluster,actor,critic,tau=0.001,gamma=0.99,mode=1,batch_size=32):
        super(DDPGAgent,self).__init__()
        self.cluster=cluster
        self.mode=mode
        self.actor=actor
        self.tau=tau*50 # scale tau because we update less frequently than ddpg paper
        self.gamma=gamma
        self.mode=mode
        self.target_actor = create_target(actor)
        self.target_critic = create_target(critic)
        self.critic=critic
        self.batch_size=batch_size
        self.lrdecay_interval=100
        print("Init")

        self.critic.trainable = True
        self.critic.compile(optimizer=Adam(lr=0.001, clipnorm=1., decay=1e-6), loss='mse', metrics=['mae', 'acc'])
        self.critic.summary()
        if self.mode == 1:
            self.actor.compile(optimizer=DDPGof(Adam)(self.critic, self.actor, batch_size=batch_size, lr=0.001, clipnorm=1., decay=1e-6),
                          loss='mse', metrics=['mae', 'acc'])
        elif self.mode == 2:
            self.combined = Model([actor.input], critic([actor.input, actor.output]))
            self.combined.layers[-1].trainable = False
            self.combined.compile(optimizer=Adam(lr=0.001, clipnorm=1., decay=1e-6), loss='mse',  metrics=['mae', 'acc'])
            self.critic.trainable=True
        else:
            cgrad = K.gradients(critic.outputs, critic.inputs[1])  # grad of Q wrt actions
            self.cgradf = K.function(critic.inputs, cgrad)
            actor.compile(optimizer=Adam(lr=0.001, clipnorm=1., decay=1e-6), loss='mse', metrics=['mae', 'acc'])

    def _train(self, memory, epochs=100, fignum=None, visualize=False, callbacks=[]):
        print("Train critic")
        self.critic.summary()
        print("Train combined")
        self.combined.summary()
        # Each environment requires an explorer instance
        explorers = [ OrnstienUhlenbeckExplorer(self.cluster.env.action_space, theta = .15, mu = 0.,nfrac=0.03 ) for i in range(self.cluster.nenv)]
        generator=memory.obs1generator(batch_size=self.batch_size)

        for i_epoch in range(epochs):
            for i_batch in range(100):
                astart = time.perf_counter()
                obs0, a0, r0, obs1, done = next(generator)
                tdq = TD_q(self.target_actor, self.target_critic, self.gamma, obs1, r0, done)
                # nqb=self.critic.predict([obs0,a0])
                history = self.critic.train_on_batch([obs0, a0], tdq)
                aend = time.perf_counter()

                # nqa=self.critic.predict([obs0,a0])
                # dq=nqa-nqb
                # plt.figure(5)
                # plt.clf()
                # plt.plot(tdq-nqb,label='nqb')
                # plt.plot(tdq-nqa,label='nqa')
                # plt.legend()
                # plt.pause(1)
                start = time.perf_counter()
                if self.mode==1:
                    self.actor.train_on_batch(obs0, a0)
                elif self.mode==2:
                    self.combined.train_on_batch(obs0,np.zeros_like(r0)) # loss = -q
                else:
                    # update the actor : critic.grad()*actor.grad()
                    actions = self.actor.predict(obs0)
                    grads = self.cgradf([obs0, actions])[0]
                    ya = actions + 0.1 * grads  # nudge action in direction that improves Q
                    self.actor.train_on_batch(obs0, ya)
                end = time.perf_counter()

                update_target(self.target_actor, self.actor, self.tau)
                update_target(self.target_critic, self.critic, self.tau)

            self._epoch_end({})

            print("RL Train {} {:0.3f},{:0.3f}  epochs {} of {}".format(self.cluster.envname,
                end - start, aend-astart, self.epoch, epochs))

            # add some data with latest policy into the memory
            self.cluster.rollout(policy=self.target_actor, nepisodes=50, memory=memory, exploration=explorers,visualize=False)

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
    run()