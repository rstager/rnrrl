import time
import keras.backend as K
import numpy as np
from keras.callbacks import Callback
from keras.models import Model, clone_model
from keras.optimizers import Adam

from rl import ExperienceMemory, TD_q, discounted_future
from rnr.keras import DDPGof, OrnstienUhlenbeckExplorer
from rnr.util import rms,reduce


def create_target(model):
    # create target networks
    target_model = clone_model(model)
    target_model.compile(optimizer='sgd', loss='mse')  # will not use optimizer or loss, but need compile
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

class Agent:
    def __init__(self):
        self.hist={"reward":[]}
        pass
    def train(self,*args,**kwargs):
        self.callbacks=kwargs.pop('callbacks',[])
        self.visualize=kwargs.get('visualize',False)
        self.eval=kwargs.get('visualize',True)
        self.epoch=1
        self._train(*args,**kwargs)

    def _train(self):
        raise NotImplemented("Agent must implement train method")

    def _epoch_end(self,logs):
        for c in self.callbacks:
            c.on_epoch_end(self.epoch, {})
        self.epoch+=1


class DDPGAgent(Agent):
    def __init__(self,cluster,actor,critic,tau=0.001,gamma=0.99,mode=2,batch_size=256,lr=0.015,decay=1e-5,clr=None,
                 clip_tdq=None,end_gamma=False,critic_training_cycles=1,verbose=False,nbatches=100,nfrac=0.03):
        super().__init__()
        self.verbose=verbose
        self.cluster=cluster
        self.mode=mode
        self.actor=actor
        self.tau=tau # *50 # scale tau because we update less frequently than ddpg paper
        self.gamma=gamma
        self.mode=mode
        self.target_actor = create_target(actor)
        self.target_critic = create_target(critic)
        self.critic=critic
        self.batch_size=batch_size
        self.lrdecay_interval=100
        clr=clr if clr is not None else lr
        self.nbatches=nbatches
        self.freeze_critic=False
        self.freeze_actor=False
        self.clip_tdq=clip_tdq
        self.end_gamma=end_gamma
        self.critic_training_cycles=critic_training_cycles
        self.nfrac=nfrac

        self.critic.trainable = True
        self.critic.compile(optimizer=Adam(lr=clr, clipnorm=1., decay=decay), loss='mse', metrics=['mae', 'acc'])
        self.critic._make_train_function()
        if self.verbose:
            self.critic.summary()
            print("DDPG mode {} gamma={:.3e} tau={:.3e} lr={:.3e} clr={:.3e} decay={:.3e} cycles={} bsz={}".format(mode,gamma,tau,lr,clr,decay,
                                                                                    self.critic_training_cycles,self.batch_size))
        if self.mode == 1:
            self.actor.compile(optimizer=DDPGof(Adam)(self.critic, self.actor, batch_size=batch_size, lr=lr, clipnorm=1., decay=decay),
                          loss='mse', metrics=['mae', 'acc'])
        elif self.mode == 2:
            self.combined = Model([actor.input], critic([actor.input, actor.output]))
            self.combined.layers[-1].trainable = False
            self.combined.compile(optimizer=Adam(lr=lr, clipnorm=1., decay=decay), loss='mse',  metrics=['mae', 'acc'])
            self.combined._make_train_function()
        else:
            cgrad = K.gradients(critic.outputs, critic.inputs[1])  # grad of Q wrt actions
            self.cgradf = K.function(critic.inputs, cgrad)
            actor.compile(optimizer=Adam(lr=lr, clipnorm=1., decay=decay), loss='mse', metrics=['mae', 'acc'])

    #lr and clr are stored in the models
    @property
    def lr(self):
        if self.mode != 2:
            return K.get_value(self.actor.optimizer.lre)
        else:
            return K.get_value(self.combined.optimizer.lr)
    @lr.setter
    def lr(self, value):
        if self.mode != 2:
            K.set_value(self.actor.optimizer.lr, value)
        else:
            K.set_value(self.combined.optimizer.lr,value)
    @property
    def clr(self):
        return K.get_value(self.critic.optimizer.lr)
    @clr.setter
    def clr(self, value):
        K.set_value(self.critic.optimizer.lr, value)

    def _train(self, memory=None, epochs=100, nepisodes=None, nsteps=10000, fignum=None, visualize=False,minsteps=10000,updates=False):
        if memory is None:
            memory = ExperienceMemory(env=self.cluster.env,sz=1000000)
        #if nepisodes is None:
        #    nepisodes = self.cluster.nenv
        if self.verbose:
            print("Train critic")
            self.critic.summary()
            if hasattr(self,'combined'):
                print("Train combined")
                self.combined.summary()
            else:
                print("Train ddpg actor")
                self.actor.summary()
        # Each environment requires an explorer instance
        explorers = [ OrnstienUhlenbeckExplorer(self.cluster.env.action_space, theta = .15, mu = 0.,nfrac=self.nfrac) for i in range(self.cluster.nenv)]
        generator = memory.obs1generator(batch_size=self.batch_size, showdone=True)
        #explorers = [None]*self.cluster.nenv
        # sample timing test...
        # 0.1 ms per next(generator)
        # 1.2 ms per tdq
        # 4 ms per train critic
        # 4ms per train actor
        # 3ms per update weights
        # 1.2s for plotting callbacks
        self.cluster.rollout(policy=self.target_actor, nsteps=minsteps, memory=memory,
                             exploration=explorers, visualize=visualize)
        for i_epoch in range(epochs):
            start = time.perf_counter()
            for i_batch in range(self.nbatches):
                if not self.freeze_critic:
                    for i_qtrain in range(self.critic_training_cycles):
                        obs0, a0, r0, obs1, done = next(generator)
                        tdq = TD_q(self.target_actor, self.target_critic, self.gamma, obs1, r0, done,
                                   end_gamma=self.end_gamma)
                        if self.clip_tdq is not None:
                            tdq = np.clip(tdq, self.clip_tdq / (1 - self.gamma), 0)
                        self.critic.train_on_batch([obs0, a0], tdq)

                if not self.freeze_actor:
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
                update_target(self.target_actor, self.actor, self.tau)
                update_target(self.target_critic, self.critic, self.tau)
            self._epoch_end({})
            end = time.perf_counter()

            # add some data with latest policy into the memory
            self.cluster.rollout(policy=self.target_actor, nepisodes=nepisodes, nsteps=nsteps, memory=memory,
                                 exploration=explorers,visualize=visualize)
            # print("RL Train {} {:0.3f} sec,  epochs {} of {}".format(self.cluster.envname,
            #     end - start,  self.epoch, epochs))
            #if self.verbose:
            print("epoch {} tau {} gamma {} lr {} clr {}".format(i_epoch,self.tau,self.gamma,self.lr,self.clr))
            print(memory.summary())

