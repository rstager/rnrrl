
import numpy as np
from keras.layers import Input,Dense,concatenate,Dropout,LeakyReLU,PReLU,ThresholdedReLU,ELU
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.initializers import Constant,RandomNormal,RandomUniform
from tensorflow import int32

from rnr.movieplot import MoviePlot
import random
import copy
from plotutils import plotdist

import matplotlib.pyplot as plt
from rl import discounted_future,TD_q

activations=['selu','relu','tanh','sigmoid','hard_sigmoid','LeakyReLU','PreLU']
activation_layers={'LeakyReLU':LeakyReLU,'PreLU':PReLU}

def make_fake_episode(rname,gamma,nsteps):
    obs0=np.expand_dims(np.linspace(-2, 0, nsteps), axis=-1)
    obs1=np.vstack([obs0[1:],np.zeros_like(obs0[0])])
    a0=np.zeros([nsteps, 1])
    done=np.zeros([nsteps, 1])
    done[-1,0]=1
    if rname=='sparse':
        r0=np.array([0.0]*(nsteps-1)+[1.0])      # sparse reward at end
    elif rname=='const':
        r0=-np.array([1.0 / nsteps] * (nsteps))  # running negative reward
    elif rname == '-d':
        r0=-np.linspace(1.0, 0, nsteps)          # -distance
    elif rname == '-d**2':
        r0=-np.square(np.linspace(1.0, 0, nsteps))  # -distance**2
    else:
        assert True,"Invalid reward profile name"

    dfr=discounted_future(r0,gamma)
    if rname!='sparse':
        r0/=abs(dfr[0])  #normalize to a -1 inital dfr
    r0 = np.expand_dims(r0, axis=-1)
    return r0,obs0,obs1,a0,done

class FakeActor(object): # fake actor always predicts zero
    def predict(self,obs):
        return np.zeros((obs[0].shape[0],1))

def experiment(layer_args, shape, activation_args={}, title="", rname=None,
               lr=0.001,decay=1e-6,visualize=False,nepochs=100,nsteps=100,gamma=None,
               oversample_count=0,Actor=FakeActor,make_episode=make_fake_episode,
               reg=None):

    r0,obs0,obs1,a0,done=make_episode(rname,gamma,nsteps) #create fake episode data
    actor = Actor()
    #first make the model
    ain = Input(shape=(1,), name='action')
    oin = Input(shape=(1,), name='observeration')  # full observation
    x = concatenate([oin, ain], name='observation_action')
    layer_args=copy.deepcopy(layer_args)
    if layer_args['activation'] in activation_layers:
        alayer = activation_layers[layer_args.pop('activation')]
    else:
        alayer = None
    if reg is not None:
        layer_args['kernel_regularizer']=l2(reg)
    for count in shape:
        x = Dense(count, **layer_args)(x)
        if alayer:
            x = alayer(**activation_args)(x)

    x = Dense(1, activation='linear', name='Q')(x)
    critic = Model([oin, ain], x, name='critic')
    critic.compile(optimizer=Adam(lr=lr, clipnorm=1., decay=decay), loss='mse', metrics=['mae', 'acc'])

    #now train it
    loss=[]
    dfr = discounted_future(r0, gamma)
    batch_sz=32
    for epoch in range(nepochs):
        tdq = TD_q(actor, critic, gamma, obs1, r0, done)
        for n in range(10):
            p=np.ones_like(done).squeeze()/(done.shape[0]+oversample_count)
            p[-1]*=(oversample_count+1)
            samples=np.random.choice(np.arange(a0.shape[0]),size=batch_sz,p=p)
            critic.train_on_batch([obs0[samples],a0[samples]], tdq[samples])
        #critic.fit([obs0,a0],tdq,shuffle=True,verbose=0) # trains much slower
        metrics=critic.test_on_batch([obs0, a0],dfr)
        loss.append(metrics[0])
        if visualize or (visualize is None and epoch == nepochs-1):
            nqa=critic.predict([obs0,a0])
            plt.figure(1)
            plt.clf()
            plt.title(" epoch {} : {}".format(epoch, title))
            plt.plot(tdq,label='tdq')
            plt.plot(nqa,label='nqa')
            plt.plot(dfr,label='dfr')
            plt.legend()
            plt.pause(0.1)
    return loss


def suite(count, *args, title, visualize=False, **kwargs):
    if visualize or visualize is None:
        plt.figure(2)
        plt.clf()
        plt.title(title)
    results=[]
    viz=visualize is None or visualize
    for i in range(count):
        results.append(experiment(*args, title=title, visualize=viz, **kwargs))
        if visualize:
            plt.figure(2)
            plt.semilogy(results[-1])
            plt.axhline()
            plt.pause(0.1)
        if visualize is None:
            viz=None
    return results

class nextone(object):
    def __init__(self):
        self.counter=0
    def next(self,i):
        if i == self.counter:
            self.counter+=1
            return True
        else:
            return False
    def reset(self):
        self.counter=0
    def __call__(self, *args, **kwargs):
        return self.next(args[0])

def run():
    movie = MoviePlot("RL", path='experiments/qfitting')
    movie.grab_on_pause(plt)
    decay=None
    next=nextone()
    layer_args={}
    for i in range(100):
        if i==0:
            shape = [32,32,16] #
            reg=1e-5
            layer_args['activation']='relu'
            layer_args['kernel_initializer']='glorot_normal'
            layer_args['bias_initializer']=RandomNormal(mean=0.0, stddev=0.05)
            lr=0.001
            decay=1e-3
            rname = 'sparse'
            gamma=0.9
            nepochs=300
            oversample_count=0

        elif i==1:
            del layer_args['bias_initializer']
        elif i==2:
            shape = [32,32,16] #
            layer_args['activation']='relu'
        elif i==3:
            rname = '-d'
            rname = '-d**2'
            lr=0.001
            decay=1e-4
        elif i==4:
            shape = [128, 64, 32, 32]  #
            layer_args = {
                'activation'        : random.choice(['relu']),
                'kernel_regularizer': l2(reg),
                'kernel_initializer': 'glorot_normal',
            }
            rname = '-d'
            lr = 0.001
        elif i==5:
            reg=1e-4
            shape = [128, 128]  #
        else:
            break

        label="{} {} lr:{}/{} reg:{} {} g:{} os:{}".format(rname,shape,lr,decay,reg,str(layer_args['activation']),gamma,oversample_count)
        r=suite(3, layer_args=layer_args, shape=shape, title=label, rname=rname,
                lr=lr,visualize=True,decay=decay,nepochs=nepochs,gamma=gamma,reg=reg,
                oversample_count=oversample_count)
    plt.pause(1000)

if __name__ == "__main__":
    run()