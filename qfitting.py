
import numpy as np
from keras.layers import Input,Dense,concatenate,Dropout,LeakyReLU,PReLU,ThresholdedReLU,ELU
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.initializers import Constant,RandomNormal,RandomUniform

from rnr.movieplot import MoviePlot
import random
import copy
from plotutils import plotdist

import matplotlib.pyplot as plt



def TD_q(target_actor, target_critic, gamma, obs1, r0, done):
    assert obs1.shape[0]==r0.shape[0],"Observation and reward sizes must match"
    assert r0.shape[-1]==1 and done.shape[-1]==1,"Reward and done have shape [None,1]"
    a1 = target_actor.predict([obs1])
    q1 = target_critic.predict([obs1, a1])
    qt = r0 + np.select([np.logical_not(done)], [gamma * np.array(q1)], 0)
    return qt

def discounted_future(reward, gamma):
    df = np.copy(reward)
    last = 0
    for i in reversed(range(df.shape[0])):
        df[i] += gamma * last
        last = df[i]
    return df

gamma=0.995
nsteps=500
obs0=np.expand_dims(np.linspace(-2, 0, nsteps), axis=-1)
obs1=np.vstack([obs0[1:],np.zeros_like(obs0[0])])
a0=np.zeros([nsteps, 1])
done=np.zeros([nsteps, 1])
done[-1,0]=1
rdata=np.array([
     np.array([0.0]*(nsteps-1)+[1.0]),                   # sparse reward at end
     -np.array([1.0/nsteps]*(nsteps)),                 # running negative reward
     -np.linspace(1.0,0,nsteps),             # -distance
     -np.square(np.linspace(1.0,0,nsteps))]) # -distance**2
for i,r0 in enumerate(rdata):
    dfr=discounted_future(r0,gamma)
    if i!=0:
        rdata[i]/=abs(dfr[0])  #normalize to a -1 inital dfr
rdata = np.expand_dims(rdata, axis=-1)
r0s={"sparse":rdata[0],'const':rdata[1],'-d':rdata[2],'-d**2':rdata[3]}

class Actor(object):
    def predict(self,obs):
        return np.zeros((nsteps,1))
actor=Actor()

activations=['selu','relu','tanh','sigmoid','hard_sigmoid','LeakyReLU','PreLU']
activation_layers={'LeakyReLU':LeakyReLU,'PreLU':PReLU}


def experiment(layer_args, shape, activation_args={}, title="", rname=None,
               lr=0.001,decay=1e-6,visualize=False,nepochs=100):
    #first make the model
    ain = Input(shape=(1,), name='action')
    oin = Input(shape=(1,), name='observeration')  # full observation
    x = concatenate([oin, ain], name='observation_action')
    if layer_args['activation'] in activation_layers:
        layer_args=copy.copy(layer_args)
        alayer = activation_layers[layer_args.pop('activation')]
    else:
        alayer = None

    for count in shape:
        x = Dense(count, **layer_args)(x)
        if alayer:
            x = alayer(**activation_args)(x)

    x = Dense(1, activation='linear', name='Q')(x)
    critic = Model([oin, ain], x, name='critic')
    critic.compile(optimizer=Adam(lr=lr, clipnorm=1., decay=decay), loss='mse', metrics=['mae', 'acc'])

    #now train it
    loss=[]
    r0=r0s[rname]
    dfr = discounted_future(r0, gamma)
    for epoch in range(nepochs):
        tdq = TD_q(actor, critic, gamma, obs1, r0, done)
        critic.train_on_batch([obs0, a0], tdq)
        #critic.fit([obs0,a0],tdq,shuffle=True,verbose=0) # trains much slower
        metrics=critic.test_on_batch([obs0, a0],dfr)
        loss.append(metrics[0]) #print(critic.metrics_names,metrics[0])
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


def suite(count, *args, title, visualization=False, **kwargs):
    plt.figure(2)
    plt.clf()
    plt.title(title)
    results=[]
    viz=visualization
    for i in range(count):
        results.append(experiment(*args, title=title, visualize=viz, **kwargs))
        plt.figure(2)
        plt.semilogy(results[-1])
        plt.axhline()
        plt.pause(0.1)
        if not viz:
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
    for i in range(100):
        if i==0:
            shape = [128,128] #
            reg=1e-5
            common_args = {
                'activation':random.choice(['LeakyReLU']),
                'kernel_regularizer':l2(reg),
                'kernel_initializer':'glorot_normal',
            }
            rname='sparse'
            lr=0.001
            decay=1e-3
            rname = '-d**2'
        elif i==1:
            shape = [32,32,128] #
        elif i==2:
            lr=0.001
            decay=1e-4
        elif i==3:
            shape = [128]
        elif i==4:
            shape = [128, 64, 32, 32]  #
            common_args = {
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

        label="{} {} lr:{}/{} reg:{} {} ".format(rname,shape,lr,decay,reg,str(common_args['activation']))
        r=suite(10, layer_args=common_args, shape=shape, title=label, rname=rname,
                lr=lr,visualization=None,decay=decay,nepochs=1000)
        plt.figure(3)
        plt.title("log mse distribution")
        plotdist(plt, np.array(r), axis=0,semilog=True,label=label)
        plt.axhline()
        plt.legend()
        plt.pause(0.1)
    plt.pause(1000)

if __name__ == "__main__":
    run()