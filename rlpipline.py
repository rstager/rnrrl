import os

from keras import regularizers
from keras.layers import Activation, BatchNormalization, Dense, Input, concatenate
from keras.models import Model, load_model
from matplotlib import pyplot as plt

from callbacks import ActorCriticEval, PlotDist, PltQEval
from rl import EnvRollout, ExperienceMemory
from rlagents import DDPGAgent
from rnr.gym import rnrenvs
from rnr.keras import OrnstienUhlenbeckExplorer
from rnr.movieplot import MoviePlot
from rnr.util import kwargs


def run(cluster, gamma=0.995, tau=0.001, prefill=10000, aepochs=0, cepochs=0, repochs=1000,lr=0.01,clr=0.01,decay=1e-4):

    reload=False
    xdir='experiments/latest'

    movie=MoviePlot("RL",path=xdir)
    movie.grab_on_pause(plt)

    actor,critic = make_models(cluster.env)

    if reload and os.path.exists(os.path.join(xdir,'actor.h5')):
        print("Load actor")
        actor = load_model(os.path.join(xdir,'actor.h5'))

    if reload and os.path.exists(os.path.join(xdir,'critic.h5')):
        print("load critic")
        critic = load_model(os.path.join(xdir,'critic.h5'))

    if reload and os.path.exists(os.path.join(xdir,'memory.p')):
        print("Load data")
        memory = ExperienceMemory.load(os.path.join(xdir,'memory.p'))
    else:
        memory=ExperienceMemory(sz=1000000)
        print("random policy envs")
        explorers = [OrnstienUhlenbeckExplorer(cluster.env.action_space, theta=.5, mu=0., dt=1.0, ) for i in range(cluster.nenv)]
        cluster.rollout(actor, nsteps=prefill, memory=memory, exploration=explorers, visualize=False)
        print("Save memory")
        memory.save(os.path.join(xdir,'memory.p'))

    agent=DDPGAgent(cluster,actor,critic,tau,gamma,mode=2,lr=lr,clr=clr,decay=decay)
    eval=ActorCriticEval(cluster,agent.target_actor,agent.target_critic,gamma)
    callbacks=[]
    callbacks.append(eval)
    callbacks.append(PltQEval(cluster, gamma, [('target', agent.target_actor, agent.target_critic),
                                               ('ddpg', agent.actor, agent.critic)], title="RL eval",fignum=1))
    callbacks.append(PlotDist(cluster, eval.hist, title="actor/critic training trends"))

    agent.train(memory, epochs=repochs, fignum=1, visualize=False,callbacks=callbacks)
    movie.finish()

def make_models(env):

    ain = Input(shape=env.action_space.shape, name='action')
    oin = Input(shape=env.observation_space.shape, name='observeration')  # full observation
    common_args = kwargs(kernel_initializer='glorot_normal',
                         #activation='relu',
                         kernel_regularizer=regularizers.l2(1e-5)
                         # bias_initializer=Constant(value=0.1),
                         )
    x = oin
    x = Dense(64, **common_args)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(32, **common_args)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(env.action_space.shape[0], activation='linear')(x)
    actor = Model(oin, x, name='actor')

    common_args=kwargs( activation='relu',
                        kernel_initializer='glorot_normal',
                        kernel_regularizer=regularizers.l2(1e-6),
                        )
    x = oin
    x = concatenate([x, ain], name='sensor_goal_action')
    x = Dense(32, **common_args)(x)
    #x = BatchNormalization()(x)  # for some reason this does not work
    x = Activation('relu')(x)
    x = Dense(128, **common_args)(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(1, activation='linear', name='Q')(x)
    critic = Model([oin, ain], x, name='critic')
    return actor,critic

if __name__ == "__main__":
    rnrenvs()
    # envname='Slider-v0'
    envname = 'Pendulum-v100'
    # envname = 'PendulumHER-v100'
    # envname='NServoArm-v6'
    # envname='ContinuousMountainCart-v100'
    # cluster = HERRollout(envname, nenvs)
    cluster=EnvRollout(envname,64)
    run(cluster, prefill=10000, gamma=0.95, tau=0.02)
