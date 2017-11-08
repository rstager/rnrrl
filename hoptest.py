import pickle
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from plotutils import plotdist
from math import log, sqrt
from keras.regularizers import l2
from rnr.movieplot import MoviePlot
from textwrap import wrap
import pickle
from functools import partial
from hyperopt.pyll.base import scope
import hyperopt.pyll.stochastic


from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import  qfitting
def objective(x):
    objective.i_run = getattr(objective, 'i_run', 0) + 1
    objective.best = getattr(objective, 'best', [])
    bestcnt=4
    # layer_args = {
    #     'activation'        : x['activation'],
    #     'kernel_regularizer': l2(x['reg']),
    #     'kernel_initializer': 'glorot_normal',
    # }

    title = '\n'.join(wrap(",".join([ n+":"+("{:.2E}" if isinstance(v,float) else "{}").format(v) for n,v in x.items()])))
    r=qfitting.suite(10, title=title,visualize=False,**x)
    ra=np.array(r)
    loss=np.mean(ra[:,-30:])
    print("{} LOSS={} {}".format(objective.i_run,loss,x))

    # objective.best.append((loss, r, objective.i_run))  # plot this against top contenders
    # plt.figure(3)
    # plt.clf()
    # plt.title("best {} log mse distribution".format(bestcnt))
    # for tmp in objective.best:
    #     plotdist(plt, np.array(tmp[1]), axis=0, semilog=True, label=str(tmp[2]))
    # plt.axhline()
    # plt.legend()
    # plt.pause(0.1)

    # if len(objective.best)>bestcnt:
    #     objective.best=sorted(objective.best, key=lambda x:x[0])
    #     objective.best= objective.best[:bestcnt]

    return {
        'loss': loss,
        'status': STATUS_OK,
        'index':objective.i_run,
        # -- store other results like this
        # 'eval_time': time.time(),
        #'x': {'type': None, 'value': saveit},
        # -- attachments are handled differently
        # 'attachments':
        #     {'time_module': pickle.dumps(time.time)}
        }

@scope.define_pure
def netshape(t,b,n):
    shape=[int(b+(t-b)*l/(n-1)) for l in range(int(n))]
    return tuple(shape)

def run():
    trials = Trials()
    shapes=[[4]*4,[16]*4,[32]*4,[64]*4,[16]*6,[16]*10,[16]*16,[32]*6,[128]*2,[32]*2+[128]]
    short_activation_list=['relu','LeakyReLU','PreLU','selu']
    space = hp.choice('classifier_type', [
        {
            'rname'  : hp.choice('rname',['sparse','-d**2']),
            'gamma':0.995,
            'nsteps': 200,
            'nepochs': 200,
            'activation_args':{},
            'shape': hp.choice('shape2', [
                hp.choice('shape3', shapes),
                scope.netshape(hp.quniform('top', 16, 128, 1),
                               hp.quniform('bottom', 16, 128, 1),
                               hp.quniform('numberlayer', 2, 16, 1))
            ]),
            'layer_args':{
                'activation':hp.choice('activation',short_activation_list),
                'kernel_initializer': 'glorot_normal',
            },
            'reg':hp.lognormal('reg',log(1e-5),2),
            'lr':hp.lognormal('lr',log(1e-2),2),
            'decay':hp.lognormal('decay',log(1e-3),2),
        },
    ])
    #movie = MoviePlot({3:"NRL"}, path='experiments/hoptest')
    #movie.grab_on_pause(plt)

    best = fmin(objective,
        space=space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials)

    print(best)
    with open('trials.p','wb') as f:
        pickle.dump(trials,f)

    sqrt(0)
    plt.pause(100000)

if __name__ == "__main__":
    run()