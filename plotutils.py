import matplotlib.pyplot as plt
import numpy as np

class AutoGrowAxes():
    def __init__(self,decay=0.1):
        self.limits=None
        self.decay=decay

    def lim(self, data):
        ulimit=np.max(data)
        llimit=np.min(data)
        if self.limits is None:
            self.limits=[llimit,ulimit]
        elif  llimit<self.limits[0] or ulimit>self.limits[1]:
            self.limits=[min(self.limits[0], llimit), max(self.limits[1], ulimit)]
        if self.decay != 0.0:
            self.limits = (self.limits[0]*(1-self.decay),self.limits[1]*(1-self.decay))

        b=(self.limits[1]-self.limits[0])*0.1
        return self.limits[0]-b,self.limits[1]+b

def plotdist(plt, rs,axis=1,label="",semilog=False):
    '''
    plot the distribution of a series of sequences
    axis is the dimension on which the distribution is computed
    the other dimension is the x value in the plot
    ie. rs is of the form rs[time,series]=value
    '''
    if isinstance(rs,list):  # support for variable length rows
        m,s=[],[]
        for row in rs:
            m.append(np.mean(row))
            s.append(np.std(row))
        m=np.array(m)
        s=np.array(s)
    else:
        m=np.mean(rs,axis=axis)
        s = np.std(rs, axis=axis)
    t = np.arange(m.shape[0])
    ax=plt.gca()
    if semilog:
        p=ax.semilogy(t, m, lw=2, label=label)
        ax.fill_between(t, m + s, m,  alpha=0.25,facecolors=p[0].get_color())
    else:
        p=ax.plot(t,m,lw=2, label=label)
        ax.fill_between(t, m + s, m - s, alpha=0.25,facecolors=p[0].get_color())

def plotdict(plt, dict, semilog=False, plotter=None):
    '''
    Plot a series of data from a dictionary as subplots
    '''
    n = len(dict)
    for idx, (name,entry) in enumerate(dict.items()):
        plt.subplot(n, 1, idx + 1)
        if plotter is not None:
            plotter(plt, entry, label=name,semilog=semilog)
        else:
            plt.plot(entry,label=name,semilog=semilog)
        plt.axhline(0, color='black')
        plt.legend(loc=1, fontsize='xx-small')

def plotmeanstd(plt, stats, x=None, label="", semilog=False):
    '''
    plot the range specified by stats= (mean[], std[])
    '''
    if x is None:
        x = range(len(stats[0]))
    ax=plt.gca()
    m = np.array(stats[0])
    s = np.array(stats[1])
    if semilog:
        p=ax.semilogy(x, m, lw=2, label=label)
        ax.fill_between(x, m+s,m, alpha=0.25, facecolors=p[0].get_color())
    else:
        p=ax.plot(x, m, lw=2, label=label)
        ax.fill_between(x, m+s,m-s, alpha=0.25, facecolors=p[0].get_color())


def accumulatedist(rs,stats,axis=1):
    '''
    refactoring of plotdist allows rs table to be accumulated one step at a time
    into mean and std
    '''
    if isinstance(rs,list):  # support for variable length rows
        m,s=[],[]
        for row in rs:
            m.append(np.mean(row))
            s.append(np.std(row))
        m=np.array(m)
        s=np.array(s)
    else:
        m=np.mean(rs,axis=axis)
        s = np.std(rs, axis=axis)

