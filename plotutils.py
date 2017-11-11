import matplotlib.pyplot as plt
import numpy as np

class AutoGrowAxes():
    def __init__(self,decay=0.01):
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

def plotdistdict(plt, dists, axis=1, semilog=False):
    '''
    Plot a set of distributions from a dictionary
    '''
    n = len(dists)
    for idx, (name,series) in enumerate(dists.items()):
        plt.subplot(n, 1, idx + 1)
        plotdist(plt, series, label=name,axis=axis,semilog=semilog)
        plt.axhline(0, color='black')
        plt.legend(loc=1, fontsize='xx-small')


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