from __future__ import absolute_import
from __future__ import with_statement
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from numpy import *
import numpy as np
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt



def load_sta_stimulus(filename, N=64):
    swn = loadmat(filename)['pixels_allframes']
    swn = reshape(swn,(shape(swn)[0],N,N))
    return swn - mean(swn,axis=(1,2))[:,None,None]

def load_sta_triggers(filename, duration, ntrials = 9):
    trig_t = loadmat(filename, squeeze_me=True)
    ntrig = trig_t['timeStampMatrix'].shape[0]
    assert ntrig/ntrials is not ntrig//ntrials, 'Number of stimuli is not multiple of trials'
    dt = ntrig//ntrials
    trigs = trig_t['timeStampMatrix']
    block_starts = np.zeros(ntrials)
    block_stops = np.zeros(ntrials)
    block_lens = np.zeros(ntrials)
    for i in range(ntrials):
        block_starts[i] = trigs[i*dt]
        block_stops[i] = trigs[(i+1)*dt-1]
        block_lens[i] = np.median(np.diff(trigs[i*dt:(i+1)*dt]))
    #print('trigs shape',trigs.shape)
    # Estimate duration of stimuli
    frame_len = np.median(block_lens)
    print('frame_len',frame_len)
    # print('In seconds',frame_len/Fs)
    stimindex = np.zeros((duration,),'int')*NaN
    i = 0
    for j in range(len(trigs)-1):
        stimindex[trigs[j]:trigs[j+1]]=i
        i+=1
    return stimindex, block_starts, block_stops

def get_loc(f, unit):
    return f['centres'].value[unit][:2]

def get_spikes(f, unit):
    return f['times'][f['cluster_id'][:] == unit]

def compute_sta_slow(f, unit, wn, triggers, window=(-353,4237,58), radius=15):
    stimindex, block_starts, block_stops = triggers
    nspace = 2*radius+1
    sc,sr = get_loc(f, unit)
    spikes = get_spikes(f, unit)
    STA = []
    for delay in range(window[0],window[1],window[2]):
        spikest = np.copy(spikes).astype(int)-delay
        spikest = np.concatenate([spikest[(spikest>=a)&(spikest<b)] for (a,b) in zip(block_starts,block_stops)])
        stiminds = int32(stimindex[spikest])
        frames = np.zeros((stiminds.shape[0], nspace, nspace))
        for i in arange(-radius,radius+1): # r
            for j in arange(-radius,radius+1): # c
                r = int(round(i+sr))
                c = int(round(j+sc))
                if r<0 or c<0 or r>=64 or c >=64: continue
                if sqrt(i*i+j*j)>radius: continue
                frames[:,i,j] = wn[stiminds,r,c]
        STA.append(fft.fftshift(np.mean(frames,0)))
    return np.array(STA)

def compute_sta(f, unit, wn, triggers, window=(-353,4237,58), radius=15):
    stimindex, block_starts, block_stops = triggers
    nspace = 2*radius+1
    sc,sr = get_loc(f, unit)
    spikes = get_spikes(f, unit)
    N = wn.shape[1]
    
    x, y = np.ogrid[0:N, 0:N]
    x = x - int(np.round(sc))
    y = y - int(np.round(sr))
    ind = np.sqrt(1. * x * x + 1. * y * y) <= radius
    inds = np.where(np.sqrt(1. * x * x + 1. * y * y) <= radius)

    x, y = np.ogrid[0:nspace, 0:nspace]
    x = x - int(nspace/2)
    y = y - int(nspace/2)
    indf = np.sqrt(1. * x * x + 1. * y * y) <= radius
    # some black magic to treat cases at the borders
    x = x + int(np.round(sc))
    y = y + int(np.round(sr))
    ii = np.ix_(((x.T>63) | (x.T<0))[0])
    indf[ii,:] = False
    ii = np.ix_(((y>63) | (y<0))[0])
    indf[:,ii] = False
    indsf = np.where(indf)

    swnt = wn[:,inds[1],inds[0]]
    frames = []
    for delay in range(window[0],window[1],window[2]):
        spikest = np.copy(spikes).astype(int)-delay
        spikest = np.concatenate([spikest[(spikest>=a)&(spikest<b)] for (a,b) in zip(block_starts,block_stops)])
        stiminds = int32(stimindex[spikest])
        frames.append(np.mean(swnt[stiminds],0))
    STA2 = np.zeros((len(frames), nspace, nspace))
    for i,frame in enumerate(frames):
        STA2[i,indsf[1],indsf[0]] = frame
    return STA2

def get_sta_peak(STA):
    peaks = (np.unravel_index(STA.argmin(), STA.shape),np.unravel_index(STA.argmax(), STA.shape))
    return peaks[0] if peaks[0][0]<peaks[1][0] else peaks[1]
#     return np.unravel_index(np.abs(STA).argmax(), STA.shape)

def Gaussian2D(cent, xo, yo, amplitude, sigma_x, sigma_y, theta, offset):
    x,y = cent
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                            + c*((y-yo)**2)))
    return g.ravel()

def fit_sta_gaussian(STA, t=None):
    nspace = STA.shape[1]
    peak = get_sta_peak(STA)
    if t is None:
        t = peak[0]
    # hopefully these are sensible initial conditions
    initial_guess = [peak[2],peak[1], STA[peak], 2, 2, 0, 0]
    x = np.linspace(0, nspace, nspace)
    y = np.linspace(0, nspace, nspace)
    x, y = np.meshgrid(x, y)
    try:
        popt, pcov = opt.curve_fit(Gaussian2D, (x, y), STA[t].flatten(), p0=initial_guess)
    except Exception as e:
        popt = np.zeros(7)
        pcov = np.zeros((7,7))
        e.__traceback__ = None
    return popt, pcov

def plot_sta_2d (STA, po=None, scale=2, verbose=True, peakloc=True):
    peak = get_sta_peak(STA)
    if verbose:
        print('Peak at %d:%d frame %d'%(peak[1],peak[2],peak[0]))
    plt.imshow(STA[peak[0]], origin='lower')
    if peakloc:
        plt.plot((peak[2],peak[2]),(0, STA.shape[2]),'r:')
        plt.plot((0, STA.shape[2]),(peak[1],peak[1]),'r:')
    if po is not None:
        theta = np.linspace(0,2*np.pi,100)
        r = 1 / np.sqrt((np.cos(theta))**2 + (np.sin(theta))**2)
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        data = np.array([x,y])
        S = np.array([[scale*po[3],0],[0,scale*po[4]]])
        R = np.array([[np.cos(po[5]),-np.sin(po[5])],[np.sin(po[5]),np.cos(po[5])]]).T
        T = np.dot(R,S)
        data = np.dot(T,data)
        data[0] += po[0]
        data[1] += po[1]
        plt.plot(data[0], data[1], 'r')
    plt.grid(False)

def get_sta_at_peak(STA):
    peak = get_sta_peak(STA)
    return STA[:,peak[1],peak[2]]
    plt.xlabel('Time (s)')
    
def plot_sta_peak(STA, window, Fs, verbose=True):
    x = np.arange(window[0], window[1], window[2])/Fs
    peak = get_sta_peak(STA)
    if verbose:
        print('Peak at %d:%d'%(peak[1],peak[2]))
    plt.plot(x,STA[:,peak[1],peak[2]])
    plt.xlabel('Time (s)')
    
def plot_sta_peak_crossect(STA, window, Fs, verbose=True, direction=1):
    x = np.arange(window[0], window[1], window[2])/Fs
    peak = get_sta_peak(STA)
    if verbose:
        print('Peak at %d:%d'%(peak[1],peak[2]))
    for i in range(STA.shape[1]):
        if direction == 1:
            plt.plot(x,STA[:,i,peak[2]],'grey')
        else:
            plt.plot(x,STA[:,peak[1],i],'grey')
    plt.xlabel('Time (s)')

