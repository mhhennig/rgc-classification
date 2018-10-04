import numpy as np
import pyspike as spk
from scipy.io import loadmat
import pandas as pd
from scipy.spatial.distance import squareform
import scipy.cluster as cluster
from matplotlib import pyplot as plt

def computeDSi(dmax,davg,num):
  x = np.arange(0,360,360/(12))
  y = np.sin(x/360*2*np.pi)
  x = np.cos(x/360*2*np.pi)
  tc = dmax / np.average(davg)
  vecs = np.vstack((tc*x,tc*y)).T
  w,v = np.linalg.eigh(np.dot(vecs.T,vecs))
  dir_tuning = np.arctan(v[0,1]/v[0,0]), np.arctan(v[1,1]/v[1,0]),w[0],w[1]
  dsi = 1-dir_tuning[2]/dir_tuning[3]
  return [dsi, dir_tuning[1], np.sqrt(dir_tuning[3])/2, dir_tuning[0], np.sqrt(dir_tuning[2])/2]

def maxSpikeRate(st_s, ms=200):
    mxs = []
    for st in st_s:
        n = spk.psth(st,ms).y
        mxs.append(np.max(n)/(ms/1000))
    return np.asarray(mxs)

def minmaxSpikeRate(st_s, ms=200):
    mxs = []
    for st in st_s:
        n = spk.psth(st,ms).y
        mxs.append((np.max(n)-np.min(n))/(ms/1000))
    return np.asarray(mxs)

def medmaxSpikeRate(st_s, ms=200):
    mxs = []
    for st in st_s:
        n = spk.psth(st,ms).y
        mxs.append((np.max(n)-np.median(n))/(ms/1000))
    return np.asarray(mxs)

# used to pull out the relevant bits...
def read_stimuli_info(report_f, trigger_f):
    # parses an integer from a line
    n = lambda l: int(l.rpartition(': ')[-1].strip())
    # parses a float from a line
    nf = lambda l: float(l.rpartition(': ')[-1].strip())
    # get the stimulus name (from directory)
    seqname =  lambda l: l.rpartition(' Executing sequence: ')[-1].strip().replace('D:\\Stimuli\\','')

    try:
        tf = h5py.File(trigger_f,'r')
    except:
        tf = loadmat(trigger_f)
    timeStampMatrix = np.array(tf.get('timeStampMatrix'),dtype='int64').flatten()
    onsetsFrame = np.array(tf.get('onsetsFrame'),dtype='int32').flatten()
    try:
        tf.close()
    except:
        print("Not closed: used loadmat")

    Stimuli_df = pd.DataFrame(columns=('Name', 'Nstim1', 'Nstim2', 'Nrefresh', 'Onset'))

    # these are the tags in the report file we need
    tags = {'Executing sequence': ((seqname, [], {}),'Name'),
          'Num of stimuli to be displayed': ((n, [], {}), 'Nstim1'),
          'Num of stimuli displayed': ((n, [], {}), 'Nstim2'),
          'Num of refresh per stimulus': ((n, [], {}), 'Nrefresh'),
          'ND': ((nf, [], {}), 'ND')
         }
    with open(report_f,'r') as file:
        i = -1
        for line in file:
            if 'Render Error' not in line and 'Resetting stimulation...' not in line and 'Restarting from stim' not in line:
                tag = line.partition(':')[0].strip()
                fun, args, kwargs = tags[tag][0]
                val = fun(line)
                if tags[tag][1] is 'Name': # make new line
                    i += 1
                    Stimuli_df = Stimuli_df.append({tags[tag][1]:fun(line)},ignore_index=True)
                    Stimuli_df['Onset'][i] = onsetsFrame[i]
                elif tags[tag][1] is not 'ND':
                    Stimuli_df[tags[tag][1]][i] = fun(line)

    del onsetsFrame
    return Stimuli_df, timeStampMatrix


# This will be used later to pull out stimulus times
def get_stimtimes(stim_n, stim_df, ts_matrix):
    s_onset = stim_df['Onset'][stim_n]
    times   = ts_matrix[ts_matrix>=stim_df['Onset'][stim_n]][:stim_df['Nstim1'][stim_n]]
    return times
        
        
def gap_score(distances, t):
    sq_dist = squareform(distances)
    l = cluster.hierarchy.linkage(distances, method='ward') 
    # create surrogate
    # this will shuffle the original, removig order
    dist_shuff = distances[np.random.permutation(distances.shape[0])]
    # here we assume Gaussian stats,YMMV
    #dist_shuff = np.random.randn(distances.shape[0])*np.std(distances)+np.mean(distances)
    #dist_shuff[dist_shuff<=0] = 1e-4
    #dist_shuff[dist_shuff>1] = 1
    sq_dist_shuff = squareform(dist_shuff)
    l_shuff = cluster.hierarchy.linkage(dist_shuff, method='ward')

    fcls = cluster.hierarchy.fcluster(l, t=t, criterion='distance')
    n_flat_clusters = np.unique(fcls).shape[0]

    fcls_shuff = cluster.hierarchy.fcluster(l_shuff, t=n_flat_clusters, criterion='maxclust')
    n_flat_clusters_shuff = np.unique(fcls_shuff).shape[0]

    cluster_dists = []
    nu = []
    cluster_dists_shuff = []
    nu_shuff = []
    for i, c in enumerate(range(n_flat_clusters)):
        inds = np.where(fcls == c+1)[0]
        nu.append(len(inds))
        if len(inds)>1:
            cluster_dists.append(sq_dist[inds].T[inds][~np.eye(len(inds),dtype=bool)])
        else:
            cluster_dists.append([0])
    for i, c in enumerate(range(n_flat_clusters_shuff)):
        inds = np.where(fcls_shuff == c+1)[0]
        nu_shuff.append(len(inds))
        if len(inds)>1:
            cluster_dists_shuff.append(sq_dist_shuff[inds].T[inds][~np.eye(len(inds),dtype=bool)])
        else:
            cluster_dists_shuff.append([0])

    Wk = 0
    Dk = []
    for i,c in enumerate(cluster_dists):
        dss = np.sum(np.linalg.norm(c)**2)
        Dk.append(dss)
        Wk += 0.5/nu[i]*dss

    Wk_shuff = 0
    Dk_shuff = []
    for i,c in enumerate(cluster_dists_shuff):
        dss = np.sum(np.linalg.norm(c)**2)
        Dk_shuff.append(dss)
        Wk_shuff += 0.5/nu_shuff[i]*dss

    return n_flat_clusters, Wk, n_flat_clusters_shuff, Wk_shuff, Dk, Dk_shuff

def eval_gap_scores(distances,ts = np.arange(0.3,2,0.01)):
    sq_dist = squareform(distances)
    l = cluster.hierarchy.linkage(distances, method='ward') 
    # create surrogate
    dist_shuff = np.random.randn(distances.shape[0])*np.std(distances)+np.mean(distances)
    #dist_shuff = distances[np.random.permutation(distances.shape[0])]
    dist_shuff[dist_shuff<=0] = 1e-4
    dist_shuff[dist_shuff>1] = 1
    sq_dist_shuff = squareform(dist_shuff)
    l_shuff = cluster.hierarchy.linkage(dist_shuff, method='ward')

    Dk = []
    Wk = []
    Nc = []
    Dk_shuff = []
    Wk_shuff = []
    Nc_shuff = []
    for t in ts:
        fcls = cluster.hierarchy.fcluster(l, t=t, criterion='distance')
        n_flat_clusters = np.unique(fcls).shape[0]

        fcls_shuff = cluster.hierarchy.fcluster(l_shuff, t=n_flat_clusters, criterion='maxclust')
        n_flat_clusters_shuff = np.unique(fcls_shuff).shape[0]

        cluster_dists = []
        nu = []
        cluster_dists_shuff = []
        nu_shuff = []
        for i, c in enumerate(range(n_flat_clusters)):
            inds = np.where(fcls == c+1)[0]
            nu.append(len(inds))
            if len(inds)>1:
                cluster_dists.append(sq_dist[inds].T[inds][~np.eye(len(inds),dtype=bool)])
            else:
                cluster_dists.append([0])
        for i, c in enumerate(range(n_flat_clusters_shuff)):
            inds = np.where(fcls_shuff == c+1)[0]
            nu_shuff.append(len(inds))
            if len(inds)>1:
                cluster_dists_shuff.append(sq_dist_shuff[inds].T[inds][~np.eye(len(inds),dtype=bool)])
            else:
                cluster_dists_shuff.append([0])

        Wkt = 0
        Dkt = []
        for i,c in enumerate(cluster_dists):
            dss = np.sum(np.linalg.norm(c)**2)
            Dkt.append(dss)
            Wkt += 0.5/nu[i]*dss
        Dk.append(Dkt)
        Wk.append(Wkt)
        Nc.append(n_flat_clusters)

        Wkt = 0
        Dkt = []
        for i,c in enumerate(cluster_dists_shuff):
            dss = np.sum(np.linalg.norm(c)**2)
            Dkt.append(dss)
            Wkt += 0.5/nu_shuff[i]*dss
        Dk_shuff.append(Dkt)
        Wk_shuff.append(Wkt)
        Nc_shuff.append(n_flat_clusters_shuff)

    return Nc, Wk, Nc_shuff, Wk_shuff, Dk, Dk_shuff, ts
        
def getPSTHs(sts,bs=100):
    single_neurons = []
    for st in sts:
        xs,ys = spk.psth(st,bs).get_plottable_data()
        single_neurons.append(ys)
    return xs,single_neurons

def plotPSTHs(ax,sts,t=None,bs=100,c='b',show_single=False,show_ticks=False, ylim=None, show_sd=True, lw=2):
    single_neurons = []
    if len(sts.shape)>1:
        for st in sts:
            xs,ys = spk.psth(st,bs).get_plottable_data()
            single_neurons.append(ys)
            if show_single:
                ax.plot(xs,ys,c='grey',lw=0.3)
    else:
        show_sd = False
    xy,ys = spk.psth(sts.flatten(), bs).get_plottable_data()
    ys = ys / sts.shape[0]
    if show_sd:
        ax.fill_between(xy, ys-np.std(single_neurons,0), ys+np.std(single_neurons,0),color=c)
        ax.plot(xy,ys,c='k',lw=2)
    else:
        ax.plot(xy,ys,c=c,lw=lw)    
    if t is not None:
        ax.set_title(t)
    if ylim is not None:
        ax.set_ylim(ylim)
    if show_ticks==False:
        ax.set_xticks(())
        ax.set_yticks(())