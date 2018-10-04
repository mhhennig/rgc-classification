# -*- coding: utf-8 -*-
import numpy as np

def get_median_correlation(spike_shapes, n_samples, trim_at=21, seed=42):
    # Seems like all spikes have zeros at [21:,:]
    spike_shapes = spike_shapes[:trim_at,:]
    # Get random indices for n_samples
    np.random.seed(seed)
    inds = np.arange(spike_shapes.shape[1])
    np.random.shuffle(inds)
    n_samples = min(n_samples, spike_shapes.shape[1])
    inds = inds[:n_samples]
    # Compute the correlation coefficients
    cc = np.corrcoef(spike_shapes[:,inds], rowvar=False)
    # We only want the lower triangular result, without the diagonal
    cc = cc[np.tril_indices(n_samples,k=-1)]
    # Return the median
    return np.median(cc)

def clusters_median_correlation(clusters):
    median_within_corr = np.zeros(clusters.NClusters())
    for c in range(clusters.NClusters()):
        # Get the shape for Cluster c
        spike_shapes = clusters.Shapes()[:, clusters.SpikesInCluster(c)]
        if spike_shapes.shape[0] < 2 or spike_shapes.shape[1] < 2:
            print("Warning: Unit " + str(c) + " has bad shape: " + str(spike_shapes.shape))
            median_within_corr[c] = 0
        else:
            # Compute the median correlation
            median_within_corr[c] = get_median_correlation(spike_shapes, 1000)
    return median_within_corr

def clusters_avg_amplitude(clusters):
    # Get the average difference between peak and trough within each cluster
    avg_diff = np.zeros(clusters.NClusters())
    for c in range(clusters.NClusters()):
        shape = clusters.Shapes()[:, clusters.SpikesInCluster(c)]
        avg_diff[c] = (shape.max(axis=0) - shape.min(axis=0)).mean()
    return avg_diff

def clusters_fano_factor(clusters):
    num_FF_bins = 300 # FIXME Why 300 ? Why not!
    max_time = np.max(clusters.Times()) / clusters.Sampling()
    # FIXME is printing the bin size useful?
    #print("Bin size: " + str(max_time/num_FF_bins) + " s")
    fano_factors = np.zeros(clusters.NClusters(), dtype=float)
    for i in range(clusters.NClusters()):
        spike_times = clusters.Times()[np.where(clusters.ClusterID()==i)[0]]
        n,v = np.histogram(spike_times, num_FF_bins)
        fano_factors[i] = np.var(n) / np.mean(n)
    return fano_factors
