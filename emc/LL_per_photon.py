# calculate the poisson prob per photon
# hopefully this metric will tell us how 'good' a frame
# is with respect to the current merge in a way that does
# not strongly depend on photon counts

# for a single photon, recorded at pixel i, the poisson prob is w_i 
# so I could just average that number:

# sum_i K_di w_i / K_d

# where K_d = sum_i K_di. This should not scale with K_d.
# but then this metric would reward a pattern with all counts
# at a pixel that happens to have a large w_i 
# I think we should punish frames with a low multiplicity

# K_d! / (prod_i K_di!)  (sum_i K_di w_i) / K_d

# where K_d! / (prod_i K_di!) is the number of unique ways to distribute
# K_d photons over the recorded pattern. This is equal to 1 if all photons
# are on the same pixel and greater than one the more distributed the photon
# counts are. This could scale quite dramatically with K_d


# calculation: don't need gpu
# (1) get most likely orientation for each pattern (from pickle file):
    # r = most_likely_orientations.pickle

# get list of tomograms for each pattern
# calculate dot product K_di w_i (perhaps I do reuse logR code for this?)
# calculate K_d! / prod_i K_di!
# we already have photon sums K_d

import pickle
import numpy as np
import time
import h5py
import tqdm 
import math
import scipy.special

import argparse
from pathlib import Path

from emc import logR
from emc.tomograms import * 
from emc.data_getter import Data_getter

if __name__ == '__main__':
    description = \
    """
    Calculate log-likelihood per photon for most likely orientation.
    """
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-I', '--merged_intensity', type=str, default='merged_intensity.pickle', \
                        help="filename of the python pickle file containing merged intensities.")
    parser.add_argument('-d', '--data', type=str, default='data.cxi', \
                        help="cxi file containing data frames")
    parser.add_argument('--dc', type=int, default=1024, \
                        help="number of detector frames to simultaneously hold in memory for inner loop.")
    parser.add_argument('-o', '--output', type=str,  \
                        help="h5 file to write probability matrix. By default the output = probability-matrix-{merged_intensity}.h5")
    args = parser.parse_args()
    
    args.dataname = Path(args.data).stem
    
    # remove .pickle from merged_intensity filename
    if '.' in args.merged_intensity :
        d = '.'.join(args.merged_intensity.split('.')[:-1])
    else :
        d = args.merged_intensity
    
    if args.output == None :
        args.output = f'probability-matrix-{d}.h5'


if __name__ == '__main__':
    print('\n\n')
    # get merged intensity
    d  = pickle.load(open(args.merged_intensity, 'rb'))
    I  = d['I'].copy()
    dq = d['dq']

    start_time = time.time()
    
    # get rotations from file or recalculate
    # --------------------------------------
    # get rotations from file
    with h5py.File(args.output, 'r') as f:
        rot_order = f['rotation-order'][()]
        qmin = f['qmin'][()]
        qmax = f['qmax'][()]
        ksums = f['photon_sums'][()]
        wsums = f['tomogram_sums'][()]
    
    R, rot_order = logR.get_rotations(rot_order)
    
    # get q-mask etc.
    # ---------------
    qmask, q, C, qmin, qmax = logR.get_qmask(qmin, qmax, args.dataname)

    # check that we are sampling within the domain of the merged intensities
    if qmax <= qmin :
        raise Exception(f'qmax is less than or equal to qmin! {qmax}<={qmin}')
    
    if qmax > d['qmax'] :
        raise Exception('qmax is greater than merged intensity-boundary')
    
    if qmin < d['qmin'] :
        raise Exception('qmin is less than merged intensity inner-boundary')

    # get most likely orientation for each pattern (from pickle file):
    rs = pickle.load(open('most_likely_orientations.pickle', 'rb'))
    
    Ndata     = len(ksums)
    Mrot      = np.int32(R.shape[0])
    Npix      = np.int32(np.sum(qmask))
    
    # get tomogram for each pattern
    #W = calculate_tomograms(I, C, q, qmask, R, dq, rs)
    # normalise tomograms
    #W /= wsums
    
    # calculate dot product sum_I K_di W_di
    data_getter = Data_getter(args.data, 'entry_1/data_1/data')
    
    # loop in chunks
    D    = math.ceil(Ndata/args.dc)
    
    pixels = np.where(qmask.ravel())[0]
    Kw     = np.empty((Ndata,), dtype = float)

    K = np.empty((args.dc, len(pixels)), dtype = data_getter.dtype)
    W = np.empty((args.dc, len(pixels)), dtype = np.float32)

    for d in tqdm.tqdm(range(D), desc='calculating K . w'):
        dstart = d*args.dc
        dstop  = min(dstart + args.dc, Ndata)
        dd     = dstop - dstart
         
        r = rs[dstart: dstop]
        
        W[: dd] = calculate_tomograms(I, C, q, qmask, R, dq, r, v = False) 
        W[: dd] /= wsums[r, None]
          
        K[: dd] = data_getter[dstart:dstop, pixels]
        
        Kw[dstart: dstop] = np.sum(W[: dd] * K[: dd], axis = 1)
    
    # calculate log multiplicity
    #m = scipy.special.factorial(ksums) / np.prod(scipy.special.factorial(K))
    # logm = log(K_d!) - sum_i log(K_di!)
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1)
    ax.scatter(ksums, Npix * Kw / ksums, alpha = 0.7, picker = 2, s = 10, label = 'log-likelihood per photon vs photon counts', cmap = 'viridis')

    import extra_geom
    geom_fnam = f'crystfel_geom_0087.geom'
    geom = extra_geom.DSSC_1MGeometry.from_crystfel_geom(geom_fnam)
    
    frame, centre = geom.position_modules(data_getter[:1, :].reshape((16, 128, 512)))
    frame.fill(0)
    
    import pyqtgraph as pg
    def on_pick(event):
        i = event.ind[0]
        geom.position_modules(data_getter[i:i+1, :].reshape((16, 128, 512)), out = frame)
        pg.show(frame.copy()**0.1)
    
    fig.canvas.mpl_connect('pick_event', on_pick)
    
    ax.set_xscale('log')
    ax.set_ylabel('Log-R')
    ax.set_xlabel('photons in frame')
    ax.legend()
    
    plt.show()
