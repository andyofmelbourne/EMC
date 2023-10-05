import argparse
from pathlib import Path

if __name__ == '__main__':
    description = \
    """
    Update merged intensity. 
    """
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--mpx', type=int, default = 65, \
                        help="linear number of voxels for merged intensities.")
    parser.add_argument('-q', '--qmax', type=int,  \
                        help="maximum q-value for determining probabilities. Default is to set qmax equal to the merged intensity value. -1 sets qmax to the data limit. -2 sets qmax to the edge limit.")
    parser.add_argument('--qmin', type=int,  \
                        help="minimum q-value for determining probabilities. Default is to set qmin equal to the merged intensity value.")
    parser.add_argument('--ic', type=int, default=128, \
                        help="number of rotations to simultaneously hold in memory for inner loop.")
    parser.add_argument('-P', '--P_file', type=str, default='probability-matrix-merged_intensity.h5', \
                        help="probability matrix h5 file contaning logR values to normalise. For multiple files use coma separated list (no spaces)")
    parser.add_argument('-d', '--data', type=str, default='data.cxi', \
                        help="cxi file containing data frames, for geometry.")
    parser.add_argument('-T', '--data_T', type=str, default='data_T.h5', \
                        help="h5 file containing transposed data frames.")
    parser.add_argument('-o', '--output', type=str, default='merged_intensity.pickle', \
                        help="name of output python pickle file. For multiple files an index will be appended.")
    args = parser.parse_args()
    
    args.dataname = Path(args.data).stem


import sys
import h5py
import numpy as np
import tqdm
import scipy.constants as sc
import time
import pickle
import math
import os

import logR
from emc.tomograms import *
import emc.merge_tomos as merge_tomos

import pyclblast


if __name__ == '__main__':
    if args.qmin == None or args.qmax == None :
        # get previous merged intensity
        d  = pickle.load(open(args.output, 'rb'))
        
    # get q-mask etc.
    # ---------------
    if args.qmin == None :
        args.qmin = d['qmin']
    
    if args.qmax == None :
        args.qmax = d['qmax']
    
    qmask, q, C, qmin, qmax = logR.get_qmask(args.qmin, args.qmax, args.dataname)

    
    # get rotations from file or recalculate
    # --------------------------------------
    with h5py.File(args.P_file) as f :
        Ndata, Mrot = f['probability_matrix'].shape
        rot_order = f['rotation-order'][()]
        wsums      = f['tomogram_sums'][()].astype(np.float32)
        ksums      = f['photon_sums'][()].astype(np.float32)
        P          = np.ascontiguousarray(f['probability_matrix'][()].T.astype(np.float32))
    
    R, _ = logR.get_rotations(rot_order)
    
    ksums  = np.float32(ksums)
    Ndata  = np.int32(Ndata)
    Mrot   = np.int32(Mrot)
    Npix   = np.int32(np.sum(qmask))
    i0     = np.float32((args.mpx - 1)//2)
    M      = np.int32(args.mpx)
    dq     = np.float32(qmax / ((args.mpx-1)/2))

    U = math.ceil(Npix/args.ic)
    
    W    = np.empty((Mrot, args.ic), dtype = np.float32)
    Wd   = np.empty((Mrot, args.ic), dtype = np.float64)
    Ipix = np.empty((Mrot, args.ic), dtype = np.int32)
    K    = np.empty((Ndata, args.ic), dtype=np.float32)
    
    P_cl  = cl.array.empty(queue, (Mrot, Ndata), dtype=np.float32)
    K_cl  = cl.array.empty(queue, (Ndata, args.ic), dtype=np.float32)
    W_cl  = cl.array.empty(queue, (Mrot, args.ic), dtype = np.float32)
    C_cl  = cl.array.empty(queue, (Npix,), dtype = np.float32)
    PK_cl = cl.array.empty(queue, (Mrot,), dtype=np.float32)
    qx_cl = cl.array.empty(queue, (Npix,), dtype = np.float32)
    qy_cl = cl.array.empty(queue, (Npix,), dtype = np.float32)
    qz_cl = cl.array.empty(queue, (Npix,), dtype = np.float32)
    Ksums_cl = cl.array.empty(queue, (Ndata,), dtype=np.float32)
    wsums_cl = cl.array.empty(queue, (Mrot,), dtype = np.float32)
    Ipix_cl  = cl.array.empty(queue, (Mrot, args.ic), dtype = np.int32)
    R_cl     = cl.array.empty(queue, (9*Mrot,), dtype = np.float32)
    
    cl.enqueue_copy(queue, P_cl.data, P)
    cl.enqueue_copy(queue, Ksums_cl.data, np.ascontiguousarray(ksums.astype(np.float32)))
    cl.enqueue_copy(queue, wsums_cl.data, wsums)
    cl.enqueue_copy(queue, C_cl.data, np.ascontiguousarray(C[qmask].astype(np.float32)))
    cl.enqueue_copy(queue, R_cl.data, np.ascontiguousarray(R.astype(np.float32)))
    
    cl.enqueue_copy(queue, qx_cl.data, np.ascontiguousarray(q[0][qmask].astype(np.float32)))
    cl.enqueue_copy(queue, qy_cl.data, np.ascontiguousarray(q[1][qmask].astype(np.float32)))
    cl.enqueue_copy(queue, qz_cl.data, np.ascontiguousarray(q[2][qmask].astype(np.float32)))
    
    # could be faster on cpu...
    # calculate denominator PK_r = sum_d P_rd (sum_i K_di)
    for i in tqdm.tqdm(range(1), desc='calculating denominator: P . Ksum'):
        pyclblast.gemv(queue, Mrot, Ndata, P_cl, Ksums_cl, PK_cl, a_ld = Ndata)
    
    PK = PK_cl.get()
    
    MT = merge_tomos.Merge_tomos(W.shape, (M, M, M))
    
    load_time = 0
    dot_time = 0
    scale_time = 0
    merge_time = 0
    
    print('number or rotations        :', Mrot)
    print('number or data frames      :', Ndata)
    print('number or pixels in q-mask :', Npix)
    start_time = time.time()
    
    t = np.where(qmask.ravel())[0]
    inds_qmask = []
    for i in range(U):
        istart = i*args.ic
        istop  = min(istart + args.ic, Npix)
        inds_qmask.append(t[istart:istop])
     
    # loop over detector pixels
    for i in tqdm.tqdm(range(U), desc='generating tomograms'):
        di = min((i+1) * args.ic, Npix) - i*args.ic
        
        # copy data-pixels to gpu
        t0 = time.time()
        with h5py.File(args.data_T) as f:
            K[:, :di] = f['data_id'][inds_qmask[i], :].T
            cl.enqueue_copy(queue, K_cl.data, K)
        load_time += time.time() - t0
        
        # calculate dot product (tomograms) W_ri = sum_d P_rd K_di 
        t0 = time.time()
        pyclblast.gemm(queue, Mrot, di, Ndata, P_cl, K_cl, W_cl, a_ld = Ndata, b_ld = args.ic, c_ld = args.ic)
        queue.finish()
        dot_time += time.time() - t0
        
        
        # scale tomograms W_ri = (sum_i W^old_ri) x w_ri / (sum_d P_dr (sum_i K_di)) / (sold + pol. correction C_i)
        t0 = time.time()
        cl_code.scale_tomograms_for_merge_w_coffset( queue, (di,), None,
                                           W_cl.data, wsums_cl.data, PK_cl.data, C_cl.data,
                                           np.int32(args.ic), np.int32(0), np.int32(Mrot), np.int32(i*args.ic))
        queue.finish()
        scale_time += time.time() - t0
        
        # calculate tomogram to merged intensity pixel mappings
        t0 = time.time()
        cl_code.calculate_W_to_I_mapping_w_ioffset(queue, (di,), None,
                                         Ipix_cl.data, R_cl.data, qx_cl.data, qy_cl.data, qz_cl.data, dq, 
                                         i0, np.int32(args.ic), M, np.int32(Mrot), np.int32(i*args.ic))
        
        cl.enqueue_copy(queue, W, W_cl.data)
        cl.enqueue_copy(queue, Ipix, Ipix_cl.data)

        
        # merge: can I do this on the gpu?
        merge_tomos.queue.finish()
        Wd[:] = W[:]
        MT.merge(Wd, Ipix, Mrot, di, is_blocking=False)
        merge_time += time.time() - t0
        
    
    
    I, O = MT.get_I_O()
    
    O[O==0] = 1
    I /= O

    pickle.dump({'qmax': qmax, 'qmin': qmin, 'dq': dq, 'I': I, 't': 0, 'sample-state': 0, 'data-set': args.data}, open(args.output, 'wb'))
     
    # record merged intensity for each iteration
    pickle.dump(I, open('past-merged_intensities.pickle', 'ab'))
    
    total_time = time.time() - start_time 
    print('\n')
    print('total time: {:.2e}s'.format(total_time))
    print('\n')
    print('total time seconds')
    print('load  time: {:.1e}'.format( load_time))
    print('dot   time: {:.1e}'.format( dot_time))
    print('scale time: {:.1e}'.format( scale_time))
    print('merge time: {:.1e}'.format( merge_time))
    print('\n')
    print('total time %')
    print('load  time: {:5.1%}'.format( load_time / total_time))
    print('dot   time: {:5.1%}'.format( dot_time / total_time))
    print('scale time: {:5.1%}'.format( scale_time / total_time))
    print('merge time: {:5.1%}'.format( merge_time / total_time))
    print('\n')
    print("These numbers shouldn't change unless things are being done more efficiently:")
    print('load  time / Ndata / Npix        : {:.2e}'.format( load_time / Ndata / Npix))
    print('dot   time / Ndata / Mrot / Npix : {:.2e}'.format( dot_time / Ndata / Mrot / Npix))
    print('merge time / Mrot / Npix         : {:.2e}'.format( 100 * merge_time / Mrot / Npix))

