import argparse
from pathlib import Path

if __name__ == '__main__':
    description = \
    """
    Update merged intensity. 
    """
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--mpx', type=int,  \
                        help="linear number of voxels for merged intensities. Default is the value in the last merged intensity pickle file.")
    parser.add_argument('-q', '--qmax', type=int,  \
                        help="maximum q-value for determining probabilities. Default is to set qmax equal to the merged intensity value. -1 sets qmax to the data limit. -2 sets qmax to the edge limit.")
    parser.add_argument('--qmin', type=int,  \
                        help="minimum q-value for determining probabilities. Default is to set qmin equal to the merged intensity value.")
    parser.add_argument('--rc', type=int, default=1024, \
                        help="number of rotations to simultaneously hold in memory for inner loop.")
    parser.add_argument('--ic', type=int, default=1024, \
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

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

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
    if rank == 0 :
        if args.qmin == None or args.qmax == None or args.mpx == None:
            # get previous merged intensity
            d  = pickle.load(open(args.output, 'rb'))
            
        # get q-mask etc.
        # ---------------
        if args.qmin == None :
            args.qmin = d['qmin']
        
        if args.qmax == None :
            args.qmax = d['qmax']
        
        if args.mpx == None :
            args.mpx = d['I'].shape[0]
        
        qmask, q, C, qmin, qmax = logR.get_qmask(args.qmin, args.qmax, args.dataname)

        # get rotations from file or recalculate
        # --------------------------------------
        with h5py.File(args.P_file) as f :
            Ndata, Mrot = f['probability_matrix'].shape
            rot_order = f['rotation-order'][()]
            wsums      = f['tomogram_sums'][()]
            ksums      = f['photon_sums'][()]
            # 2GB limit for mpi 
            #P          = np.ascontiguousarray(f['probability_matrix'][()].T.astype(np.float32))
    
        R, _ = logR.get_rotations(rot_order)
    
    else :
        qmask = q = C = qmin = qmax = ksums = Ndata = Mrot = rot_order = wsums = R = P = None

    qmask = comm.bcast(qmask, root=0)
    q     = comm.bcast(q, root=0)
    C     = comm.bcast(C, root=0)
    qmin  = comm.bcast(qmin, root=0)
    qmax  = comm.bcast(qmax, root=0)
    Ndata = comm.bcast(Ndata, root=0)
    Mrot  = comm.bcast(Mrot, root=0)
    ksums = comm.bcast(ksums, root=0)
    wsums = comm.bcast(wsums, root=0)
    R     = comm.bcast(R, root=0)
    #P     = comm.bcast(P, root=0)
    rot_order = comm.bcast(rot_order, root=0)
    args.mpx = comm.bcast(args.mpx, root=0)
    
    with h5py.File(args.P_file) as f :
        P = np.ascontiguousarray(f['probability_matrix'][()].T.astype(np.float32))

    P_buf = np.empty((args.rc, Ndata), dtype=np.float32)


    
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
    
    P_cl  = cl.array.empty(queue, (args.rc, Ndata), dtype=np.float32)
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
    
    cl.enqueue_copy(queue, Ksums_cl.data, np.ascontiguousarray(ksums.astype(np.float32)))
    cl.enqueue_copy(queue, wsums_cl.data, np.ascontiguousarray(wsums.astype(np.float32)))
    cl.enqueue_copy(queue, C_cl.data, np.ascontiguousarray(C[qmask].astype(np.float32)))
    cl.enqueue_copy(queue, R_cl.data, np.ascontiguousarray(R.astype(np.float32)))
    
    cl.enqueue_copy(queue, qx_cl.data, np.ascontiguousarray(q[0][qmask].astype(np.float32)))
    cl.enqueue_copy(queue, qy_cl.data, np.ascontiguousarray(q[1][qmask].astype(np.float32)))
    cl.enqueue_copy(queue, qz_cl.data, np.ascontiguousarray(q[2][qmask].astype(np.float32)))
    

    PK = P.dot(ksums).astype(np.float32)
    cl.enqueue_copy(queue, PK_cl.data, PK)
    
    PK_on_W_r = PK / wsums
     
    MT = merge_tomos.Merge_tomos(W.shape, (M, M, M))
    
    load_time = 0
    dot_time = 0
    scale_time = 0
    merge_time = 0
    
    if rank == 0 :
        print('number of rotations        :', Mrot)
        print('number of data frames      :', Ndata)
        print('number of pixels in q-mask :', Npix)
    
    start_time = time.time()
    
    i_iter = list(range(0, Npix, args.ic))[rank::size]
    
    Rrot = math.ceil(Mrot/args.rc)
    if rank == 0 :
        r_iter = tqdm.tqdm(range(Rrot), desc='generating tomograms')
    else :
        r_iter = range(Rrot)

    Kinds = np.arange(qmask.size).reshape(qmask.shape)
    qinds = Kinds[qmask]

    for r in r_iter:
        rstart = r*args.rc
        rstop  = min(rstart + args.rc, Mrot)
        dr     = rstop - rstart
         
        P_buf[:dr] = P[rstart:rstop, :]
        cl.enqueue_copy(queue, P_cl.data, P_buf)
         
        # loop over detector pixels
        for i in i_iter :
            istart = i
            istop  = min(i + args.ic, Npix)
            di     = istop - istart 
            
            # copy data-pixels to gpu
            t0 = time.time()
            with h5py.File(args.data_T) as f:
                if di != args.ic :
                    K.fill(0)
                K[:, :di] = f['data_id'][qinds[istart:istop], :].T
                cl.enqueue_copy(queue, K_cl.data, K)
            load_time += time.time() - t0

            # calculate dot product (tomograms) W_ri = sum_d P_rd K_di 
            t0 = time.time()
            pyclblast.gemm(queue, dr, di, Ndata, P_cl, K_cl, W_cl, a_ld = Ndata, b_ld = args.ic, c_ld = args.ic, c_offset = args.ic * rstart)
            queue.finish()
            dot_time += time.time() - t0
            
            # scale tomograms w_ri <-- w_ri / (sold + pol. correction C_i)
            t0 = time.time()
            cl_code.scale_tomograms_for_merge_w_coffset( queue, (di,), None,
                                               W_cl.data, C_cl.data,
                                               np.int32(args.ic), np.int32(rstart), np.int32(rstop), np.int32(istart))
            queue.finish()
            scale_time += time.time() - t0
            
            
            # calculate tomogram to merged intensity pixel mappings
            t0 = time.time()
            cl_code.calculate_W_to_I_mapping_w_ioffset(queue, (di,), None,
                                             Ipix_cl.data, R_cl.data, qx_cl.data, qy_cl.data, qz_cl.data, dq, 
                                             i0, np.int32(args.ic), M, np.int32(rstart), np.int32(rstop), np.int32(istart))
            
            # should reduce W and Ipix to rc buffers to cut down on transfer...
            cl.enqueue_copy(queue, W, W_cl.data)
            cl.enqueue_copy(queue, Ipix, Ipix_cl.data)
            
            # merge tomograms Isum[n] +=  sum_d P_rd K_di[n] / (sold + pol. correction C_i[n])
            #                 O[n]    +=  sum_d P_rd sum_i K_di / sum_i Wold_ri
            #                                      PK_r         /    wsums_r
            
            # merge: can I do this on the gpu?
            merge_tomos.queue.finish()
            Wd[:] = W[:]
             
            MT.merge(Wd, Ipix, rstart, rstop, PK_on_W_r, di, is_blocking=False)
            merge_time += time.time() - t0
    
    I, O = MT.get_I_O()
    
    O = comm.reduce(O, op=MPI.SUM, root=0)
    I = comm.reduce(I, op=MPI.SUM, root=0)

    if rank == 0 :
        overlap = O.copy()
        Isum = I.copy()
        O[O==0] = 1
        I /= O

        pickle.dump({'qmax': qmax, 'qmin': qmin, 'dq': dq, 'I': I, 'Isum': Isum, 'overlap': overlap, 't': 0, 'sample-state': 0, 'data-set': args.data}, open(args.output, 'wb'))
         
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

