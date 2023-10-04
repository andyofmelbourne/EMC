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

def write_pixel_chunks(ic, fnam, dataname, qmin, qmax, qmask, Npix, cachedir='cache'):
    """
    ic = chunk size (no of det pixels)
    
    write pixel chunks to a cache dir for fast retrieval:
        cache/{i}_{ic}_{qmin}_{qmax}_{dataname}.h5
    
    keep chunks in separate files, so that different processes 
    can read simultaneously (does this make much difference?)
    """
    # check if cache exists
    if not os.path.exists(cachedir) :
        os.mkdir(cachedir)
    
    U = math.ceil(Npix/ic)
    
    fnams = [f'{cachedir}/{i}_{ic}_{qmin}_{qmax}_{dataname}.h5' for i in range(U)]

    # if fnams already exists and were completed then exit
    chunk_exist = np.array([os.path.exists(f) for f in fnams])
    
    # check if they finished
    chunk_done = np.zeros(len(fnams), dtype=bool)
    for i in range(len(fnams)) :
        if chunk_exist[i] :
            with h5py.File(fnams[i]) as f:
                chunk_done[i] = f['done'][()]

    if np.all(chunk_done) :
        print(f'found all pixels chunks in {cachedir} for this qmask')
        return fnams
    else :
        print(f'did not found all pixels chunks in {cachedir} for this qmask')
         
        # edge case where some of the files exist, due to a previously aborted run
        for i in np.where(chunk_exist)[0]:
            os.remove(fnams[i])
    
    with h5py.File(fnam) as f:
        Ndata = f['entry_1/data_1/data'].shape[0]
    
    print(f'pixel chunk size {Ndata}x{ic} = {int(Ndata*ic*4/1024**2)} megabytes')
    
    # create files and datasets
    chunk_h5s = [h5py.File(f, 'w-') for f in fnams]
    for g in chunk_h5s:
        g.create_dataset('data', shape = (Ndata, ic), dtype=np.float32)
        g['done'] = False
            
    # read data and split pixels 
    #Ks = np.zeros((U, Ndata, args.ic), dtype=np.float32)
    with h5py.File(fnam) as f:
        data = f['entry_1/data_1/data']
            
        for d in tqdm.tqdm(range(data.shape[0]), desc='splitting pixels'):
            K = data[d]
            
            for u in range(U):
                istart = u * ic
                istop  = min(istart+ic, Npix)
                chunk_h5s[u]['data'][d, :istop-istart] = K[qmask][istart:istop]

    # close files
    for g in chunk_h5s:
        g['done'][...] = True
        g.close()

    return fnams
    


# GEMMK=0 KREG=1 KWG=32 KWI=2 MDIMA=8 MDIMC=8 MWG=64 NDIMB=8 NDIMC=8 NWG=64 PRECISION=32 SA=1 SB=1 STRM=0 STRN=0 VWM=1 VWN=4
#params ={"GEMMK": 0,"KREG": 1,"KWG": 32,"KWI": 2,"MDIMA": 8,"MDIMC": 8,"MWG": 64,"NDIMB": 8,"NDIMC": 8,"NWG": 64,"PRECISION": 32,"SA": 1,"SB": 1,"STRM": 0,"STRN": 0,"VWM": 1,"VWN": 4} 

# causes errors... might have to look into cuda for nvidia stuff
#params = {"GEMMK": 1,"KREG": 8,"KWG": 1,"KWI": 1,"MDIMA": 8,"MDIMC": 8,"MWG": 32,"NDIMB": 8,"NDIMC": 8,"NWG": 64,"PRECISION": 32,"SA": 0,"SB": 0,"STRM": 0,"STRN": 0,"VWM": 4,"VWN": 2}
#pyclblast.override_parameters(context.devices[0], 'Xgemm', 32, params)

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

    # get non-zero photons within qmask from pickle file
    # and make one if necessary
    # --------------------------------------------------
    Knz, inds, ksums = logR.get_non_zero(args.data, qmin, qmax, qmask)
    
    # get rotations from file or recalculate
    # --------------------------------------
    with h5py.File(args.P_file) as f :
        Ndata, Mrot = f['probability_matrix'].shape
        rot_order = f['rotation-order'][()]
        wsums      = f['tomogram_sums'][()].astype(np.float32)
        P          = np.ascontiguousarray(f['probability_matrix'][()].T.astype(np.float32))
    
    R, _ = logR.get_rotations(rot_order)
    
    ksums  = np.float32(ksums)
    Ndata  = np.int32(Ndata)
    Mrot   = np.int32(Mrot)
    Npix   = np.int32(np.sum(qmask))
    i0     = np.float32((args.mpx - 1)//2)
    M      = np.int32(args.mpx)
    dq     = np.float32(qmax / ((args.mpx-1)/2))

    # chunk pixels into a cache if not already done 
    chunk_fnams = write_pixel_chunks(args.ic, args.data, args.dataname, qmin, qmax, qmask, Npix)
    
    U = math.ceil(Npix/args.ic)
    
    W    = np.empty((Mrot, args.ic), dtype = np.float32)
    Wd   = np.empty((Mrot, args.ic), dtype = np.float64)
    Ipix = np.empty((Mrot, args.ic), dtype = np.int32)
    
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
    #assert(np.allclose(PK_cl.get(), P.dot(np.ascontiguousarray(ksums.astype(np.float32)))))
    PK = PK_cl.get()
    
    MT = merge_tomos.Merge_tomos(W.shape, (M, M, M))
    
    load_time = 0
    dot_time = 0
    merge_time = 0
    
    print('number or rotations        :', Mrot)
    print('number or data frames      :', Ndata)
    print('number or pixels in q-mask :', Npix)
    start_time = time.time()
    
    # test
    #Wk = np.zeros((100, 4, 512, 512), dtype=np.float32)
    #t = np.zeros((100, Npix))
     
    # loop over detector pixels
    for i in tqdm.tqdm(range(U), desc='generating tomograms'):
        di = min((i+1) * args.ic, Npix) - i*args.ic
        
        # copy data-pixels to gpu
        t0 = time.time()
        with h5py.File(chunk_fnams[i]) as f:
            cl.enqueue_copy(queue, K_cl.data, f['data'][()])
        load_time += time.time() - t0
        
        # calculate dot product (tomograms) W_ri = sum_d P_rd K_di 
        pyclblast.gemm(queue, Mrot, di, Ndata, P_cl, K_cl, W_cl, a_ld = Ndata, b_ld = args.ic, c_ld = args.ic)
        
        #queue.finish()
        #assert(np.allclose(W_cl.get()[:, :di], P_cl.get().dot(K_cl.get())[:, :di]))
        
        # test
        #cl.enqueue_copy(queue, W, W_cl.data)
        #cl.enqueue_copy(queue, W, W_cl.data)
        #t[:100, i*args.ic: i*args.ic + di] = W[:100, :di]
        
        # scale tomograms W_ri = (sum_i W^old_ri) x w_ri / (sum_d P_dr (sum_i K_di)) / (sold + pol. correction C_i)
        t0 = time.time()
        cl_code.scale_tomograms_for_merge_w_coffset( queue, (di,), None,
                                           W_cl.data, wsums_cl.data, PK_cl.data, C_cl.data,
                                           np.int32(args.ic), np.int32(0), np.int32(Mrot), np.int32(i*args.ic))
        queue.finish()
        dot_time += time.time() - t0
        
        #queue.finish()
        #nmax = Mrot * di
        #t = W.ravel()[:nmax].reshape(Mrot, di)
        #t *= wsums[:, None] / PK[:, None] / C[qmask][i*args.ic : i*args.ic + di].astype(np.float32)
        #assert(np.allclose(W_cl.get().ravel()[:nmax], t.ravel()))
        
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
        
    
    #import pickle 
    #for ii in range(Wk.shape[0]):
    #    Wk[ii][qmask] = t[ii]
    #pickle.dump(Wk, open('tomograms_split_pix.pickle', 'wb'))
    
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
    print('total time %')
    print('load  time: {:5.1%}'.format( load_time / total_time))
    print('dot   time: {:5.1%}'.format( dot_time / total_time))
    print('merge time: {:5.1%}'.format( merge_time / total_time))
    print('\n')
    print("These numbers shouldn't change unless things are being done more efficiently:")
    print('load  time / Ndata / (Mrot/rc) / Npix : {:.2e}'.format( load_time / Ndata / Npix))
    print('dot   time / Ndata / Mrot / Npix      : {:.2e}'.format( dot_time / Ndata / Mrot / Npix))
    print('merge time / Mrot / Npix              : {:.2e}'.format( 100 * merge_time / Mrot / Npix))


# test matvec: dodgy
"""
# 10080, 9033, 2559

Mrot2, Ndata2, di2 = 10080, 9033, 2559
P2 = np.random.random((Mrot2, Ndata2)).astype(np.float32)
K2 = np.random.random((Ndata2, di2)).astype(np.float32)
w2 = np.empty((Mrot2, di2), dtype=np.float32)

P2_cl = cl.array.to_device(queue, P2)
K2_cl = cl.array.to_device(queue, K2)
w2_cl = cl.array.to_device(queue, w2)

queue.finish()
pyclblast.gemm(queue, Mrot2, di2, Ndata2, P2_cl, K2_cl, w2_cl, a_ld = Ndata2, b_ld = di2, c_ld = di2)
queue.finish()

w2 = P2.dot(K2)
assert(np.allclose(w2_cl.get(), w2))
"""
