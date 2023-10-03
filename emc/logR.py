import argparse
from pathlib import Path

if __name__ == '__main__':
    description = \
    """
    Calculate logR values for EMC.
    """
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-M', '--rotations', type=int, \
                        help="Number of rotations (Mrot) = pi^2 M^3. M is the diameter of the sphere, in pixel units, that will be completely filled by Mrot rotations. If not set, then M will be set to mpx.")
    parser.add_argument('-q', '--qmax', type=int,  \
                        help="maximum q-value for determining probabilities. Default is to set qmax equal to the merged intensity value.")
    parser.add_argument('--qmin', type=int,  \
                        help="minimum q-value for determining probabilities. Default is to set qmin equal to the merged intensity value.")
    parser.add_argument('-s', '--tomo_scale', type=float,  \
                        help="scale tomograms by tomo_scale before computing log. Default: tomo_scale = number of pixels in q-mask")
    parser.add_argument('--rc', type=int, default=128, \
                        help="number of rotations to simultaneously hold in memory for inner loop.")
    parser.add_argument('--dc', type=int, default=128, \
                        help="number of data frames to simultaneously hold in memory for inner loop.")
    parser.add_argument('-I', '--merged_intensity', type=str, default='merged_intensity.pickle', \
                        help="filename of the python pickle file containing merged intensities.")
    parser.add_argument('-d', '--data', type=str, default='data.cxi', \
                        help="cxi file containing data frames")
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

import pickle 
import numpy as np
import os
# this is so we also load opencl stuff
from emc.tomograms import * 
import h5py
import tqdm
import pyclblast
import time

def get_rotations(M):
    M_in_plane = int(np.pi * M)+1
    M_sphere   = int(np.pi * M**2)+1
    Mrot = M_in_plane * M_sphere
        
    # if already written then grab it
    fnam = f'rotations-{M}.pickle'
    if os.path.exists(fnam) :
        print(f'found precalculated rotations in {fnam}.')
        R = pickle.load(open(fnam, 'rb'))
    else :
        print(f'did not find {fnam}. recalculating rotations')
        from emc.calculate_rotations import calculate_rotation_matrices
        R = calculate_rotation_matrices(Mrot, M_in_plane, M_sphere)
        
        # output
        print(f'saving rotations for later in {fnam}')
        pickle.dump(R, open(fnam, 'wb'))
    return R, M

def get_qmask(qmin, qmax, dataname):
    fnam = 'solid-angle-polarisation-'+dataname+'.pickle'

    if os.path.exists(fnam) :
        d = pickle.load(open(fnam, 'rb'))

        if qmax == -2 :
            qmax = d['q-edge']
        elif qmax == -1 :
            qmax = d['q-corner']
        
        q     = d['q'].copy()
        qr    = np.sum(q**2, axis=0)**0.5
        qmask = d['mask'].copy()
        qmask[qr > qmax] = False
        qmask[qr < qmin] = False
    
    else :
        raise Exception(f'could not find {fnam}! \n Have you run init_emc.py?')
    
    return qmask, q, d['C'].copy(), qmin, qmax
    

def get_non_zero(fnam_cxi, qmin, qmax, qmask):
    dataname = Path(fnam_cxi).stem
    
    qmaxint = int(qmax)
    qminint = int(qmin)
    fnam = f'non-zero-photons-{dataname}-{qminint}-{qmaxint}.pickle'
    if os.path.exists(fnam) :
        d = pickle.load(open(fnam, 'rb'))
        K, locs, ksums = d['K'].copy(), d['inds'].copy(), d['ksums'].copy()
    else :
        # output unmasked non-zero pixels and indices to pickle file
        K = {}
        locs   = {}
        ksums = []
        inds      = np.arange(np.sum(qmask), dtype=np.uint32)
        with h5py.File(fnam_cxi, 'r') as f:
            data = f['entry_1/data_1/data']
            shape = data.shape
            
            for d in tqdm.tqdm(range(0, shape[0]), desc='writing non-zero photon counts to pickle file'):
                Kd = data[d]
                frame = Kd[qmask]
                m     = frame > 0 
                K[d] = frame[m]
                locs[d] = inds[m.ravel()]
                
                ksums.append(np.sum(K[d]))
        
        pickle.dump({'K': K, 'inds': locs, 'ksums': np.array(ksums)}, open(fnam, 'wb'))
    return K, locs, ksums

def check_output_file(output, Ndata, Mrot):
    # if the number of rotations has changed, then delete 
    # output file and re-create (h5 files grow large if you add / remove datasets)
    if os.path.exists(output): 
        remove = False
        with h5py.File(output) as f:
            if 'probability_matrix' not in f or f['probability_matrix'].shape != (Ndata, Mrot):
                remove = True

        if remove :
            print(f'{output} found but with the wrong shaped data. removing.')
            os.remove(output)
    
    if not os.path.exists(output): 
        print(f'{output} not found. creating.')
        with h5py.File(output, 'w') as f:
            f.create_dataset('logR', 
                shape = (Ndata, Mrot), 
                dtype=float)
            f.create_dataset('probability_matrix', 
                shape = (Ndata, Mrot), 
                dtype=float)
            f.create_dataset('tomogram_sums', 
                shape = (Mrot,), 
                dtype=float)
            f['qmin'] = 0.
            f['qmax'] = 0.
            f['rotation-order'] = 1
    

if __name__ == '__main__':
    # get merged intensity
    d  = pickle.load(open(args.merged_intensity, 'rb'))
    I  = d['I'].copy()
    dq = d['dq']
    
    # get rotations from file or recalculate
    # --------------------------------------
    if args.rotations == None :
        args.rotations = I.shape[0]
    
    R, rot_order = get_rotations(args.rotations)
    
    # get q-mask etc.
    # ---------------
    if args.qmax == None :
        qmax = d['qmax']
    else :
        qmax = args.qmax
    
    if args.qmin == None :
        qmin = d['qmin']
    else :
        qmin = args.qmin
    
    qmask, q, C, qmin, qmax = get_qmask(qmin, qmax, args.dataname)

    # check that we are sampling within the domain of the merged intensities
    if qmax > d['qmax'] :
        raise Exception('qmax is greater than merged intensity-boundary')
    
    if qmin < d['qmin'] :
        raise Exception('qmin is less than merged intensity inner-boundary')
    
    # precalculate tomogram sums
    # --------------------------
    wsums = calculate_tomogram_sums(I, C, q, qmask, R, dq)

    # get non-zero photons within qmask from pickle file
    # and make one if necessary
    # --------------------------------------------------
    Knz, inds, ksums = get_non_zero(args.data, qmin, qmax, qmask)
    
    Ndata     = len(Knz)
    Mrot      = np.int32(R.shape[0])
    Npix      = np.int32(np.sum(qmask))
    
    check_output_file(args.output, Ndata, Mrot)
    
    # write tomogram sums to file 
    print(f'writing tomogram sums to {args.output}')
    with h5py.File(args.output, 'a') as f:
        f['tomogram_sums'][...] = wsums
        f['qmin'][...] = qmin
        f['qmax'][...] = qmax
        f['rotation-order'][...] = rot_order
    
    # caluclate logR_dr = sum_i K_di log( w_ri )
    # ------------------------------------------    
    # pixel coordinate of q = 0 in I
    i0 = np.float32((I.shape[0] - 1)//2)

    # scale tomograms
    if args.tomo_scale == None :
        wscale = (Npix / wsums).astype(np.float32)
    else :
        wscale = (args.tomo_scale / wsums).astype(np.float32)
    
    K          = np.empty((args.dc, Npix), dtype=np.float32)
    logR       = np.empty((args.dc, Mrot))
    logR_chunk = np.empty((args.dc, args.rc), dtype=np.float32)
    
    R_cl      = cl.array.empty(queue, (9*args.rc,), dtype = np.float32)
    K_cl      = cl.array.empty(queue, (args.dc, Npix), dtype=np.float32)
    wscale_cl = cl.array.empty(queue, (args.rc,), dtype = np.float32)
    qx_cl     = cl.array.empty(queue, (Npix,), dtype = np.float32)
    qy_cl     = cl.array.empty(queue, (Npix,), dtype = np.float32)
    qz_cl     = cl.array.empty(queue, (Npix,), dtype = np.float32)
    C_cl      = cl.array.empty(queue, (Npix,), dtype = np.float32)
    W_cl      = cl.array.empty(queue, (args.rc, Npix), dtype = np.float32)
    logR_cl   = cl.array.empty(queue, (args.dc, args.rc), dtype=np.float32)
    
    cl.enqueue_copy(queue, qx_cl.data, np.ascontiguousarray(q[0][qmask].astype(np.float32)))
    cl.enqueue_copy(queue, qy_cl.data, np.ascontiguousarray(q[1][qmask].astype(np.float32)))
    cl.enqueue_copy(queue, qz_cl.data, np.ascontiguousarray(q[2][qmask].astype(np.float32)))
    cl.enqueue_copy(queue, C_cl.data,  np.ascontiguousarray(C[qmask].astype(np.float32)))

    # copy I as an opencl "image" for trilinear sampling
    I_cl = cl.Image(context, cl.mem_flags.READ_ONLY, cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT), shape=I.shape[::-1])
    cl.enqueue_copy(queue, I_cl, I.T.copy().astype(np.float32), is_blocking=True, origin=(0, 0, 0), region=I.shape[::-1])
    
    load_time = 0
    tomo_time = 0
    dot_time = 0

    print('number or rotations        :', Mrot)
    print('number or data frames      :', Ndata)
    print('number or pixels in q-mask :', Npix)
    
    for d in tqdm.tqdm(range(0, Ndata, args.dc), desc='calculating probabilities'):
        dd = min(Ndata, d+args.dc) - d
        
        # load data onto gpu
        t0 = time.time()
        K.fill(0)
        for f in range(d, d+dd, 1):
            K[f-d][inds[f]] = Knz[f]
        
        cl.enqueue_copy(queue, K_cl.data, K)
        load_time += time.time() - t0
        
        for r in range(0, Mrot, args.rc):
            dr = min(Mrot, r+args.rc) - r
            
            # load rotation matrices and wscale onto gpu
            t0 = time.time()
            cl.enqueue_copy(queue, R_cl.data, R[r:r+dr])
            cl.enqueue_copy(queue, wscale_cl.data, wscale[r:r+dr])
            load_time += time.time() - t0
             
            # calculate all tomograms from r -> r + dr 
            t0 = time.time()
            cl_code.calculate_tomogram_w_scale_log_batch(queue, (Npix,), None, 
                I_cl, C_cl.data, qx_cl.data, qy_cl.data, qz_cl.data, 
                R_cl.data, W_cl.data, i0, np.float32(dq), 
                wscale_cl.data, Npix, np.int32(0), np.int32(dr))
            tomo_time += time.time() - t0
             
            # calculate dot product: logR_dr = sum_i K_di log(w_ri)
            #                           w_ri = tomo_scale * W_ri / sum_i W_ri 
            queue.finish()
            t0 = time.time()
            pyclblast.gemm(queue, dd, dr, Npix, K_cl, W_cl, logR_cl, a_ld=Npix, b_ld=Npix, c_ld = args.rc, b_transp=True)
            queue.finish()
            dot_time += time.time() - t0
            
            # copy to cpu
            cl.enqueue_copy(queue, logR_chunk, logR_cl.data)
            logR[:dd, r:r+dr] = logR_chunk[:dd, :dr]
            
        assert(np.all(np.isfinite(logR)))
            
        # write probabilities to file
        # (not probabilities yet, still need to be normalised)
        with h5py.File(args.output, 'a') as f:
            f['logR'][d:d+dd] = logR[:dd]
    
    print('\n')
    print('load  time:', 1e6 * load_time / Mrot / Ndata, 'ms')
    print('dot   time:', 1e6 * dot_time / Mrot / Ndata, 'ms')
    print('tomo  time:', 1e6 * tomo_time / Mrot / Ndata, 'ms')
    print('\n')
