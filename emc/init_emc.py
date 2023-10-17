import argparse
from pathlib import Path

if __name__ == '__main__':
    description = \
    """
    Generate random merged-intensities for EMC and initialise emc.h5 file for further processing.
    """
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', type=str, default='data.cxi', \
                        help="cxi file(s) containing data frames. For multiple files use coma separated list with no spaces.")
    parser.add_argument('-q', '--qmax', type=int, default = -1, \
                        help="maximum q-value for merged intensities. -1 sets qmax to the data limit. -2 sets qmax to the edge limit.")
    parser.add_argument('--qmin', type=int, default = 0, \
                        help="minimum q-value for merged intensities.")
    parser.add_argument('-m', '--mpx', type=int, default = 65, \
                        help="linear number of voxels for merged intensities.")
    parser.add_argument('-p', '--polarisation', type=str, default = 'x', \
                        help="polarisation direction: x, y or None.")
    parser.add_argument('-s', '--sample_states', type=int, default = 1, \
                        help="number of discrete intensity models.")
    parser.add_argument('-o', '--output', type=str, default='merged_intensity.pickle', \
                        help="name of output python pickle file. For multiple files an index will be appended.")
    args = parser.parse_args()
    args.data = args.data.split(',')


import h5py
import numpy as np
import scipy.constants as sc
import pickle
import os, sys

if __name__ == '__main__':
    # remove old files
    for fnam in ['past-merged_intensities.pickle', 'merged_intensities.pickle', 'most_likely_orientations.pickle']:
        try:
            os.remove(fnam)
            
        except FileNotFoundError :
            pass
    
    # determine qmax and output q-vals
    # --------------------------------
    qmax_max = qmax = 0
    for fnam in args.data :
        with h5py.File(fnam, 'r') as f:
            # pixel mask
            mask = f['entry_1/instrument_1/detector_1/mask'][()]

            # pixel map
            xyz  = f['/entry_1/instrument_1/detector_1/xyz_map'][()]
            E    = np.mean(f['/entry_1/instrument_1/source_1/energy'][()])
            dx   = f['entry_1/instrument_1/detector_1/x_pixel_size'][()]
            dy   = f['entry_1/instrument_1/detector_1/y_pixel_size'][()]
            
        # Determine q, dq, qmax, qmask and correction (data to merge)
        # -----------------------------------------------------------
        wav = sc.h * sc.c / E
        r = np.sum(xyz**2, axis=0)**0.5
        q = xyz.copy() / r
        q[2] -= 1
        q /= wav

        r = np.sum(xyz**2, axis=0)**0.5
            
        if args.polarisation == 'x' :
            P = 1 - (xyz[0] / r)**2
        elif args.polarisation == 'y' :
            P = 1 - (xyz[1] / r)**2
        elif args.polarisation == 'None' :
            P = np.ones(dshape[1:])
        
        # solid angle correction  
        Omega = dx * dy * xyz[2] / r**3
        
        # merged intensity to frame correction factor
        C = Omega * P
        
        # scale 
        C /= C[mask].max()

        qr = np.sum(q**2, axis=0)**0.5
        
        q_corner = qr[mask].max()
        q_edge   = max(np.abs(q[0][mask]).max(), np.abs(q[1][mask]).max())
        # only true at low angles, and for square pixels
        dq_data  = dx / (dx**2 + xyz[2].min()**2)**0.5 / wav
        
        print("q-values:")
        print("edge          :", int(q_edge), '1/m')
        print("corner        :", int(q_corner), '1/m')
        print("dq (approx.)  :", int(dq_data), '1/m')
        print("\n")
        print("full period resolution (nm):")
        print("edge          : {:.2e} nm".format(1e9/q_edge))
        print("corner        : {:.2e} nm".format(1e9/q_corner))
        
        # maximum un-masked qvalue on detector
        if args.qmax == -1 :
            qmax = max(qmax, q_corner)
            
        # maximum un-masked qvalue along x-y axes
        elif args.qmax == -2 :
            qmax = max(qmax, q_edge)

        else :
            qmax = args.qmax

        if qmax > qmax_max :
            qmax_max = qmax
        
        qmin = args.qmin
        
        # output
        dataname = Path(fnam).stem
        fnam_out = f'solid-angle-polarisation-{dataname}.pickle'
        pickle.dump({'mask': mask, 'q': q, 'C': C, 'solid-angles': Omega, 'polarisation': P, 'data-set': fnam, 'q-corner': q_corner, 'q-edge': q_edge}, open(fnam_out, 'wb'))
    
    # transpose data for faster pixel chunks
    # --------------------------------------
    #for fnam in args.data :
    #    dataname = Path(fnam).stem
    #    with h5py.File(fnam, 'r') as f:
            
    # use np.fft.fftreq style indexing
    # so for n = mpx even:
    # |q|max = qmax = (n / 2 - 1) / (d n)
    #              and so d = (n / 2 - 1) / (qmax n) 
    # dq = 1 / (d n) = qmax n / (n (n / 2 - 1))
    # 
    # so for n = mpx odd:
    # |q|max = qmax = (n-1) / (2 d n) 
    #                       and so d = (n-1) / (2 n qmax)
    # dq = 1 / (d n) = (2 qmax) / (n-1)
    if (args.mpx % 2) == 0 :
        dq = (args.mpx / 2 - 1) / (qmax n)
    else :
        dq = 2 * qmax_max / (args.mpx - 1)
    
    # output initial merged intensities
    # ---------------------------------
    T = len(args.data) * args.sample_states
    
    print("\n")
    print("Merged intensity:")
    print("pixel size    : {:.2e} nm".format(1e9/qmax_max))
    print("field-of-view : {:.2e} nm".format(1e9/dq))
    print("dq            :", int(dq))
    print("(approx.) det. pixels per projected voxel:", (dq / dq_data)**2)
     
    t = 0
    for s in range(args.sample_states):
        for d in range(len(args.data)):
            # initialise I with random numbers
            I = np.random.random((args.mpx, args.mpx, args.mpx)) + 0.1
              
            # output args.output-t.pickle
            if T == 1 :
                fnam = args.output
            else :
                fnam      = args.output.split('.')
                fnam[-2] += f'-{t}'
                fnam      = '.'.join(fnam)
            
            pickle.dump({'qmax': qmax_max, 'qmin': qmin, 'dq': dq, 'I': I, 't': t, 'sample-state': s, 'data-set': args.data[d]}, open(fnam, 'wb'))
    
    
