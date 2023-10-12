# look at a tomogram  
# and compare with most likely data frames


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
    parser.add_argument('-I', '--merged_intensity', type=str, default='merged_intensity.pickle', \
                        help="filename of the python pickle file containing merged intensities.")
    parser.add_argument('-d', '--data', type=str, default='data.cxi', \
                        help="cxi file containing data frames")
    parser.add_argument('-T', '--data_T', type=str, default='data_T.h5', \
                        help="h5 file containing transposed data frames.")
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


import numpy as np
import h5py
import pickle

import logR
import look_at_data

from emc.tomograms import * 


if __name__ == '__main__':
    print('\n\n')
    with h5py.File(args.data) as f:
        xyz    = f['/entry_1/instrument_1/detector_1/xyz_map'][()]
        dx = f['entry_1/instrument_1/detector_1/x_pixel_size'][()]
        dy = f['entry_1/instrument_1/detector_1/y_pixel_size'][()]

    most = pickle.load(open('most_likely_orientations.pickle', 'rb'))
    
    # look at most populated orientations
    counts = np.bincount(most)
    #ds = np.where(most == counts.max())[0]
    
    ds = np.where(most == np.argmax(counts))[0]
    rs = most[ds]


    # get merged intensity
    d  = pickle.load(open(args.merged_intensity, 'rb'))
    I  = d['I'].copy()
    dq = d['dq']
    
    # get rotations from file or recalculate
    # --------------------------------------
    if args.rotations == None :
        args.rotations = I.shape[0]//2
    
    R, rot_order = logR.get_rotations(args.rotations)
    
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
    
    qmask, q, C, qmin, qmax = logR.get_qmask(qmin, qmax, args.dataname)
    
    tomos = calculate_tomograms(I, C, q, qmask, R, dq, rs)
    
    # now show tomograms in geometry corrected view
    ims = []
    W = np.zeros((4,512,512))
    for i, tomo in enumerate(tomos):
        W[qmask] = tomo
        imw = look_at_data.make_real_im_xyz(W, xyz, dx, dy)
        
        # compare with data
        with h5py.File(args.data) as f:
            k = f['entry_1/data_1/data'][ds[i]].astype(float)
        
        # scale to tomo
        k *= np.sum(tomo) / np.sum(k[qmask])
        imk = look_at_data.make_real_im_xyz(k, xyz, dx, dy)
        
        
        # blend images
        image = np.zeros(imw.shape + (3,))
        image[:, :, 2] = imw
        image[:, :, 1] = imw
        m = imw.max()
        image[:, :, 0] = imw + imk * (m - imw) / m
        ims.append(image.copy())
        
        #ims.append(imw.copy())
        #ims.append(imk.copy())

        # calculate sum_i log R = sum_i k log(w)
        print(i, 2*i, np.sum( k[qmask] * np.log(tomo / np.sum(tomo)) ) / np.sum(k[qmask]))
