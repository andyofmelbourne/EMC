import argparse

if __name__ == '__main__':
    description = \
    """
    Calculate probability values for EMC by normalising logR values over rotations and possibly sample states or datasets.
    """
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--beta', type=float, default=0.001, \
                        help="beta parameter for probabilities: P <-- P^beta.")
    parser.add_argument('-P', '--P_file', type=str, default='probability-matrix-merged_intensity.h5', \
                        help="probability matrix h5 file contaning logR values to normalise. For multiple files use coma separated list (no spaces)")
    args = parser.parse_args()
    args.P_file = '.'.join(args.P_file.split(','))
    
import h5py
import numpy as np
from tqdm import tqdm
import pickle
import math

def print_change_in_most_likely_orientations(fnam):
    Ls = []
    f = open(fnam, 'rb')
    while True :
        try :
            Ls.append(pickle.load(f))
        except EOFError :
            break
    
    if len(Ls) < 2 :
        return
    
    for i in range(1, len(Ls)):
        if len(Ls[i]) == len(Ls[i-1]) :
            changed = np.sum( (Ls[i] - Ls[i-1]) > 0 )
            per_change = 100 * changed / len(Ls[i])
            print('iteration {}: {:2.2f}% of frames have changed most likely orientation'.format(i, per_change))

def check_sparsity(fnam):
    with h5py.File(fnam) as f:
        P = f['probability_matrix'][()]
         
        threshold = 1e-2 * np.max(P, axis=0)
        
        no_per_rot = np.sum(P > threshold[None, :], axis=0)

        print('average number of frames with prob. more than 1% of max per orientation : {}'.format(int(round(np.mean(no_per_rot)))))
        print('percentage     of frames with prob. more than 1% of max per orientation : {:.2f}%'.format(100*np.mean(no_per_rot)/P.shape[0]))



# for now: assume 2xlogR fit's on cpu ram

if __name__ == '__main__':
    print('\n\n')
    with h5py.File(args.P_file, 'a') as f:
        D, Mrot = f['logR'].shape
        dtype = f['logR'].dtype
        
        logR   = np.empty((D, Mrot), dtype = dtype)
        logR_d = np.empty((Mrot,), dtype = dtype)
        #P      = np.empty((D, Mrot), dtype = dtype)
        
        # save most likely orientations
        most_likely = np.empty((D,), dtype = int)
        
        # calculate mutual information = < sum_r P_dr log(P_dr) >_d
        #mi = np.empty((D,))
        
        # calculate log likelihood = sum_dr P logR 
        LL = np.empty((D,))
         
        # read direct is faster and uses less memory
        #logR = f['logR'][()]
        for i in tqdm(range(1), desc=f'reading logR from {args.P_file}'):
            f['logR'].read_direct(logR, np.s_[:], np.s_[:])
            wscale = np.log(f['tomogram_scales'][()])
            ksums  = f['photon_sums'][()] 
        
        for d in tqdm(range(D), desc='normalising probabilities: logR -> P'):
            logR_d[:] = logR[d]
            Lmax_r = np.argmax(logR_d)
            Lmax   = logR_d[Lmax_r]
                
            logR[d]  = np.exp( args.beta * (logR_d - Lmax)) 
            logR[d]  = logR[d] / np.sum(logR[d])
            
            most_likely[d] = Lmax_r
            
            #mi[d]          = np.sum(P * np.log(P))
            logR_d += ksums[d] * wscale
            LL[d]   = np.sum(logR[d] * logR_d)
    
        for i in tqdm(range(1), desc=f'writing probabilities to {args.P_file}'):
            #f['probability_matrix'][...] = P
            f['probability_matrix'].write_direct(logR, np.s_[:], np.s_[:])
        
        # now write transpose for faster reading in update_I
        for i in tqdm(range(1), desc=f'writing transposed probabilities to {args.P_file}'):
            if 'probability_matrix_rd' not in f :
                f.create_dataset('probability_matrix_rd', 
                    shape = (Mrot, D), 
                    dtype=float)
            
            rc = 2**12
            Rrot = math.ceil(Mrot/rc)
            PT = np.empty((rc, D), dtype=float)
            for r in tqdm(range(Rrot), desc='writing'):
                rstart = r * rc
                rstop  = min(rstart + rc, Mrot)
                dr     = rstop - rstart
                
                PT[:dr] = logR[:, rstart:rstop].T
                f['probability_matrix_rd'].write_direct(PT, np.s_[:dr, :], np.s_[rstart:rstop, :])
    
    # output most likely orientation for analysis
    pickle.dump(most_likely, open('most_likely_orientations.pickle', 'ab'))
    
    # print % of patterns that have changed orientation
    print_change_in_most_likely_orientations('most_likely_orientations.pickle')
    
    # print mututal information
    #print('mutual information: {:.2e}'.format(np.mean(mi)))
    
    # print log likelihood 
    print('Log likelihood per pattern per rotation: {:.2e}'.format(np.mean(LL)/Mrot))
    
    #check_sparsity(args.P_file)
