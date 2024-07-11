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
    parser.add_argument('-s', '--sample_smoothing', type=float, default=0, \
                        help="make orientation probabilities closer to average value over sample states, help with alignment")
    args = parser.parse_args()
    args.P_file = args.P_file.split(',')
    
import h5py
import numpy as np
from tqdm import tqdm
import pickle
import math

def print_change_in_most_likely_orientations(fnam, txt = 'orientations'):
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
            print('iteration {}: {:2.2f}% of frames have changed most likely {}'.format(i, per_change, txt))

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
    
    # get a reference to all P-files
    files = [h5py.File(fnam, 'a') for fnam in args.P_file]
    
    # check shapes
    ##############
    print('checking logR shapes')
    Mrots  = [f['logR'].shape[1] for f in files]
    Ds     = [f['logR'].shape[0] for f in files]
    if len(set(Mrots)) == 1 and len(set(Ds)) == 1:
        print('yes')
    else :
        raise ValueError('No, logR matrices are not consistent!', Mrots, Ds)
    Mrot  = Mrots[0]
    D     = Ds[0]
    dtype = files[0]['logR'].dtype


    # check tomogram scale factors
    ##############
    tomo_scales  = [f['tomogram_scale'][()] for f in files]
    print('checking tomogram scale factors')
    if len(set(tomo_scales))==1:
        print('yes')
    else :
        raise ValueError('No, scales are not consistent!', tomo_scales)
    tomo_scale = tomo_scales[0]
    log_tomo_scale = np.log(tomo_scale)
    
    sample_states = len(files)

    T = sample_states * Mrot
    
    logR   = np.empty((sample_states, Mrot), dtype = float)
    logR_d = np.empty((sample_states, Mrot), dtype = float)
    P      = np.empty((sample_states, Mrot), dtype = float)
    Pmean  = np.empty((Mrot,), dtype = float)
    ksums  = np.empty((D,), dtype = float)
        
    # save most likely orientations
    most_likely = np.empty((D,), dtype = int)
    
    # save most likely sample state
    most_likely_state = np.empty((D,), dtype = int)
        
    # calculate mutual information = < sum_r P_dr log(P_dr) >_d
    #mi = np.empty((D,))
        
    # calculate log likelihood = sum_dr P logR 
    LL = np.empty((D,))

    # same for all files
    for i in tqdm(range(1), desc=f'reading photon sums from {args.P_file[0]}'):
        ksums[:]  = files[0]['photon_sums'][()] 
    
    # this could be very slow without chunking
    for d in tqdm(range(D), desc='normalising probabilities: logR -> P'):
        # read logR's
        for t, f in enumerate(files) :
            f['logR'].read_direct(logR, source_sel = np.s_[d, :], dest_sel = np.s_[t, :])
        
        # find maximum likelihood and most_likely orientation / sample state
        ml = np.argmax(logR) 
        most_likely[d]       = ml % Mrot 
        most_likely_state[d] = ml // Mrot 
        Lmax = logR[most_likely_state[d], most_likely[d]]
        
        # calculate probability 
        #######################
        # P = e^( beta * logR ) / sum_r e^( beta * logR )
        #   P[r]  = e^( beta * (logR[r] - logR_max))  
        #   P[r] /= sum_r P[r]
        P[:] = np.exp( args.beta * (logR - Lmax)) 
        P   /= np.sum(P)

        if args.sample_smoothing != 0. :
            Pmean[:] = np.mean(P, axis = 0)
            for t in range(P.shape[0]):
                P[t] = (1-args.sample_smoothing) * P[t] + args.sample_smoothing * Pmean 
            P /= np.sum(P)
        
        # calculate log likelihood per pattern
        # LL[d] = sum_r P[d, r] logR[d, r]
        # 
        # we have calculated logR with scaling
        #   logR0[d, r] = sum_i K[d, i] log(wscale * w[r, i])
        #               = sum_i K[d, i] log(w[r, i]) + log(wscale) sum_i K[d, i] 
        #               = logR[d, r] + log(wscale) ksum[d]
        #   LL0[d] = LL[d] + log(wscale) ksum[d] sum_r P[d, r]
        #          = LL[d] + log(wscale) ksum[d] 
        # LL[d] = LL0[d] - log(wscale) ksum[d]
        LL[d] = np.sum(P * logR) - log_tomo_scale * ksums[d]

        # write P's
        for t, f in enumerate(files) :
            f['probability_matrix'].write_direct(P, source_sel = np.s_[t, :], dest_sel = np.s_[d, :])
         
    # output most likely orientation for analysis
    pickle.dump(most_likely, open('most_likely_orientations.pickle', 'ab'))
    
    # print % of patterns that have changed orientation
    print_change_in_most_likely_orientations('most_likely_orientations.pickle', 'orientation')

    if sample_states > 1 :
        pickle.dump(most_likely_state, open('most_likely_sample_state.pickle', 'ab'))
        print_change_in_most_likely_orientations('most_likely_sample_state.pickle', 'sample state')
    
    # print mututal information
    #print('mutual information: {:.2e}'.format(np.mean(mi)))
    
    # print log likelihood 
    print('Log likelihood per pattern: {:.2e}'.format(np.mean(LL)))
    
    #check_sparsity(args.P_file)
