# cache results using photon sparse format in pickle files
# all dimension in dataset except the first are ravelled (the pixel coordinates are flattened)
import h5py
import numpy as np
from tqdm import tqdm
import joblib
import os
import pickle

class Data_getter():
    """
    data_getter = Data_getter('data.cxi', 'entry_1/data_1/data', pixel_mask)

    data = data_getter[::2, :]
    data = data_getter[:, :100]

    not allowed:
    data = data_getter[2, :]
    data = data_getter[:]
    """
    def __init__(self, fnam, dataset, mask, cachedir = './cachedir'):
        self.fnam    = fnam
        self.dataset = dataset
        self.mask    = mask
         
        with h5py.File(self.fnam) as f:
            self.dtype = f[dataset].dtype
            self.shape = f[dataset].shape
        
        self.pixels = np.sum(self.mask)

        # hash pixel mask and filename
        self.hash = joblib.hash([fnam, mask])
        
        # store indices
        self.indices      = np.arange(self.shape[0])
        
        # flattened pixel indices
        self.indices_mask = np.arange(self.pixels)

        # create cachedir if needed
        if not os.path.exists(cachedir):
            os.mkdir(cachedir)
        self.cachedir = cachedir

    def load_h5(self, key, out = None):
        # loop over key[0] since we assume data is chunked along first dimension
        frames = self.indices[key[0]]
        pixels = self.indices_mask[key[1]]
        
        if out is None :
            out = np.empty((len(frames), len(pixels)), dtype = self.dtype)
        
        with h5py.File(self.fnam) as f:
            for i, d in tqdm(enumerate(frames), total = len(frames)):
                out[i, :len(pixels)] = f[self.dataset][d][self.mask][key[1]]
        return out

    def load_sparse(self, fnam):
        shape, inds, data = pickle.load(open(fnam, 'rb'))
        out = np.zeros(shape, self.dtype)
        out[inds] = data
        return out
        
    def save_sparse(self, fnam, array):
        inds = array > 0 
        pickle.dump((array.shape, inds, array[inds]), open(fnam, 'wb'))
    
    def __getitem__(self, key):
        return self.get(key)
    
    def get(self, key, cache = True):
        if cache :
            # generate file name
            h    = joblib.hash(key)
            fnam = f'{self.cachedir}/{self.hash}_{h}.pickle'
        
        if os.path.exists(fnam) and cache :
            out = self.load_sparse(fnam)
        else :
            print('\nloading from h5 file')
            out = self.load_h5(key)
            print('caching...\n')
            self.save_sparse(fnam, out)
        
        return out
