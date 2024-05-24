# cache results using photon sparse format in pickle files
# all dimension in dataset except the first are ravelled (the pixel coordinates are flattened)
import h5py
import numpy as np
from tqdm import tqdm
import joblib
import os
import pickle


class Data_getter_0():
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

class Data_getter_1():
    """
    cache to RAM instead of disk
    
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
        self.cache   = {}
         
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
            for i, d in tqdm(enumerate(frames), total = len(frames), disable = True):
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
        
        if fnam not in self.cache and os.path.exists(fnam) and cache :
            out = self.load_sparse(fnam)
            
            # save to RAM
            self.cache[fnam] = out
        
        elif fnam in self.cache and cache :
            out = self.cache[fnam]
            
        else :
            #print('\nloading from h5 file')
            out = self.load_h5(key)
            
            # save to RAM
            self.cache[fnam] = out
            
            # save to disk
            #print('caching...\n')
            self.save_sparse(fnam, out)
        
        return out


class Data_getter_2():
    """Save full dataset in sparse format
    
    Load full dataset in memory and keep it there"""
    def __init__(self, fnam, dataset, cachedir = './cachedir'):
        self.fnam    = fnam
        self.dataset = dataset
        self.cache   = {}
        self.sparse_fnam = f'cachedir/{fnam}-sparse.h5'
         
        with h5py.File(self.fnam) as f:
            self.dtype = f[dataset].dtype
            self.shape = f[dataset].shape
        
        # store indices
        self.indices      = np.arange(self.shape[0])
        
        # create cachedir if needed
        if not os.path.exists(cachedir):
            os.mkdir(cachedir)
        self.cachedir = cachedir
        
        # check if sparse file exists
        self.sparse_file = os.path.exists(self.sparse_fnam) 
        self.loaded      = False
        
        if not self.sparse_file :   
            self.save_sparse() 
        
        if not self.loaded :
            self.load_sparse()

        # index frames 
        self.frame_indices = np.concatenate(([0], np.cumsum(self.litpix)))
    
    def save_sparse(self):
        inds    = []
        photons = []
        litpix  = []
        with h5py.File(self.fnam, 'r') as f:
            for d in tqdm(range(self.shape[0]), desc = 'extracting data into sparse format'):
                frame   = f[self.dataset][d].ravel() 
                inds.append(np.where(frame > 0)[0])
                photons.append(frame[inds[-1]].copy())
                litpix.append(len(inds[-1]))

        self.photons = np.concatenate(photons)
        self.litpix  = np.array(litpix)
        self.inds    = np.concatenate(inds)
            
        for _ in tqdm(range(1), desc = 'saving data in sparse format'):
            with h5py.File(self.sparse_fnam, 'w') as out:
                out['photons'] = self.photons
                out['litpix']  = self.litpix
                out['inds']    = self.inds
        
        self.sparse_file = True
        self.loaded      = False
    
    def load_sparse(self):
        for _ in tqdm(range(1), desc = 'loading sparse photons from file'):
            with h5py.File(self.sparse_fnam, 'r') as f:
                self.photons = f['photons'][()]
                self.litpix  = f['litpix'][()]
                self.inds    = f['inds'][()]
        
        self.loaded = True
                
    def __getitem__(self, key):
        frames = np.arange(len(self.litpix))[key[0]]
        pixels = np.arange(np.prod(self.shape[1:]))[key[1]]
        
        out   = np.zeros((len(frames), len(pixels)), dtype = self.dtype)
        frame = np.zeros( (np.prod(self.shape[1:]),), dtype = self.dtype)
        for i, d in enumerate(frames) :
            frame.fill(0)
            j0 = self.frame_indices[d]
            j1 = self.frame_indices[d + 1]
            frame[self.inds[j0: j1]] = self.photons[j0: j1]
            out[i] = frame[pixels]
            
        return out
        
            
    

Data_getter = Data_getter_2
