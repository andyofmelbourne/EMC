import h5py
import numpy as np
from tqdm import tqdm

fnam = 'data.cxi'
with h5py.File(fnam) as f:
    data = f['entry_1/data_1/data']

    with h5py.File('data_T.h5', 'w') as g:
        data_T = g.create_dataset('data_id', shape = (np.prod(data.shape[1:]), data.shape[0]), dtype=np.uint16)
        
        for d in tqdm(range(0, data.shape[0], 1024)):
            dmax = min(d+1024, data.shape[0])
            
            data_T[:,d:dmax] = data[d:dmax].reshape(dmax-d, -1).T
