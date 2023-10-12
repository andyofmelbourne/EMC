import numpy as np
def make_xyz(corner, basis, shape):
    i, j = np.indices(shape[1:])
    xyz = corner[:, np.newaxis, np.newaxis, :] \
        + basis[:, 0, np.newaxis, np.newaxis, :] * i[np.newaxis,:,:,np.newaxis] \
        + basis[:, 1, np.newaxis, np.newaxis, :] * j[np.newaxis,:,:,np.newaxis]
    return np.transpose(xyz, axes=(3, 0, 1, 2))

def make_real_im(ar, corner, basis, dx, dy):
    xyz = make_xyz(corner, basis, ar.shape)
    ij = xyz[:2]
    ij[0] -= ij[0].min()
    ij[1] -= ij[1].min()
    ij[0] /= dx
    ij[1] /= dy
    ij = np.rint(ij).astype(int)
    im = np.zeros((ij[0].max()+1, ij[1].max()+1), dtype=ar.dtype)
    im[ij[0], ij[1]] = ar
    return im

def make_real_im_xyz(ar, xyz, dx, dy):
    ij = xyz[:2].copy()
    ij[0] -= ij[0].min()
    ij[1] -= ij[1].min()
    ij[0] /= dx
    ij[1] /= dy
    ij = np.rint(ij).astype(int)
    im = np.zeros((ij[0].max()+1, ij[1].max()+1), dtype=ar.dtype)
    im[ij[0], ij[1]] = ar
    return im

if __name__ == '__main__':
    # For example
    #############
    import h5py
    #f = h5py.File("amo06516.cxi", 'r')
    f = h5py.File("/home/andyofmelbourne/Documents/git_repos/EMC/data/data_amo87215.cxi", 'r')
    xyz    = f['/entry_1/instrument_1/detector_1/xyz_map'][()]
    corner = f['/entry_1/instrument_1/detector_1/corner_positions'][0]
    basis  = f['/entry_1/instrument_1/detector_1/basis_vectors'][0]
    dx = f['entry_1/instrument_1/detector_1/x_pixel_size'][()]
    dy = f['entry_1/instrument_1/detector_1/y_pixel_size'][()]
    
    #tags = f['/entry_1/instrument_1/detector_1/tags'][()]
    data = f['entry_1/data_1/data']


    ims = []

    I = 0
    for i in range(100):
        #if tags[i] :
        if True :
            print(i)
            image_2d = make_real_im_xyz(data[i], xyz, dx, dy)
            ims.append(image_2d.copy())
            I += 1

        if I > 100 :
            break

    import pyqtgraph as pg
    pg.show(np.array(ims))
