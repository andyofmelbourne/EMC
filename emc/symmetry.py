import numpy as np


def octahedral_symmetry(ar):
    # must be cube 
    assert(np.allclose(*ar.shape))
    
    # must be 3D
    assert(len(ar.shape) == 3)
    
    N  = ar.shape[0]
    
    # assume fftshifted 
    i0 = N//2
    
    if N % 2 == 0 :
        i = np.fft.fftshift(np.fft.fftfreq(N-1, 1/(N-1)).astype(np.uint16))
    else :
        i = np.fft.fftshift(np.fft.fftfreq(N, 1/N).astype(np.uint16))
    
    i, j, k = np.meshgrid(i, i, i, indexing='ij')

    out = ar.copy()

    # not optimised but shouldn't take long anyway
        
    # 90deg rotation about z-axis
    # 90deg rotation: I(x, y, z) = I(y, -x, z)
    #                 I[i+i0, j+j0, k+k0] = I[j+i0, -i+j0, k+k0]
    out[i+i0, j+i0, k+i0] += out[j+i0, -i+i0, k+i0]
    
    # 180 deg rotation about z-axis
    # I(x, y, z) = I(-x, -y, z)
    # I[i+i0, j+j0, k+k0] = I[-i+i0, -j+j0, k+k0]
    out[i+i0, j+i0, k+i0] += out[-i+i0, -j+i0, k+i0]
    
    # now we have filled out the z+ face of the cube (dual to ocathedra)
    out2 = out.copy()
    
    # 3 x 90deg rotation about x-axis
    # 90deg rotation: I(x, y, z) = I(x, z, -y)
    #                 I[i+i0, j+j0, k+k0] = I[i+i0, k+j0, -j+k0]
    for a in range(3):
        out[i+i0, j+i0, k+i0] = out[i+i0, k+i0, -j+i0]
        out2 += out
    
    # 90 deg about z-axis
    out[i+i0, j+i0, k+i0] = out[j+i0, -i+i0, k+i0]
    out2 += out

    # 180deg rotation about z-axis
    out[i+i0, j+i0, k+i0] = out[-i+i0, -j+i0, k+i0]
    out2 += out
    
    return out2

def inversion_symmetry(ar):
    # must be cube 
    assert(np.allclose(*ar.shape))
    
    # must be 3D
    assert(len(ar.shape) == 3)
    
    N  = ar.shape[0]
    
    # assume fftshifted 
    i0 = N//2
    
    if N % 2 == 0 :
        i = np.fft.fftshift(np.fft.fftfreq(N-1, 1/(N-1)).astype(np.uint16))
    else :
        i = np.fft.fftshift(np.fft.fftfreq(N, 1/N).astype(np.uint16))
    
    i, j, k = np.meshgrid(i, i, i, indexing='ij')

    out = ar.copy()

    out[i+i0, j+i0, k+i0] += out[-i+i0, -j+i0, -k+i0]
    return out

    
if __name__ == '__main__':
    import pyqtgraph as pg

    # paint on +z 
    a = np.zeros((16,16,16), dtype = int)
    a[8,2,-1] = 2
    a[7,2,-1] = 1

    b = octahedral_symmetry(a)
    c = inversion_symmetry(b)

    # show cube faces
    #pg.show(np.array([b[1, :, :], b[-1, :, :], b[:, 1, :], b[:, -1, :], b[:, :, 1], b[:, :, -1]]))
    pg.show(np.array([c[1, :, :], c[-1, :, :], c[:, 1, :], c[:, -1, :], c[:, :, 1], c[:, :, -1]]))



