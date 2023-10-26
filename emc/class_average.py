# generate 2D cryoEM style class averages

# should I try and hack the 3D stuff to work on this?
# probably just adds complexity, but also means lots of duplication

# initialise Mtomo 2D models
# calculate photon sums 
# calculate tomogram sums for each in-plane rotation and for each 2D model
# calculate probability matrix - one for each model or just one for all models?  one for each model
    # Mrot = no. of 2D models x no. of in plane rotations

import numpy as np
import h5py
import scipy.constants as sc
import math
import pickle

import pyclblast

from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


#fnam_data = "/home/andyofmelbourne/Documents/2023/continuity_constraint/EMC/amo06516.cxi"
fnam_data_T = "/home/andyofmelbourne/Documents/git_repos/EMC/data/data_T.h5"
fnam_data   = "/home/andyofmelbourne/Documents/git_repos/EMC/data/data.cxi"
fnam_models = "emc_2d_models.h5"

# merge in tomo or I space
merge_I = True

# number of 2D models or "classes"
N   = 10

# number of in plane rotations
M_in_plane = 100

# number of linear pixels in each model
mpx = 128

# min / max radial values for merge (metres)
rmin_merge = 0
rmax_merge = 200 * 75e-6

# min / max radial pixel values for calculating probabilities
rmin_prob = 10 * 75e-6
rmax_prob = 180 * 75e-6

# beta parameter
beta = 0.01

# location of zero pixel in merge
i0 = np.float32(mpx//2)

# skip init
skip_init = True
skip_prob = False

with h5py.File(fnam_data) as f:
    Ndata = f['entry_1/data_1/data'].shape[0]


import pyopencl as cl
import pyopencl.array 
from tqdm import tqdm

gpu_precision = np.float32

# find an opencl device (preferably a GPU and preferably NVIDIA) in one of the available platforms
for p in cl.get_platforms():
    if 'NVIDIA' in p.name :
        break
    
if len(devices) == 0 :
    for p in cl.get_platforms():
        devices = p.get_devices()
        if len(devices) > 0:
            break

context = cl.Context(devices)
queue   = cl.CommandQueue(context)


# find an opencl device (preferably a CPU) in one of the available platforms
for p in cl.get_platforms():
    devices_cpu = p.get_devices(cl.device_type.CPU)
    if len(devices_cpu) > 0:
        break
    
if len(devices_cpu) == 0 :
    devices_cpu = devices

context_cpu = cl.Context(devices_cpu)
queue_cpu   = cl.CommandQueue(context_cpu)



cl_code = cl.Program(context, r"""
    
    constant sampler_t trilinear = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR ;

    float2 _calculate_I_coord (
    float rx, 
    float ry,
    float *R, 
    const float i0, 
    const float dr)
    {
    float2 coord ;
    
    coord.x = i0 + (R[0] * rx + R[1] * ry) / dr + 0.5;
    coord.y = i0 + (R[2] * rx + R[3] * ry) / dr + 0.5;
    
    return coord;
    }

    // one worker per tomogram index
    float _calculate_tomogram (
    image2d_t I, 
    float C,  
    float rx, 
    float ry,
    float *R, 
    const float i0, 
    const float dr)
    {
    float2 coord = _calculate_I_coord(rx,ry,R,i0,dr);
    
    float4 v = read_imagef(I, trilinear, coord);

    return v.x * C;
    }

    __kernel void calculate_tomogram_batch (
    image2d_t I, 
    global float *Cg,  
    global float *rxg, 
    global float *ryg,
    global float *Rg, 
    global float *out, 
    const float i0, 
    const float dr, 
    const int Npix,
    const int rmin, 
    const int rmax)
    {
    int n = get_global_id(0);
    
    float C = Cg[n];  
    float rx = rxg[n];
    float ry = ryg[n];

    float R[4];
    
    for (int r=rmin; r<rmax; r++){
        for (int i=0; i<4; i++) {
            R[i] = Rg[4*r + i];
        }

        
        out[Npix * (r-rmin) + n] = _calculate_tomogram(I, C,  rx, ry, R, i0, dr);
        
    }
    }

    // one worker per pixel index
    __kernel void calculate_tomogram_w_scale_log_batch_pix (
    image2d_t I, 
    global float *Cg,  
    global float *rxg, 
    global float *ryg,
    global float *Rg, 
    global float *out, 
    const float i0, 
    const float dr, 
    global float *wscale, 
    const int Npix,
    const int rmin,
    const int rmax,
    const int ioff)
    {
    int n = get_global_id(0);
    
    float C  = Cg[ioff + n];  
    float rx = rxg[ioff + n];
    float ry = ryg[ioff + n];
    
    float R[4];
    
    float t;

    //float t1, t2;
    
    for (int r=rmin; r<rmax; r++){
        for (int i=0; i<9; i++) {
            R[i] = Rg[4*r + i];
        }
        
        t = wscale[r] * _calculate_tomogram(I, C,  rx, ry, R, i0, dr);

        //t1 = wscale[r];
        //t2 = _calculate_tomogram(I, C,  rx, ry, R, i0, dr);
        //printf("%f %f %f %f\n", t1, t2, t, log(t));
        
        if (t > 0.) 
            out[r*Npix + n] = log(t);
        else 
            out[r*Npix + n] = 0.;
    }
    }


    __kernel void calculate_W_to_I_mapping_w_ioffset(
    global int *out,
    global float *Rg, 
    global float *rxg, 
    global float *ryg,
    const float dr, 
    const float i0, 
    const int Npix,
    const int M,
    const int rstart,
    const int rstop,
    const int ioff)
    {

    int n = get_global_id(0);

    float rx = rxg[ioff + n];
    float ry = ryg[ioff + n];

    float2 coord;
    
    for (int r=rstart; r<rstop; r++){
        float R[4];

        for (int i=0; i<4; i++) {
            R[i] = Rg[4*r + i];
        }
        
        coord = _calculate_I_coord(rx,ry,R,i0,dr);
         
        // get flattened I index
        out[Npix * (r-rstart) + n] = convert_int_rte(coord.x) * M + convert_int_rte(coord.y);

    }
    }

    __kernel void scale_tomograms_for_merge_w_coffset (
    global float *W,  
    global float *C, 
    const int Npix, 
    const int rmin, 
    const int rmax, 
    const int coff)
    {
    int n = get_global_id(0);
    
    for (int r=rmin; r<rmax; r++)
        W[r*Npix + n] /= C[coff + n];
    
    }


""").build()


cl_code_cpu = cl.Program(context_cpu, r"""
    __kernel void merge_tomos_I ( 
        global double *I, global double *O, global double *W, global double *PK_on_W_r,
        global int *Ipix, 
        const int rmin, const int rmax, 
        const int Npix, const int M, const int imax, const int model)
    {
    int c = get_global_id(0);
    int C = get_global_size(0);
    
    int i, r, n, d;
    
    int offset = M * M * model ;
    
    double PK ;
    
    // loop over tomograms (rotations) and then pixels (data space)
    for (r = rmin + c; r < rmax; r+=C) {
        PK = PK_on_W_r[r];
        
        // index offset for tomogram r
        d = Npix * r;
        for (n = 0; n < imax; n++) {
            i = Ipix[d + n];
            
            // add photons to I  
            I[offset + i] += W[d + n] ;
            
            // add counter to O  
            O[offset + i] += PK;
        }
    }
    }

    __kernel void merge_tomos ( 
        global double *I, global double *O, global double *W, global double *PK_on_W_r,
        global int *Ipix, 
        const int rmin, const int rmax, 
        const int Npix, const int M, const int imax, const int model)
    {
    int c = get_global_id(0);
    int C = get_global_size(0);
    
    int i, r, n, d;
    
    int offset = M * M * model ;

    double PK;
    
    // loop over tomograms (rotations) and then pixels (data space)
    for (r = rmin + c; r < rmax; r+=C) {
        PK = PK_on_W_r[r];
        
        // index offset for tomogram r
        d = Npix * r;
        for (n = 0; n < imax; n++) {
            i = Ipix[d + n];
            
            // add photons to I  
            I[offset + i] += W[d + n] / PK;
            
            // add counter to O  
            O[offset + i] += 1;
        }
    }
    }
""").build()


class Merge_tomos():

    def __init__(self, N, Wshape, Ishape):
        self.N     = np.int32(N)
        self.M     = np.int32(Ishape[0])
        self.Npix  = np.int32(np.prod(Wshape[1:]))
        
        self.cu = np.int32(1)
        
        # one intensity and overlap for each compute unit
        self.Is_cl = cl.array.zeros(queue_cpu, (N,) + Ishape, dtype=np.float64)
        self.Os_cl = cl.array.zeros(queue_cpu, (N,) + Ishape, dtype=np.float64)
    
    def merge(self, n, W, Ipix, rmin, rmax, PK_on_W_r, imax = None, merge_I = False, is_blocking=True):
        """
        Ipix and W: (rot chunks, pix chunks)
        """
        if imax == None :
            imax = np.int32(self.Npix)
        else :
            imax = np.int32(imax)
        
        if merge_I :
            cl_merge = cl_code_cpu.merge_tomos_I
        else :
            cl_merge = cl_code_cpu.merge_tomos
        
        rmin = np.int32(rmin)
        rmax = np.int32(rmax)
        cl_merge(queue_cpu, (1,), (1,), 
            self.Is_cl.data, self.Os_cl.data, cl.SVM(W), cl.SVM(PK_on_W_r), 
            cl.SVM(Ipix), rmin, rmax, self.Npix, self.M, imax, np.int32(n))
        
        if is_blocking :
            queue_cpu.finish()
             
    def get_I_O(self):
        queue_cpu.finish()
        
        return self.Is_cl.get(), self.Os_cl.get()
        






# assume this will fit in memory
def init_models(N, Ndata, M_in_plane, mpx):
    # split over ranks
    rs = np.linspace(0, N, size+1).astype(int)
    
    models = np.random.random((rs[rank+1]-rs[rank], mpx, mpx))
    
    if rank == 0 :
        with h5py.File(fnam_models, 'a') as f:
            f.create_dataset('models', shape = (N, mpx, mpx), dtype = np.float64)
            f.create_dataset('logR',   shape = (N, Ndata, M_in_plane), dtype = np.float64)
            f.create_dataset('probability_matrix',   shape = (N, Ndata, M_in_plane), dtype = np.float64)
    
    comm.barrier()
        
    # write to file sequentially
    for r in range(size):
        if rank == r:
            # split over models 
            with h5py.File(fnam_models, 'a') as f:
                f['models'][rs[rank]:rs[rank+1]] = models
        
        comm.barrier()

def make_rmask(rmin, rmax, fnam_data):
    """
    basically replace q with xyz 
    """
    # pixel map
    with h5py.File(fnam_data, 'r') as f:
        xyz  = f['/entry_1/instrument_1/detector_1/xyz_map'][()]
        mask  = f['/entry_1/instrument_1/detector_1/mask'][()]
    
    # radius in metres
    r    = np.sum(xyz[:2]**2, axis=0)**0.5 
    
    r_corner = r[mask].max()
    r_edge   = max(np.abs(xyz[0][mask]).max(), np.abs(xyz[1][mask]).max())
    
    if rmax == -1 :
        rmax = r_corner
    
    elif rmax == -2 :
        rmax = r_edge
    
    rmask = (r >= rmin) * (r <= rmax)
    
    # apply detector mask
    rmask[~mask] = False
    
    return xyz[:2], rmask
    

def make_in_plane_rotations(M_in_plane):
    # theta[r] = 2 pi r / M_in_plane
    # R[r]     = [cos -sin]
    #            |sin  cos|

    t = 2 * np.pi * np.arange(M_in_plane) / M_in_plane
    R = np.empty((M_in_plane, 2, 2), dtype = np.float32)
    R[:, 0, 0] =  np.cos(t)
    R[:, 0, 1] = -np.sin(t)
    R[:, 1, 0] =  np.sin(t)
    R[:, 1, 1] =  np.cos(t)
    return R

def solid_angle_polarisation_factors(fnam_data, polarisation='x'):
    with h5py.File(fnam_data, 'r') as f:
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
        
    if polarisation == 'x' :
        P = 1 - (xyz[0] / r)**2
    elif polarisation == 'y' :
        P = 1 - (xyz[1] / r)**2
    elif polarisation == 'None' :
        P = np.ones(dshape[1:])
    
    # solid angle correction  
    Omega = dx * dy * xyz[2] / r**3
    
    # merged intensity to frame correction factor
    C = Omega * P
    
    # scale 
    C /= C[mask].max()
    
    return C

def calculate_tomogram_sums(I, C, r, rmask, R, dr, rc=256):
    Mrot = R.shape[0]
    
    # number of pixels within rmask
    Npix  = np.int32(np.sum(rmask))
    # pixel coordinate of r = 0 in I
    i0 = np.float32(I.shape[0]//2)
    
    R_cl      = cl.array.empty(queue, (4*Mrot,), dtype = np.float32)
    W_cl      = cl.array.empty(queue, (rc, Npix,), dtype = np.float32)
    wscale_cl = cl.array.empty(queue, (Mrot,), dtype = np.float32)
    rx_cl     = cl.array.empty(queue, (Npix,), dtype = np.float32)
    ry_cl     = cl.array.empty(queue, (Npix,), dtype = np.float32)
    C_cl      = cl.array.empty(queue, (Npix,), dtype = np.float32)
    
    cl.enqueue_copy(queue, R_cl.data, R.astype(np.float32))
    cl.enqueue_copy(queue, rx_cl.data, np.ascontiguousarray(r[0][rmask].astype(np.float32)))
    cl.enqueue_copy(queue, ry_cl.data, np.ascontiguousarray(r[1][rmask].astype(np.float32)))
    cl.enqueue_copy(queue, C_cl.data , np.ascontiguousarray(C[rmask].astype(np.float32)))
    
    # copy I as an opencl "image" for trilinear sampling
    I_cl = cl.Image(context, cl.mem_flags.READ_ONLY, cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT), shape = I.shape[::-1])
    cl.enqueue_copy(queue, I_cl, I.T.copy().astype(np.float32), is_blocking=True, origin=(0, 0), region=I.shape[::-1])
    
    wsums_out = np.empty((Mrot,))
    W         = np.empty((rc, Npix,), dtype = np.float32)
    
    # calculate tomograms then sum then output
    #for rmin in tqdm(range(0, Mrot, rc), desc='calculating tomogram sums'):
    for rmin in range(0, Mrot, rc):
        rmax = min(rmin + rc, Mrot)
        cl_code.calculate_tomogram_batch(queue, (Npix,), None,
                I_cl, C_cl.data, rx_cl.data, ry_cl.data, 
                R_cl.data, W_cl.data, i0, np.float32(dr), Npix, np.int32(rmin), np.int32(rmax))
        
        cl.enqueue_copy(queue, W, W_cl.data)
        assert(np.all(np.isfinite(W)))
        
        wsums_out[rmin:rmax] = np.sum(W, axis=1)[:rmax-rmin]
    return wsums_out
    
def calculate_tomogram_sums_MPI(fnam_models, C, r, rmask_prob, R, dr, rc = M_in_plane):
    M_in_plane = R.shape[0]
    
    with h5py.File(fnam_models, 'r') as f:
        N = f['models'].shape[0]
    
    # split over ranks
    ns = np.linspace(0, N, size+1).astype(int)
    
    nmin = ns[rank]
    nmax = ns[rank+1]
    wsums = np.empty((nmax - nmin, M_in_plane), dtype=np.float32)
    
    wsums_all = np.empty((N, M_in_plane), dtype=np.float32)
    
    for n in tqdm(range(nmin, nmax, 1), desc='calculating tomogram sums'):
        with h5py.File(fnam_models, 'r') as f:
            I = f['models'][n]
        
        wsums[n-nmin] = calculate_tomogram_sums(I, C, r, rmask_prob, R, dr, rc)
        assert(np.all(np.isfinite(wsums[n-nmin])))
    
    # distribute
    for r in range(size):
        wsums_all[ns[r]:ns[r+1]] = comm.bcast(wsums, root=r)
    return wsums_all

def get_photon_sums_in_mask(fnam_cxi, qmin, qmax, qmask):
    dataname = Path(fnam_cxi).stem
    
    qmaxint = int(qmax)
    qminint = int(qmin)
    fnam = f'photons-sums-{dataname}-{qminint}-{qmaxint}.pickle'
    if os.path.exists(fnam) :
        ksums = pickle.load(open(fnam, 'rb'))
    else :
        # output unmasked non-zero pixels and indices to pickle file
        ksums = []
        with h5py.File(fnam_cxi, 'r') as f:
            data = f['entry_1/data_1/data']
            shape = data.shape
            
            for d in tqdm.tqdm(range(0, shape[0]), desc='calculating photon sums in qmask (writing to pickle file)'):
                ksums.append(np.sum(data[d][qmask]))
        
        pickle.dump(np.array(ksums), open(fnam, 'wb'))
    return ksums

    
# initialise 2D models with random numbers
if not skip_init :
    init_models(N, Ndata, M_in_plane, mpx)

# calculate rmask
if rank == 0:
    r, rmask_merge = make_rmask(rmin_merge, rmax_merge, fnam_data)
    r, rmask_prob  = make_rmask(rmin_prob,  rmax_prob,  fnam_data)
else :
    rmask_merge = None
    rmask_prob  = None
    r           = None

rmask_merge = comm.bcast(rmask_merge, root=0)
rmask_prob  = comm.bcast(rmask_prob,  root=0)
r           = comm.bcast(r,           root=0)


# sampling
if (mpx % 2) == 0 :
    dr = rmax_merge / (mpx / 2 - 1)
else :
    dr = 2 * rmax_merge / (mpx - 1)


# calculate solid angle and polarisation factors
if rank == 0:
    C = solid_angle_polarisation_factors(fnam_data)
else :
    C = None
C = comm.bcast(C, root = 0)


# calculate photon sums within rmask_prob (use rank 1 if available)
rankK = min(1, size-1)
if rank == rankK :
    # Hack: scale r's to pixel units for labeling of pickle files  
    ksums = get_photon_sums_in_mask(fnam_data, rmin_prob / 75e-6, rmax_prob / 75e-6, rmask_prob)
else :
    ksums = None
    
ksums = comm.bcast(ksums, root = rankK)

# calculate in-plane rotations (use rank 2 if available)
rankR = min(2, size-1)
if rank == rankR :
    R = make_in_plane_rotations(M_in_plane)
else :
    R = None
R = comm.bcast(R, root = rankR)


# calculate tomogram sums within rmask_prob
wsums = calculate_tomogram_sums_MPI(fnam_models, C, r, rmask_prob, R, dr, rc = M_in_plane)

# gather (or allgather?) sums
#wsums = np.empty((N, M_in_plane), dtype=np.float32)

# calculate logR - one logR for each 2d model. Assume it fits in memory? 
#   cannot load data for each model (we would have to read the entire dataset N times)
#   I think calculating w_ri is cheap because it is 2D
#   splitting over pixels makes sense again
#   no split over data (why don't I always do this?)

# logR[d, r] = sum_(i in mask) K_di log(w_ri)
# w_ri = tomo_scale * W_ri / sum_i W_ri

"""
# split models over ranks (as in wsums)
ns = np.linspace(0, N, size+1).astype(int)
nmin = ns[rank]
nmax = ns[rank+1]

# loop over detector pixels
ic   = np.int32(2048) # number of pixels per chunk
Npix = np.int32(np.sum(rmask_prob))
U    = math.ceil(Npix/ic)



logR     = np.zeros((nmax-nmin, Ndata, M_in_plane), dtype = np.float64)
logR_buf = np.zeros((Ndata, M_in_plane), dtype = np.float32)
logR_cl  = cl.array.zeros(queue, shape = (Ndata, M_in_plane), dtype = np.float32)

K    = np.zeros((Ndata, ic), dtype = np.float32)
K_cl = cl.array.zeros(queue, shape = (ic, Npix), dtype = np.float32)

R_cl = cl.array.empty(queue, (4*M_in_plane,), dtype = np.float32)
cl.enqueue_copy(queue, R_cl.data, R.astype(np.float32))

W_cl      = cl.array.empty(queue, (M_in_plane, ic,), dtype = np.float32)

# wsums are split over ranks
wscale    = (Npix / wsums).astype(np.float32)
wscale_cl = cl.array.empty(queue, (M_in_plane,), dtype = np.float32)

rx_cl = cl.array.empty(queue, (Npix,), dtype = np.float32)
ry_cl = cl.array.empty(queue, (Npix,), dtype = np.float32)
cl.enqueue_copy(queue, rx_cl.data, np.ascontiguousarray(r[0][rmask_prob].astype(np.float32)))
cl.enqueue_copy(queue, ry_cl.data, np.ascontiguousarray(r[1][rmask_prob].astype(np.float32)))

C_cl = cl.array.empty(queue, (Npix,), dtype = np.float32)
cl.enqueue_copy(queue, C_cl.data , np.ascontiguousarray(C[rmask_prob].astype(np.float32)))



# copy I as an opencl "image" for trilinear sampling
I_cl = cl.Image(context, cl.mem_flags.READ_ONLY, cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT), shape = (mpx, mpx))

# calculate indices for pixel chunks
t = np.where(rmask_prob.ravel())[0]
inds_qmask = []

if rank == 0 :
    i_iter = tqdm(range(U))
    n_iter = tqdm(range(nmin, nmax, 1))
else :
    i_iter = range(U)
    n_iter = range(nmin, nmax, 1)

for i in i_iter:
    istart = i * ic
    istop  = min(istart + ic, Npix)
    di     = istop - istart
    inds_qmask.append(t[istart:istop])
    
    with h5py.File(fnam_data_T) as f:
        K[:, :di] = f['data_id'][inds_qmask[i], :].T
        K[:, di:] = 0
    cl.enqueue_copy(queue, K_cl.data, K)
    
    # loop over models
    for n in n_iter :
        with h5py.File(fnam_models, 'r') as f:
            I = f['models'][n]
        
        cl.enqueue_copy(queue, I_cl, I.T.copy().astype(np.float32), is_blocking=True, origin=(0, 0), region=(mpx, mpx))
        
        cl.enqueue_copy(queue, wscale_cl.data, wscale[n])
         
        # calculate all tomograms: log( (tomoscale / wscale)_r x W_ri )
        cl_code.calculate_tomogram_w_scale_log_batch_pix(queue, (di,), None, 
            I_cl, C_cl.data, rx_cl.data, ry_cl.data,  
            R_cl.data, W_cl.data, i0, np.float32(dr), 
            wscale_cl.data, ic, np.int32(0), np.int32(M_in_plane), np.int32(istart))
        
        # logR[d, r] = sum_(i in mask) K_di log(w_ri)
        pyclblast.gemm(queue, Ndata, M_in_plane, di, K_cl, W_cl, logR_cl, a_ld=ic, b_ld=ic, c_ld = M_in_plane, b_offset = 0, b_transp=True)
        
        # copy to cpu
        cl.enqueue_copy(queue, logR_buf, logR_cl.data)
        logR[n-nmin] += logR_buf

# write to file sequentially
for r in range(size):
    if rank == r:
        # split over models 
        with h5py.File(fnam_models, 'a') as f:
            f['logR'][nmin:nmax] = logR
    
    comm.barrier()
"""

# split data over ranks 
dc = 512

Npix = np.int32(np.sum(rmask_prob))

logR     = np.zeros((N, dc, M_in_plane), dtype = np.float64)
logR_buf = np.zeros((dc, M_in_plane), dtype = np.float32)
logR_cl  = cl.array.zeros(queue, shape = (dc, M_in_plane), dtype = np.float32)

K    = np.zeros((dc, Npix), dtype = np.float32)
K_cl = cl.array.zeros(queue, shape = (dc, Npix), dtype = np.float32)

R_cl = cl.array.empty(queue, (4*M_in_plane,), dtype = np.float32)
cl.enqueue_copy(queue, R_cl.data, R.astype(np.float32))

W_cl  = cl.array.empty(queue, (M_in_plane, Npix,), dtype = np.float32)
W_buf = np.empty((M_in_plane, Npix,), dtype = np.float32)

# wsums are split over ranks
wscale    = (Npix / wsums).astype(np.float32)
wscale_cl = cl.array.empty(queue, (M_in_plane,), dtype = np.float32)

rx_cl = cl.array.empty(queue, (Npix,), dtype = np.float32)
ry_cl = cl.array.empty(queue, (Npix,), dtype = np.float32)
cl.enqueue_copy(queue, rx_cl.data, np.ascontiguousarray(r[0][rmask_prob].astype(np.float32)))
cl.enqueue_copy(queue, ry_cl.data, np.ascontiguousarray(r[1][rmask_prob].astype(np.float32)))

C_cl = cl.array.empty(queue, (Npix,), dtype = np.float32)
cl.enqueue_copy(queue, C_cl.data , np.ascontiguousarray(C[rmask_prob].astype(np.float32)))

# copy I as an opencl "image" for trilinear sampling
I_cl = cl.Image(context, cl.mem_flags.READ_ONLY, cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT), shape = (mpx, mpx))

import time
load_time = 0
tomo_time = 0
dot_time = 0

D  = math.ceil(Ndata / dc)
Ds = np.linspace(0, D, size+1).astype(int)

if rank==0 :
    d_iter = tqdm(range(Ds[rank], Ds[rank+1]))
else :
    d_iter = range(Ds[rank], Ds[rank+1])

if not skip_prob :
    for d in d_iter:
        dstart = d * dc
        dstop  = min(dstart + dc, Ndata)
        dd     = dstop - dstart

        t0 = time.time()
        with h5py.File(fnam_data) as f:
            for t in range(dstart, dstop):
                K[t-dstart] = f['entry_1/data_1/data'][t][rmask_prob]
            K[dd:] = 0
        cl.enqueue_copy(queue, K_cl.data, K)
        load_time += time.time() - t0

        # loop over models
        for n in range(N) :
            t0 = time.time()
            for i in range(100):
                try :
                    with h5py.File(fnam_models, 'r') as f:
                        I = f['models'][n]
                
                    break
                except BlockingIOError as e :
                    print('\nCould not read file, retrying...', i)
                    time.sleep(0.01)
            
            cl.enqueue_copy(queue, I_cl, I.T.copy().astype(np.float32), is_blocking=True, origin=(0, 0), region=(mpx, mpx))
            
            cl.enqueue_copy(queue, wscale_cl.data, wscale[n])
            
            load_time += time.time() - t0
            
            # calculate all tomograms: log( (tomoscale / wscale)_r x W_ri )
            t0 = time.time()
            cl_code.calculate_tomogram_w_scale_log_batch_pix(queue, (Npix,), None, 
                I_cl, C_cl.data, rx_cl.data, ry_cl.data,  
                R_cl.data, W_cl.data, i0, np.float32(dr), 
                wscale_cl.data, np.int32(Npix), np.int32(0), np.int32(M_in_plane), np.int32(0))

            #print('\n', dd, np.sum(W_cl.get(), axis=1), '\n')
                
            queue.finish()
            
            tomo_time += time.time() - t0
            
            # logR[d, r] = sum_(i in mask) K_di log(w_ri)
            t0 = time.time()
            pyclblast.gemm(queue, dd, M_in_plane, Npix, K_cl, W_cl, logR_cl, a_ld=Npix, b_ld=Npix, c_ld = M_in_plane, b_transp=True)
            dot_time += time.time() - t0
            
            # copy to cpu
            cl.enqueue_copy(queue, logR_buf, logR_cl.data)
            logR[n] = logR_buf
        
        # write to file sequentially
        for ra in range(size):
            if rank == ra:
                # split over models 
                with h5py.File(fnam_models, 'a') as f:
                    f['logR'][:,  dstart: dstop] = logR[:, :dd]

            comm.barrier()

    if rank == 0 :
        print('\n')
        print('load  time:', load_time , 's')
        print('dot   time:', dot_time ,  's')
        print('tomo  time:', tomo_time , 's')
        print('\n')


if rank == 0 :
    # normalise logR -> probabilities
    P = np.zeros((N, Ndata, M_in_plane), dtype=np.float64)
    with h5py.File(fnam_models, 'a') as f:
        logR = f['logR'][()]
        
        for d in tqdm(range(Ndata)):
            logR_d = logR[:, d]
            Lmax = np.max(logR_d)
            
            logR_d  = np.exp(beta * (logR_d - Lmax)) 
            logR_d /= np.sum(logR_d)

            P[:, d] = logR_d

        f['probability_matrix'].write_direct(P, np.s_[:], np.s_[:])
else :
    P = None

# assume fits in memory
P = comm.bcast(P, root = 0)


# update models

Npix = np.int32(np.sum(rmask_merge))
K    = np.zeros((dc, Npix), dtype = np.float32)
K_cl = cl.array.zeros(queue, shape = (dc, Npix), dtype = np.float32)

P_buf = np.zeros(shape = (M_in_plane, dc), dtype = np.float32)
P_cl  = cl.array.zeros(queue, shape = (M_in_plane, dc), dtype = np.float32)

W_cl  = cl.array.empty(queue, (M_in_plane, Npix), dtype = np.float32)
W_buf = np.empty(             (M_in_plane, Npix), dtype = np.float32)
Wd    = np.empty(             (M_in_plane, Npix), dtype = np.float64)

PK_on_W_r     = np.empty((N, M_in_plane), dtype=np.float64)
PK_on_W_r_buf = np.empty((M_in_plane,), dtype=np.float64)

Ipix_cl  = cl.array.empty(queue, (M_in_plane, Npix), dtype = np.int32)
Ipix     = np.empty((M_in_plane, Npix), dtype = np.int32)

C_cl = cl.array.empty(queue, (Npix,), dtype = np.float32)
cl.enqueue_copy(queue, C_cl.data , np.ascontiguousarray(C[rmask_merge].astype(np.float32)))

rx_cl = cl.array.empty(queue, (Npix,), dtype = np.float32)
ry_cl = cl.array.empty(queue, (Npix,), dtype = np.float32)
cl.enqueue_copy(queue, rx_cl.data, np.ascontiguousarray(r[0][rmask_merge].astype(np.float32)))
cl.enqueue_copy(queue, ry_cl.data, np.ascontiguousarray(r[1][rmask_merge].astype(np.float32)))

# transpose P: P[n, d, r] -> P[n, r, d]
# could be expensive for larger arrays
P = np.transpose(P, (0, 2, 1)).copy()

# test:
#P.fill(1.)

assert(P.flags.c_contiguous == True)
assert(P.flags.aligned == True)

# PK_on_W_nr = sum_d P_nrd (sum_i K_di) / (sum_i W_nri) 
PK_on_W_r[:] = P.dot(ksums) 
PK_on_W_r   /= wsums


MT = Merge_tomos(N, W_buf.shape, (mpx, mpx))

start_time = time.time()
load_time = 0.
dot_time = 0.
scale_time = 0.
merge_time = 0.

dc = 512
D  = math.ceil(Ndata / dc)
Ds = np.linspace(0, D, size+1).astype(int)

if rank==0 :
    d_iter = tqdm(range(Ds[rank], Ds[rank+1]))
else :
    d_iter = range(Ds[rank], Ds[rank+1])

# loop over data
for d in d_iter:
    dstart = d * dc
    dstop  = min(dstart + dc, Ndata)
    dd     = dstop - dstart

    t0 = time.time()
    with h5py.File(fnam_data) as f:
        for t in range(dstart, dstop):
            K[t-dstart] = f['entry_1/data_1/data'][t][rmask_merge]
        K[dd:] = 0
    cl.enqueue_copy(queue, K_cl.data, K)

    load_time += time.time() - t0

    # loop over models
    for n in range(N) :
        P_buf[:, :dd] = P[n, :, dstart:dstop]
        P_buf[:, dd:] = 0
        cl.enqueue_copy(queue, P_cl.data, P_buf)
         
        # calculate dot product (tomograms) W_ri = sum_d P_rd K_di 
        t0 = time.time()
            
        pyclblast.gemm(queue, M_in_plane, Npix, dd, P_cl, K_cl, W_cl, a_ld = dc, b_ld = Npix, c_ld = Npix)
        
        queue.finish()
        dot_time += time.time() - t0
        
        # scale tomograms w_ri <-- w_ri / (sold + pol. correction C_i)
        t0 = time.time()
        cl_code.scale_tomograms_for_merge_w_coffset( queue, (Npix,), None,
                                           W_cl.data, C_cl.data,
                                           np.int32(Npix), np.int32(0), np.int32(M_in_plane), np.int32(0))
        queue.finish()
        scale_time += time.time() - t0
        
        # calculate tomogram to merged intensity pixel mappings
        t0 = time.time()
        cl_code.calculate_W_to_I_mapping_w_ioffset(queue, (Npix,), None,
                                         Ipix_cl.data, R_cl.data, rx_cl.data, ry_cl.data, np.float32(dr), 
                                         i0, np.int32(Npix), np.int32(mpx), np.int32(0), np.int32(M_in_plane), np.int32(0))
        
        cl.enqueue_copy(queue, W_buf, W_cl.data)
        cl.enqueue_copy(queue, Ipix, Ipix_cl.data)
        
        queue_cpu.finish()
        Wd[:] = W_buf
         
        PK_on_W_r_buf[:] = PK_on_W_r[n]
        
        MT.merge(n, Wd, Ipix, 0, M_in_plane, PK_on_W_r_buf, Npix, merge_I = merge_I, is_blocking=True)
        merge_time += time.time() - t0

I, O = MT.get_I_O()

O = comm.reduce(O, op=MPI.SUM, root=0)
I = comm.reduce(I, op=MPI.SUM, root=0)

if rank == 0 :
    overlap = O.copy()
    Isum = I.copy()
    O[O==0] = 1
    I /= O
    
    pickle.dump({'PK': PK_on_W_r, 'rmax': rmax_merge, 'rmin': rmin_merge, 'dr': dr, 'I': I, 'Isum': Isum, 'overlap': overlap, 't': 0, 'sample-state': 0, 'data-set': fnam_data}, open('merged_models.pickle', 'wb'))
    
    with h5py.File(fnam_models, 'a') as f:
        f['models'][...] = I
    
    # record merged intensity for each iteration
    pickle.dump(I, open('past-merged_intensities.pickle', 'ab'))
    
    total_time = time.time() - start_time 
    print('\n')
    print('total time: {:.2e}s'.format(total_time))
    print('\n')
    print('total time seconds')
    print('load  time: {:.1e}'.format( load_time))
    print('dot   time: {:.1e}'.format( dot_time))
    print('scale time: {:.1e}'.format( scale_time))
    print('merge time: {:.1e}'.format( merge_time))
    print('\n')
    print('total time %')
    print('load  time: {:5.1%}'.format( load_time / total_time))
    print('dot   time: {:5.1%}'.format( dot_time / total_time))
    print('scale time: {:5.1%}'.format( scale_time / total_time))
    print('merge time: {:5.1%}'.format( merge_time / total_time))
    print('\n')
    print("These numbers shouldn't change unless things are being done more efficiently:")
    print('load  time / Ndata / Npix        : {:.2e}'.format( load_time / Ndata / Npix))
    print('dot   time / Ndata / Mrot / Npix : {:.2e}'.format( dot_time / Ndata / M_in_plane / Npix))
    print('merge time / Mrot / Npix         : {:.2e}'.format( 100 * merge_time / M_in_plane / Npix))
