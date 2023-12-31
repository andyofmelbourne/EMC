# merge many tomograms 
# this is better left to a cpu implementation
# since there are many non-local memory access patterns
import pyopencl as cl
import pyopencl.array 
import numpy as np
import tqdm

gpu_precision = np.float64

# find an opencl device (preferably a CPU) in one of the available platforms
for p in cl.get_platforms():
    devices = p.get_devices(cl.device_type.CPU)
    if len(devices) > 0:
        break
    
if len(devices) == 0 :
    for p in cl.get_platforms():
        devices = p.get_devices()
        if len(devices) > 0:
            break

context = cl.Context(devices)
queue   = cl.CommandQueue(context)

cl_code = cl.Program(context, r"""

    // should I "round" or "int" q-coords?
    // I think rounding is more consistent with the trillinear interpolation 
    // used to extract values from I
    
    __kernel void merge_tomos ( 
        global double *I, global double *O, global double *W, global double *PK_on_W_r,
        global int *Ipix, 
        const int rmin, const int rmax, 
        const int Npix, const int M, const int imax)
    {
    int c = get_global_id(0);
    int C = get_global_size(0);
    
    int i, r, n, d;
    
    int offset = M * M * M * c ;

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

    __kernel void merge_tomos_I ( 
        global double *I, global double *O, global double *W, global double *PK_on_W_r,
        global int *Ipix, 
        const int rmin, const int rmax, 
        const int Npix, const int M, const int imax)
    {
    int c = get_global_id(0);
    int C = get_global_size(0);
    
    int i, r, n, d;
    
    int offset = M * M * M * c ;

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


    __kernel void sum_Is ( 
        global double *Is, global double *Os, 
        global double *I, global double *O, 
        const int M, const int c )
    {

    int n, m, offset;
    
    for (m = 0; m < c; m++) {
        offset = M*M*M*m;
        for (n = 0; n < M*M*M; n++) {
            I[n] += Is[offset + n]; 
            O[n] += Os[offset + n]; 
        }
    }
    }

""").build()

class Merge_tomos():

    def __init__(self, Wshape, Ishape):
        self.M     = np.int32(Ishape[0]);
        self.Npix  = np.int32(np.prod(Wshape[1:]))
        
        self.cu = np.int32(8)
        
        # one intensity and overlap for each compute unit
        self.Is_cl = cl.array.zeros(queue, (self.cu,) + Ishape, dtype=np.float64)
        self.Os_cl = cl.array.zeros(queue, (self.cu,) + Ishape, dtype=np.float64)
        self.I_cl = cl.array.zeros(queue, Ishape, dtype=np.float64)
        self.O_cl = cl.array.zeros(queue, Ishape, dtype=np.float64)
    
    def merge(self, W, Ipix, rmin, rmax, PK_on_W_r, imax = None, merge_I = False, is_blocking=True):
        """
        Ipix and W: (rot chunks, pix chunks)
        """
        if imax == None :
            imax = np.int32(self.Npix)
        else :
            imax = np.int32(imax)
        
        if merge_I :
            cl_merge = cl_code.merge_tomos_I
        else :
            cl_merge = cl_code.merge_tomos
        
        rmin = np.int32(rmin)
        rmax = np.int32(rmax)
        cl_merge(queue, (self.cu,), (1,), 
            self.Is_cl.data, self.Os_cl.data, cl.SVM(W), cl.SVM(PK_on_W_r), 
            cl.SVM(Ipix), rmin, rmax, self.Npix, self.M, imax)
        
        if is_blocking :
            queue.finish()
             
    def get_I_O(self):
        queue.finish()
        
        cl_code.sum_Is(queue, (1,), (1,), self.Is_cl.data, self.Os_cl.data, self.I_cl.data, self.O_cl.data, self.M, self.cu)
        
        I = np.empty(self.I_cl.shape, self.I_cl.dtype)
        O = np.empty(self.O_cl.shape, self.O_cl.dtype)
        cl.enqueue_copy(queue, I, self.I_cl.data)
        cl.enqueue_copy(queue, O, self.O_cl.data)
        return I, O


