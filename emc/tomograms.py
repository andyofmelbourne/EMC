import argparse

if __name__ == '__main__':
    description = \
    """
    Calculate tomogram sums. 
    """
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--data', type=str, default='data.cxi', \
                        help="cxi file containing data frames, for geometry.")
    parser.add_argument('-f', '--emc_file', type=str, default = "EMC.h5", \
                        help="EMC h5 file.")
    args = parser.parse_args()





import numpy as np
import tqdm

import pyopencl as cl
import pyopencl.array 

gpu_precision = np.float32

# find an opencl device (preferably a GPU) in one of the available platforms
for p in cl.get_platforms():
    devices = p.get_devices(cl.device_type.GPU)
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
    #include <pyopencl-complex.h>

    constant sampler_t trilinear = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR ;

    float4 _calculate_I_coord (
    float qx, 
    float qy,
    float qz, 
    float *R, 
    const float i0, 
    const float dq)
    {
    float4 coord ;
    
    coord.x = i0 + (R[0] * qx + R[1] * qy + R[2] * qz) / dq + 0.5;
    coord.y = i0 + (R[3] * qx + R[4] * qy + R[5] * qz) / dq + 0.5;
    coord.z = i0 + (R[6] * qx + R[7] * qy + R[8] * qz) / dq + 0.5;
    
    return coord;
    }

    
    // one worker per tomogram index
    float _calculate_tomogram (
    image3d_t I, 
    float C,  
    float qx, 
    float qy,
    float qz, 
    float *R, 
    const float i0, 
    const float dq)
    {
    float4 coord = _calculate_I_coord(qx,qy,qz,R,i0,dq);
    
    float4 v = read_imagef(I, trilinear, coord);
    
    return v.x * C;
    }

    __kernel void calculate_tomogram (
    image3d_t I, 
    global float *Cg,  
    global float *qxg, 
    global float *qyg,
    global float *qzg, 
    global float *Rg, 
    global float *out, 
    const float i0, 
    const float dq, 
    const int Npix,
    const int r)
    {
    int n = get_global_id(0);

    float C = Cg[n];  
    float qx = qxg[n];
    float qy = qyg[n];
    float qz = qzg[n];

    float R[9];

    for (int i=0; i<9; i++) {
        R[i] = Rg[9*r + i];
    }
    
    out[n] = _calculate_tomogram(I, C,  qx, qy, qz, R, i0, dq);
    }

    __kernel void calculate_tomogram_batch (
    image3d_t I, 
    global float *Cg,  
    global float *qxg, 
    global float *qyg,
    global float *qzg, 
    global float *Rg, 
    global float *out, 
    const float i0, 
    const float dq, 
    const int Npix,
    const int rmin, 
    const int rmax)
    {
    int n = get_global_id(0);

    float C = Cg[n];  
    float qx = qxg[n];
    float qy = qyg[n];
    float qz = qzg[n];

    float R[9];
    
    for (int r=rmin; r<rmax; r++){
        for (int i=0; i<9; i++) {
            R[i] = Rg[9*r + i];
        }
        
        out[Npix * (r-rmin) + n] = _calculate_tomogram(I, C,  qx, qy, qz, R, i0, dq);
    }
    }


    __kernel void calculate_W_to_I_mapping(
    global int *out,
    global float *Rg, 
    global float *qxg, 
    global float *qyg,
    global float *qzg, 
    const float dq, 
    const float i0, 
    const int Npix,
    const int M,
    const int dr)
    {

    int n = get_global_id(0);

    float qx = qxg[n];
    float qy = qyg[n];
    float qz = qzg[n];

    float4 coord;

    for (int r=0; r<dr; r++){
        float R[9];

        for (int i=0; i<9; i++) {
            R[i] = Rg[9*r + i];
        }
        
        float4 coord = _calculate_I_coord(qx,qy,qz,R,i0,dq);
         
        // get flattened I index
        out[Npix * r + n] = convert_int_rte(coord.x) * M * M + convert_int_rte(coord.y) * M + convert_int_rte(coord.z);
    }
    }

    
    __kernel void calculate_W_to_I_mapping_w_ioffset(
    global int *out,
    global float *Rg, 
    global float *qxg, 
    global float *qyg,
    global float *qzg, 
    const float dq, 
    const float i0, 
    const int Npix,
    const int M,
    const int rstart,
    const int rstop,
    const int ioff)
    {

    int n = get_global_id(0);

    float qx = qxg[ioff + n];
    float qy = qyg[ioff + n];
    float qz = qzg[ioff + n];

    float4 coord;

    for (int r=rstart; r<rstop; r++){
        float R[9];

        for (int i=0; i<9; i++) {
            R[i] = Rg[9*r + i];
        }
        
        float4 coord = _calculate_I_coord(qx,qy,qz,R,i0,dq);
         
        // get flattened I index
        out[Npix * (r-rstart) + n] = convert_int_rte(coord.x) * M * M + convert_int_rte(coord.y) * M + convert_int_rte(coord.z);
    }
    }


    // one worker per index
    __kernel void calculate_tomogram_w_scale_log (
    image3d_t I, 
    global float *Cg,  
    global float *qxg, 
    global float *qyg,
    global float *qzg, 
    global float *Rg, 
    global float *out, 
    const float i0, 
    const float dq, 
    global float *wscale, 
    const int Npix,
    const int r)
    {
    int n = get_global_id(0);

    float C = Cg[n];  
    float qx = qxg[n];
    float qy = qyg[n];
    float qz = qzg[n];
    
    float R[9];

    for (int i=0; i<9; i++) {
        R[i] = Rg[9*r + i];
    }
    
    out[n] = log(wscale[r] * _calculate_tomogram(I, C,  qx, qy, qz, R, i0, dq));
    }

    // one worker per index
    __kernel void calculate_tomogram_w_scale_log_batch (
    image3d_t I, 
    global float *Cg,  
    global float *qxg, 
    global float *qyg,
    global float *qzg, 
    global float *Rg, 
    global float *out, 
    const float i0, 
    const float dq, 
    global float *wscale, 
    const int Npix,
    const int rmin,
    const int rmax)
    {
    int n = get_global_id(0);
    
    float C = Cg[n];  
    float qx = qxg[n];
    float qy = qyg[n];
    float qz = qzg[n];
    
    float R[9];
    
    float t;
    
    for (int r=rmin; r<rmax; r++){
        for (int i=0; i<9; i++) {
            R[i] = Rg[9*r + i];
        }
        
        t = wscale[r] * _calculate_tomogram(I, C,  qx, qy, qz, R, i0, dq);

        if (t > 0.) 
            out[r*Npix + n] = log(t);
        else 
            out[r*Npix + n] = 0.;
    }
    }

    // one worker per pixel index
    __kernel void calculate_tomogram_w_scale_log_batch_pix (
    image3d_t I, 
    global float *Cg,  
    global float *qxg, 
    global float *qyg,
    global float *qzg, 
    global float *Rg, 
    global float *out, 
    const float i0, 
    const float dq, 
    global float *wscale, 
    const int Npix,
    const int rmin,
    const int rmax,
    const int ioff)
    {
    int n = get_global_id(0);
    
    float C  = Cg[ioff + n];  
    float qx = qxg[ioff + n];
    float qy = qyg[ioff + n];
    float qz = qzg[ioff + n];
    
    float R[9];
    
    float t;
    
    for (int r=rmin; r<rmax; r++){
        for (int i=0; i<9; i++) {
            R[i] = Rg[9*r + i];
        }
        
        t = wscale[r] * _calculate_tomogram(I, C,  qx, qy, qz, R, i0, dq);
        
        if (t > 0.) 
            out[r*Npix + n] = log(t);
        else 
            out[r*Npix + n] = 0.;
    }
    }



    __kernel void scale_tomograms_for_merge (
    global float *W,  
    global float *wsums,  
    global float *PK,  
    global float *C, 
    const int Npix, 
    const int rmin, 
    const int rmax)
    {
    int n = get_global_id(0);

    for (int r=rmin; r<rmax; r++)
        W[r*Npix + n] *= wsums[r] / C[n] / PK[r];
    
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


def calculate_tomograms(I, C, q, qmask, R, dq, rs):
    rc = len(rs)
    
    Mrot = R.shape[0]
    
    # number of pixels within qmask
    Npix  = np.int32(np.sum(qmask))
    # pixel coordinate of q = 0 in I
    i0 = np.float32((I.shape[0] - 1)//2)
    
    R_cl      = cl.array.empty(queue, (9*Mrot,), dtype = np.float32)
    W_cl      = cl.array.empty(queue, (Npix,), dtype = np.float32)
    wscale_cl = cl.array.empty(queue, (Mrot,), dtype = np.float32)
    qx_cl     = cl.array.empty(queue, (Npix,), dtype = np.float32)
    qy_cl     = cl.array.empty(queue, (Npix,), dtype = np.float32)
    qz_cl     = cl.array.empty(queue, (Npix,), dtype = np.float32)
    C_cl      = cl.array.empty(queue, (Npix,), dtype = np.float32)
    
    cl.enqueue_copy(queue, R_cl.data, R.astype(np.float32))
    cl.enqueue_copy(queue, qx_cl.data, np.ascontiguousarray(q[0][qmask].astype(np.float32)))
    cl.enqueue_copy(queue, qy_cl.data, np.ascontiguousarray(q[1][qmask].astype(np.float32)))
    cl.enqueue_copy(queue, qz_cl.data, np.ascontiguousarray(q[2][qmask].astype(np.float32)))
    cl.enqueue_copy(queue, C_cl.data , np.ascontiguousarray(C[qmask].astype(np.float32)))
    
    # copy I as an opencl "image" for trilinear sampling
    I_cl = cl.Image(context, cl.mem_flags.READ_ONLY, cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT), shape=I.shape[::-1])
    cl.enqueue_copy(queue, I_cl, I.T.copy().astype(np.float32), is_blocking=True, origin=(0, 0, 0), region=I.shape[::-1])
    
    Ws = np.empty((rc, Npix,), dtype = np.float32)
    W  = np.empty((Npix,), dtype = np.float32)

    # calculate tomograms 
    index = 0
    for r in tqdm.tqdm(rs, desc='calculating tomogram sums'):
        cl_code.calculate_tomogram(queue, (Npix,), None,
                I_cl, C_cl.data, qx_cl.data, qy_cl.data, qz_cl.data, 
                R_cl.data, W_cl.data, i0, np.float32(dq), Npix, np.int32(r))
        
        cl.enqueue_copy(queue, W, W_cl.data)

        Ws[index] = W
        index += 1
    return Ws
    

def calculate_tomogram_sums(I, C, q, qmask, R, dq, rc=256):
    Mrot = R.shape[0]
    
    # number of pixels within qmask
    Npix  = np.int32(np.sum(qmask))
    # pixel coordinate of q = 0 in I
    i0 = np.float32(I.shape[0]//2)
    
    R_cl      = cl.array.empty(queue, (9*Mrot,), dtype = np.float32)
    W_cl      = cl.array.empty(queue, (rc, Npix,), dtype = np.float32)
    wscale_cl = cl.array.empty(queue, (Mrot,), dtype = np.float32)
    qx_cl     = cl.array.empty(queue, (Npix,), dtype = np.float32)
    qy_cl     = cl.array.empty(queue, (Npix,), dtype = np.float32)
    qz_cl     = cl.array.empty(queue, (Npix,), dtype = np.float32)
    C_cl      = cl.array.empty(queue, (Npix,), dtype = np.float32)
    
    cl.enqueue_copy(queue, R_cl.data, R.astype(np.float32))
    cl.enqueue_copy(queue, qx_cl.data, np.ascontiguousarray(q[0][qmask].astype(np.float32)))
    cl.enqueue_copy(queue, qy_cl.data, np.ascontiguousarray(q[1][qmask].astype(np.float32)))
    cl.enqueue_copy(queue, qz_cl.data, np.ascontiguousarray(q[2][qmask].astype(np.float32)))
    cl.enqueue_copy(queue, C_cl.data , np.ascontiguousarray(C[qmask].astype(np.float32)))
    
    # copy I as an opencl "image" for trilinear sampling
    I_cl = cl.Image(context, cl.mem_flags.READ_ONLY, cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT), shape=I.shape[::-1])
    cl.enqueue_copy(queue, I_cl, I.T.copy().astype(np.float32), is_blocking=True, origin=(0, 0, 0), region=I.shape[::-1])
    
    wsums_out = np.empty((Mrot,))
    W         = np.empty((rc, Npix,), dtype = np.float32)
    
    # calculate tomograms then sum then output
    for rmin in tqdm.tqdm(range(0, Mrot, rc), desc='calculating tomogram sums'):
        rmax = min(rmin + rc, Mrot)
        cl_code.calculate_tomogram_batch(queue, (Npix,), None,
                I_cl, C_cl.data, qx_cl.data, qy_cl.data, qz_cl.data, 
                R_cl.data, W_cl.data, i0, np.float32(dq), Npix, np.int32(rmin), np.int32(rmax))
        
        cl.enqueue_copy(queue, W, W_cl.data)
        
        wsums_out[rmin:rmax] = np.sum(W, axis=1)[:rmax-rmin]
    return wsums_out


if __name__ == '__main__':
    import h5py
    
    with h5py.File(args.data) as f:
        Ndata = f['entry_1/data_1/data'].shape[0]
    
    with h5py.File(args.emc_file) as f:
        I     = f['merged_intensity'][()]
        dq    = np.float32(f['dq'][()])
        q     = f['q'][()]
        qmask = f['qmask_prob'][()]
        Mrot       = f['Mrot'][()]
        C          = f['solid_angle_polarisation_correction'][()]
        R          = f['rotation_matrices'][()]
    
    # number of pixels within qmask
    Npix  = np.int32(np.sum(qmask))
    # pixel coordinate of q = 0 in I
    i0 = np.float32((I.shape[0] - 1)//2)
    
    R_cl  = cl.array.empty(queue, (9*Mrot,), dtype = np.float32)
    W_cl       = cl.array.empty(queue, (Npix,), dtype = np.float32)
    wscale_cl  = cl.array.empty(queue, (Mrot,), dtype = np.float32)
    qx_cl = cl.array.empty(queue, (Npix,), dtype = np.float32)
    qy_cl = cl.array.empty(queue, (Npix,), dtype = np.float32)
    qz_cl = cl.array.empty(queue, (Npix,), dtype = np.float32)
    C_cl  = cl.array.empty(queue, (Npix,), dtype = np.float32)
    
    cl.enqueue_copy(queue, R_cl.data, R.astype(np.float32))
    cl.enqueue_copy(queue, qx_cl.data, np.ascontiguousarray(q[0][qmask].astype(np.float32)))
    cl.enqueue_copy(queue, qy_cl.data, np.ascontiguousarray(q[1][qmask].astype(np.float32)))
    cl.enqueue_copy(queue, qz_cl.data, np.ascontiguousarray(q[2][qmask].astype(np.float32)))
    cl.enqueue_copy(queue, C_cl.data , np.ascontiguousarray(C[qmask].astype(np.float32)))
    
    # copy I as an opencl "image" for trilinear sampling
    I_cl = cl.Image(context, cl.mem_flags.READ_ONLY, cl.ImageFormat(cl.channel_order.R, cl.channel_type.FLOAT), shape=I.shape[::-1])
    cl.enqueue_copy(queue, I_cl, I.T.copy().astype(np.float32), is_blocking=True, origin=(0, 0, 0), region=I.shape[::-1])
    
    wsums_out = np.empty((Mrot,))
    W         = np.empty((Npix,), dtype = np.float32)
    
    # calculate tomograms then sum then output
    for r in tqdm.tqdm(range(Mrot), desc='calculating tomogram sums'):
        cl_code.calculate_tomogram(queue, (Npix,), None,
                I_cl, C_cl.data, qx_cl.data, qy_cl.data, qz_cl.data, 
                R_cl.data, W_cl.data, i0, dq, Npix, np.int32(r))
        
        queue.finish()
        
        cl.enqueue_copy(queue, W, W_cl.data)
        
        wsums_out[r] = np.sum(W)

         
    # write result
    with h5py.File(args.emc_file, 'a') as f:
        f['tomogram_sums'][...] = wsums_out

