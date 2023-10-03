import pyopencl as cl
import pyopencl.array 
import numpy as np
import tqdm

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

# perhaps opencl image has each pixel coordinate at 0.5, 1.5 ... (N-1) + 0.5
# yes thats it # the coordinates are also transposed
cl_code = cl.Program(context, r"""
    
    // R = Rz(theta).dot(Ry(phi).dot(Rz(phi2)))
    //[-sin(phi_2)*sin(theta) + cos(phi)*cos(phi_2)*cos(theta), -sin(phi_2)*cos(phi)*cos(theta) - sin(theta)*cos(phi_2), sin(phi)*cos(theta)],
    //[ sin(phi_2)*cos(theta) + sin(theta)*cos(phi)*cos(phi_2), -sin(phi_2)*sin(theta)*cos(phi) + cos(phi_2)*cos(theta), sin(phi)*sin(theta)],
    //[                                   -sin(phi)*cos(phi_2),                                     sin(phi)*sin(phi_2),            cos(phi)]])
    
    __kernel void calculate_rotation_matrix ( global float *R, const int M_in_plane, const int M_sphere)
    {
    // this is the rotation index r
    int r = get_global_id(0);
    
    int n = r / M_in_plane ;
    int m = r % M_in_plane ;
    
    float i = n + 0.5;
    float phi  = acos(1 - 2 * i / M_sphere);
    float phi_2 = 2. * M_PI * m / M_in_plane; 
    float goldenRatio = (1. + sqrt(5.)) / 2.;
    float theta = 2. * M_PI * i / goldenRatio;
    
    R[r * 9 + 0] = -sin(phi_2) * sin(theta) + cos(phi) * cos(phi_2) * cos(theta) ;
    R[r * 9 + 1] = -sin(phi_2)*cos(phi)*cos(theta) - sin(theta)*cos(phi_2) ;
    R[r * 9 + 2] = sin(phi)*cos(theta) ;
    
    R[r * 9 + 3] = sin(phi_2)*cos(theta) + sin(theta)*cos(phi)*cos(phi_2) ;
    R[r * 9 + 4] = -sin(phi_2)*sin(theta)*cos(phi) + cos(phi_2)*cos(theta) ;
    R[r * 9 + 5] = sin(phi)*sin(theta) ;
    
    R[r * 9 + 6] = -sin(phi)*cos(phi_2) ;
    R[r * 9 + 7] = sin(phi)*sin(phi_2) ;
    R[r * 9 + 8] = cos(phi) ;
    }

""").build()


def calculate_rotation_matrices(Mrot, M_in_plane, M_sphere):
    # pre-calculate rotations
    # -----------------------
    R_cl  = cl.array.empty(queue, (9*Mrot,), dtype = np.float32)
    for t in tqdm.tqdm(range(1), desc='pre-calculating rotation matrices'):
        cl_code.calculate_rotation_matrix(queue, (Mrot,), None, R_cl.data, np.int32(M_in_plane), np.int32(M_sphere))
    
    R = R_cl.get().reshape((-1, 3, 3)).astype(np.float32)
    return R
