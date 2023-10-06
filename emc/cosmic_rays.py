import h5py
import numpy as np
from scipy.ndimage import binary_dilation
from skimage import measure


# look for pixels > 5 counts outside r>128 radius
# if the integrated signal within 10x10 pixel window > 100 then it might be a cosmic ray pixel
# if the number of connected pixels is 4<50 and the counts in these pixels have an average > 5
#   then cosmic ray

fnam = 'data.cxi'
rmin = 128
threshold = 8
connected_min = 3
connected_max = 50
average = 5

max_window = np.ones((20, 20), dtype=bool)

connected_min_near = 3
connected_max_near = 100
average_near = 3

# half width
#window = 3
#window_threshold = 50
#dilation = 2



# frame shape needs to be 3D

with h5py.File(fnam, 'a') as f:
    data = f['entry_1/data_1/data']
    xyz  = f['/entry_1/instrument_1/detector_1/xyz_map'][()]
    dx   = f['entry_1/instrument_1/detector_1/x_pixel_size'][()]
    dy   = f['entry_1/instrument_1/detector_1/y_pixel_size'][()]
    
    # prevent detecting artefacts as rays
    data_mask = f['entry_1/instrument_1/detector_1/mask'][()]
    
    # pixel radius
    r = ((xyz[0]/dx)**2 + (xyz[1]/dy)**2)**0.5 
    
    rmask = (r > rmin) 
    
    #cs = []
    #fs = []
    for d in range(data.shape[0]):
        frame = data[d] * data_mask

        is_ray = False
          
        # threshold 
        mask = (frame > threshold) 
        
        # we might have a ray
        if np.any(mask) :
            cosmic_mask = np.zeros(frame.shape, dtype = bool)

            print(f'\nframe {d} has pixels above threshold')
            # loop over 2D panels
            for i in range(mask.shape[0]):
                if np.any(mask[i]) :
                    # look within max_window x max_window of threshold
                    mask2 = binary_dilation(mask[i], structure = max_window)

                    # label connected regions in mask
                    labeled, num = measure.label((mask2 * frame[i])>0, connectivity=2, return_num=True)
                    
                    # loop over labels and test
                    props = measure.regionprops(labeled, intensity_image = frame[i])
                     
                    for prop in props:
                        if prop.area < connected_max and prop.area >= connected_min :
                            print(f'    connected area is within bounds {connected_min} > {prop.area} > {connected_max}')
                            if prop.mean_intensity > average :
                                print(f'    connected area has mean intenity greater than threshold {prop.mean_intensity} > {average}')
                                cosmic_mask[i][labeled==prop.label] = True
                                is_ray = True
    
            if is_ray :
                #cs.append(np.concatenate([cosmic_mask[i] for i in range(cosmic_mask.shape[0])], axis=1))
                #fs.append(np.concatenate([frame[i] for i in range(cosmic_mask.shape[0])], axis=1))
                print(f'    adding to cosmic ray mask')
                data[d] *= cosmic_mask

