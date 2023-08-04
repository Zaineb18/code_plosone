import os
import numpy as np
import nibabel as nib

def get_4D_fMRI_data(folder):
    files = os.listdir(folder)
    files = sorted(files, key=lambda x: int((x.split('.')[0])))
    print(len(files))
    volumes = np.zeros((192,192,128,len(files)), np.float64)
    for i in range(len(files)):
        print(i)
        volumes[:,:,:,i] = np.load(os.path.join(folder, files[i]))
    return(volumes, volumes.shape)

def from_npy_to_nifti(npy_file, ref_nifti):
    npy_volumes = np.load(npy_file)
    npy_volumes = np.moveaxis(npy_volumes, (0,1,2,3), (1,0,2,3))
    npy_volumes = npy_volumes[:,:,::-1,:]
    aff = nib.load(ref_nifti).affine
    nifti_volumes = nib.Nifti1Image(npy_volumes, affine=aff)
    return(nifti_volumes)

