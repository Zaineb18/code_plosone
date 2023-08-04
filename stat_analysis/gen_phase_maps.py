import nibabel as nib 
import os 
from utils import * 

root_folder = "/neurospin/optimed/ZainebAmor/fmri_Dataset2/"
subj = "sub_03"
tech='_SPARK'

rot_glob = nib.load(os.path.join(os.path.join(root_folder,subj,'stat'),'eff_interest'+tech+'.nii'))
sin_clock = nib.load(os.path.join(os.path.join(root_folder,subj,'stat'),'clock_sin'+tech+'.nii'))
cos_clock = nib.load(os.path.join(os.path.join(root_folder,subj,'stat'),'clock_cos'+tech+'.nii'))
sin_anticlock = nib.load(os.path.join(os.path.join(root_folder,subj,'stat'),'anticlock_sin'+tech+'.nii'))
cos_anticlock = nib.load(os.path.join(os.path.join(root_folder,subj,'stat'),'anticlock_cos'+tech+'.nii'))
t1 = nib.load("/neurospin/optimed/ZainebAmor/fmri_Dataset2/sub_03/anat/csub_03_T1.nii")

mask, phase_clock, phase_anticlock = get_neg_pos_phases_and_mask(rot_glob, cos_clock, sin_clock, cos_anticlock, sin_anticlock)
phase = estimate_phase(phase_clock, phase_anticlock)
#phase = np.multiply(phase, mask)
phase[mask==False]=0
phase = nib.Nifti1Image(phase, rot_glob.affine)
#phase.to_filename(os.path.join(os.path.join(root_folder,subj,'stat'),'phase'+tech+'.nii'))