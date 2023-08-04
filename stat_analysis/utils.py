import nibabel as nib 
import os 
import pandas as pd 
from pandas import read_csv
from nilearn.image import index_img, mean_img
from nilearn import plotting
from nilearn.plotting import plot_anat, plot_surf_stat_map
import numpy as np
from pylab import *
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix
import matplotlib.pyplot as plt
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm import threshold_stats_img
from nilearn.plotting import plot_stat_map, plot_anat, plot_img



def subject_dict(root_dir, subject_id='sub_01'):
    Acq_Folders = os.listdir(os.path.join(root_dir, subject_id))
    sub_d={}
    for folder in Acq_Folders:
        files = os.listdir(os.path.join(os.path.join(root_dir, subject_id), folder))
        sub_d.update({folder: files})
    return(sub_d, subject_id)

def get_Clock_AntiClock(files, folder, root_dir, subject_id, method='SPARKLING3D'):
    antiClock_files= [os.path.join(os.path.join(root_dir, subject_id), folder,f) for  f in files if 'AntiClock' in f and method in f ]
    Clock_files = [os.path.join(os.path.join(root_dir, subject_id), folder,f) for  f in files if 'AntiClock' not in f and not(f.startswith('.')) and method in f]
    print(Clock_files,'\n', antiClock_files,'\n')
    return(Clock_files, antiClock_files)

def load_fmri_volumes_and_confounds(clock_files, anticlock_files):
    retino_data_clock= nib.load([f for f in clock_files if f.endswith("nii")][0])
    retino_data_anticlock= nib.load([f for f in anticlock_files if f.endswith("nii")][0])
    motion_clock = read_csv([f for f in clock_files if f.endswith("txt")][0], sep='  ')
    motion_anticlock = read_csv([f for f in anticlock_files if f.endswith("txt")][0], sep='  ')
    return(retino_data_clock, retino_data_anticlock, motion_clock, motion_anticlock)

def load_T1(root_dir, subject_id,  T1_files):
    t1_den= nib.load([os.path.join(os.path.join(root_dir, subject_id),'anat',f) for f in T1_files if f.endswith("nii")][0])
    return(t1_den)

def make_sinus_regressors(n_cycles=9, n_scans=120, sign=1):
    # sign = 1 if anticlockwise and -1 else
    cos_sin_regs = np.zeros((n_scans,2))
    for i in range(n_scans):
        cos_sin_regs[i,0] = cos( - np.pi/2 + sign*i*2*n_cycles*np.pi/n_scans)
        cos_sin_regs[i,1] = sin( - np.pi/2 + sign*i*2*n_cycles*np.pi/n_scans)
    return(cos_sin_regs)

def make_first_level_single_sess_design_matrix(motion, n_scans, tr, n_cycles, sign, hrf_model='spm'):
    frame_times = np.arange(n_scans) * tr
    motion = np.asarray(motion)
    cos_sin_regs = make_sinus_regressors(n_cycles, n_scans=n_scans, sign=sign)
    add_regs = np.hstack((cos_sin_regs, motion))
    X = make_first_level_design_matrix(frame_times, drift_model='polynomial',
                                    drift_order=1, hrf_model=hrf_model,
                                    add_regs=add_regs,
                                    add_reg_names=['cos', 'sin', 'tx','ty', 'tz', 'rx', 'ry', 'rz'], 
                                    min_onset=44
                                    )
    return(X)

def make_and_fit_glm(fmri_time_series, design_matrices, tr=2.4):
    fmri_glm = FirstLevelModel(t_r=tr, n_jobs=10,)
    fmri_glm = fmri_glm.fit(fmri_time_series, design_matrices=design_matrices)
    return(fmri_glm)

def _elementary_contrasts(design_matrix_columns):
    """Returns a dictionary of contrasts for all columns
        of the design matrix"""
    con = {}
    n_columns = len(design_matrix_columns)
    # simple contrasts
    for i in range(n_columns):
        
        con[design_matrix_columns[i]] = np.eye(n_columns)[i]
    return con

def retino(design_matrix_columns):
    el_con = _elementary_contrasts(design_matrix_columns)
    contrasts = {
                 'cos': el_con['cos'],
                 'sin': el_con['sin'],
                 'rotation' : np.vstack((el_con['cos'],
                                         el_con['sin']))
                 }
    return contrasts

def run_glm(fmri_glm, design_matrix_columns):
    contrast = retino(design_matrix_columns)
    output_rot  = fmri_glm.compute_contrast(contrast['rotation'], stat_type='F', output_type='all')
    output_cos = fmri_glm.compute_contrast(contrast['cos'], stat_type='t', output_type='all')
    output_sin = fmri_glm.compute_contrast(contrast['sin'], stat_type='t', output_type='all')
    return(output_rot, output_sin, output_cos)

def shift_cmap(cmap, frac):
    """Shifts a colormap by a certain fraction.

    Keyword arguments:
    cmap -- the colormap to be shifted. Can be a colormap name or a Colormap object
    frac -- the fraction of the colorbar by which to shift (must be between 0 and 1)
    """
    N=256
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    n = cmap.name
    x = np.linspace(0,1,N)
    out = np.roll(x, int(N*frac))
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(f'{n}_s', cmap(out))
    return new_cmap



def get_neg_pos_phases_and_mask(rot_glob, cos_clock, sin_clock, cos_anticlock, sin_anticlock):
    rot_glob_thresh, threshold = threshold_stats_img(rot_glob, alpha=0.001, height_control='fpr',
                                          two_sided=False )
    mask = rot_glob_thresh.get_fdata()>=threshold
    phase_clock = np.arctan2(-cos_clock.get_fdata(),-sin_clock.get_fdata())
    phase_anticlock = np.arctan2(cos_anticlock.get_fdata(),-sin_anticlock.get_fdata())
    return(mask, phase_clock, phase_anticlock)


def estimate_phase(phase_clock, phase_anticlock):
    # estimate hemodynamic delay
    hemo = 0.5 * (phase_clock + phase_anticlock)
    hemo += np.pi * (hemo < 0)
    hemo += np.pi * (hemo < 0)
    pr1 = phase_clock - hemo
    pr2 = - phase_anticlock + hemo
    pr2[(pr1 - pr2) > np.pi] += (2 * np.pi)
    pr2[(pr1 - pr2) > np.pi] += (2 * np.pi)
    pr1[(pr2 - pr1) > np.pi] += (2 * np.pi)
    pr1[(pr2 - pr1) > np.pi] += (2 * np.pi)
    phase = 0.5 * (pr1 + pr2)

    # add the offset and bring back to [-pi, +pi]
    phase += np.pi/2
    phase += 2 * np.pi * (phase < - np.pi)
    phase += 2 * np.pi * (phase < - np.pi)
    phase -= 2 * np.pi * (phase > np.pi)
    phase -= 2 * np.pi * (phase > np.pi)
    return(phase)