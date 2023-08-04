from utils import * 

#read all files
root_dir = "/neurospin/optimed/ZainebAmor/fmri_Dataset2/"
sub_d, subject_id = subject_dict(root_dir, 'sub_03')
clock_files_epi, anticlock_files_epi = get_Clock_AntiClock(sub_d['func'], 'func', root_dir, subject_id, method='EPI3D')
clock_files_spark, anticlock_files_spark = get_Clock_AntiClock(sub_d['func'], 'func', root_dir, subject_id)
retino_data_clock, retino_data_anticlock, motion_clock, motion_anticlock = load_fmri_volumes_and_confounds(clock_files_spark, anticlock_files_spark)
t1_den = load_T1(root_dir, subject_id, sub_d['anat'])


#design matrix
X_clock = make_first_level_single_sess_design_matrix(motion_clock, n_scans=120, tr=2.4,n_cycles=9, sign=-1, hrf_model='spm')
X_anticlock = make_first_level_single_sess_design_matrix(motion_anticlock, n_scans=120, tr=2.4, n_cycles=9, sign=1, hrf_model='spm')
design_matrices = [X_clock, X_anticlock]

#fit GLM
glm_clock = make_and_fit_glm(retino_data_clock, X_clock, tr=2.4)
glm_anticlock = make_and_fit_glm(retino_data_anticlock, X_anticlock, tr=2.4)

fmri_time_series = retino_data_clock, retino_data_anticlock
glm_glob = make_and_fit_glm(fmri_time_series, design_matrices, tr=2.4)

rot_clock, sin_clock, cos_clock = run_glm(glm_clock, X_clock.keys() )
rot_anticlock, sin_anticlock, cos_anticlock = run_glm(glm_anticlock, X_anticlock.keys() )
rot_glob, sin_glob, cos_glob = run_glm(glm_glob, design_matrices[0].keys() )

#rot_glob['z_score'].to_filename(os.path.join(os.path.join(root_dir, subject_id),'stat', 'eff_int_SPARK'))
#sin_clock['z_score'].to_filename(os.path.join(os.path.join(root_dir, subject_id),'stat', 'clock_sin_SPARK'))
#cos_clock['z_score'].to_filename(os.path.join(os.path.join(root_dir, subject_id),'stat', 'clock_cos_SPARK'))
#sin_anticlock['z_score'].to_filename(os.path.join(os.path.join(root_dir, subject_id),'stat', 'anticlock_sin_SPARK'))
#cos_anticlock['z_score'].to_filename(os.path.join(os.path.join(root_dir, subject_id),'stat', 'anticlock_cos_SPARK'))