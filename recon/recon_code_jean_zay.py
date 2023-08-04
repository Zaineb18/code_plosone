#from mri.operators.fourier.orc_wrapper import ORCFFTWrapper
#import os
import numpy as np
import pysap
from mri.operators import Stacked3DNFFT, NonCartesianFFT, WaveletN, FFT, ORCFFTWrapper
from mri.operators.utils import normalize_frequency_locations
from mri.reconstructors import SelfCalibrationReconstructor,CalibrationlessReconstructor
from mri.operators.fourier.utils import estimate_density_compensation
from modopt.opt.linear import Identity
from modopt.opt.proximity import SparseThreshold, Ridge
from sparkling.utils.gradient import get_kspace_loc_from_gradfile
from sparkling.utils.shots import convert_NCxNSxD_to_NCNSxD, convert_NCNSxD_to_NCxNSxD
import mapvbvd
import argparse

def get_samples(filename,dwell_t, num_adc_samples, kmax):
    sample_locations = convert_NCxNSxD_to_NCNSxD(get_kspace_loc_from_gradfile(filename, dwell_t, num_adc_samples)[0])
    sample_locations = normalize_frequency_locations(sample_locations, Kmax= kmax)
    return sample_locations

def add_phase_kspace(kspace_data, kspace_loc, shifts={}):
    if shifts == {}:
        shifts = (0,) * kspace_loc.shape[1]
    if len(shifts) != kspace_loc.shape[1]:
        raise ValueError("Dimension mismatch between shift and kspace locations! "
                         "Ensure that shifts are right")
    phi = np.zeros_like(kspace_data)
    for i in range(kspace_loc.shape[1]):
        phi += kspace_loc[:, i] * shifts[i]
    phase = np.exp(-2 * np.pi * 1j * phi)
    return kspace_data * phase

def read_shifts(twixObj):
    y = twixObj.search_header_for_val('Phoenix',('sWiPMemBlock','adFree' , '6'))
    if len(y)!=0:
        y=y[0]
    else:
        y=0.0
    x = twixObj.search_header_for_val('Phoenix',('sWiPMemBlock','adFree' , '7'))
    if len(x)!=0:
        x=x[0]
    else:
        x=0.0
    z = twixObj.search_header_for_val('Phoenix',('sWiPMemBlock','adFree' , '8'))
    if len(z)!=0:
        z=z[0]
    else:
        z=0.0
    return(x,y,z)

def read_kspace_data_rep(twixObj, i):
    kspace_data = twixObj.image[:,:,:,i]
    kspace_data = np.moveaxis(kspace_data, (0,1,2),(2,0,1))
    kspace_data = np.reshape(kspace_data, (32,kspace_data.shape[1]*kspace_data.shape[2]))
    return(kspace_data)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--i",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--obs",
        type=str,
    )
    parser.add_argument(
        "--out",
        type=str,
    )
    parser.add_argument(
        "--mask",
        type=str,
    )
    parser.add_argument(
        "--b0",
        type=str,
    )
    parser.add_argument(
        "--smaps",
        type=str,
    )
    args = parser.parse_args()
    i =  args.i
    obs_filename = args.obs
    out_filename = args.out
    mask_filename = args.mask
    b0_filename = args.b0
    smaps_filename = args.smaps
    # get the b0 map and its mask
    b0Map = np.load(b0_filename)
    mask = np.load(mask_filename)
    # define the traj parameters
    M = [192, 192, 128]
    FOV = [0.192, 0.192, 0.128]
    Ns = 2688
    OS = 5
    # get the data samples and the kspace samples
    kspace_loc = []
    kspace_data = []
    traj_file = "/gpfsstore/rech/hih/uwa98fg/InputData/at_retest/dim3_i_RadialIO_P0.75_N192x192x128_FOV0.192x0.192x0.128_Nc8_Ns2688_c25_d2__D9M9Y2021T1017_reproject.bin"
    kspace_loc = get_samples(traj_file, 0.01 / OS, Ns * OS,
                             kmax=(M[0] / (2 * FOV[0]), M[1] / (2 * FOV[1]), M[2] / (2 * FOV[2])))
    twixObj = mapvbvd.mapVBVD(obs_filename)
    x_shift, y_shift, z_shift = read_shifts(twixObj)
    print(x_shift, y_shift, z_shift)
    twixObj.image.flagRemoveOS = False
    twixObj.image.squeeze = True
    kspace_data = read_kspace_data_rep(twixObj, i)
    kspace_data = add_phase_kspace(kspace_data, kspace_loc, shifts=(x_shift, y_shift, z_shift))
    # for the extended Fourier op
    dwell_time = 10e-6 / 5
    nb_adc_samples = 13440
    Te = 20e-3
    time_vec = dwell_time * np.arange(nb_adc_samples)
    echo_time = Te - dwell_time * nb_adc_samples / 2
    time_vec = (time_vec + echo_time).astype(np.float32)
    # define the operators and the reconstructor
    density_comp = estimate_density_compensation(kspace_loc, M)
    smaps = np.load(smaps_filename, allow_pickle=True, fix_imports=True)
    regularizer_op = SparseThreshold(Identity(), 1e-8 , thresh_type="soft")
    #regularizer_op = Ridge(Identity(), 1e-8)
    linear_op = WaveletN(wavelet_name='sym8',
                         nb_scale=3,
                         dim=3,
                         padding='periodization')
    fourier_op = NonCartesianFFT(samples=kspace_loc,
                                 shape=M,
                                 implementation='gpuNUFFT',
                                 density_comp=density_comp,
                                 n_coils=32,
                                 #smaps=smaps
                                 )
    ifourier_op = ORCFFTWrapper(fourier_op, b0Map, time_vec, mask, n_bins=1000, num_interpolators=30)
    reconstructor = SelfCalibrationReconstructor(
        fourier_op=ifourier_op,
        linear_op=linear_op,
        regularizer_op=regularizer_op,
        gradient_formulation='synthesis',
        Smaps=smaps,
        num_check_lips=0,
        #lipschitz_cst=0.06,
        #lips_calc_max_iter = 5,
        n_jobs=1,
        verbose=5
    )
    reconst_data, cost, _ = reconstructor.reconstruct(kspace_data=kspace_data,
                                                      optimization_alg='pogm', num_iterations=12,
                                                      recompute_smaps=False,
                                                      # cost_op_kwargs={'cost_interval': 1}
                                                      )
    #reconst_data = ifourier_op.adj_op(kspace_data)
    filename=str(i)
    np.save(out_filename+filename,abs(reconst_data), allow_pickle=True, fix_imports=True)











