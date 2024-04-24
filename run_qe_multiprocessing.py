cls_path = '/global/homes/j/jianyao/non_gau_lensing/theory/cls/'

import plancklens
from plancklens.filt import filt_simple, filt_util, filt_cinv
from plancklens import utils
from plancklens import qest, qecl, qresp
from plancklens import nhl
from plancklens.n1 import n1
from plancklens.sims import planck2018_sims, phas, maps, utils as maps_utils

from plancklens.utils import clhash, hash_check
import os

import time
from tqdm import tqdm

import numpy as np
import healpy as hp
import multiprocessing as mp

import time
from lensre import cmb_noise, cmb_len

import argparse
# Create the parser
parser = argparse.ArgumentParser()

parser.add_argument('--fg_case', type=str, required=True)
parser.add_argument('--experiment', type=str, required=True)
parser.add_argument('--nproc', type=int, required=True, help = 'number of processes called')
parser.add_argument('--range', nargs='+', required=True, help = 'range of the realizations')
args = parser.parse_args()

add_foreground = args.fg_case
experiment = args.experiment #experiment = 'S4_LAT'
nproc = args.nproc
ranges = args.range
ranges = [int(x) for x in ranges]
print(ranges)

w = lambda ell : ell ** 2 * (ell + 1.) ** 2 * 0.5 / np.pi * 1e7

cl_unl = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))
cl_len = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))
#: Fiducial unlensed and lensed power spectra used for the analysis.

cl_weight = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))
cl_weight['bb'] *= 0.
#: CMB spectra entering the QE weights (the spectra multplying the inverse-variance filtered maps in the QE legs) 

qe_key = 'p_eb'
nside = 1024;
lmin_ivf = 100; lmax_ivf = 2000; lmax_qlm = 1000; nsim = 1; lmax_Bmode = None; 

# x_eb and p_eb results are saved simultaneously at the same time
qe_keys_dir = [qe_key.replace('x_', 'p_') if qe_key.startswith('x_') else qe_key]

transf = np.ones(lmax_ivf + 1)
nlev_t = 0; #2.16
nlev_p = 0; #2.16

if lmax_Bmode is None:
    lmax_Bmode = lmax_ivf

mask = '/pscratch/sd/j/jianyao/data_lensing/%s_mask_1024.fits'%experiment

qnorms = []; nhl_datas = []; qlms = []  
qlms_fg = []; nhl_datas_fg = []

dir_cleaned_cmb = '/pscratch/sd/j/jianyao/data_lensing/simulations/cleaned_CMB/%s_MASK/'%experiment # SO_LAT/S4_MASK

if add_foreground == 'no_fore':
    inv_with_mask = dir_cleaned_cmb + 'Inverse_noise_variance_map_mask_zeros_%s.npy'%'d9' # no foreground cases uses residual noise of pysm_d9 case
else:
    inv_with_mask = dir_cleaned_cmb + 'Inverse_noise_variance_map_mask_zeros_%s.npy'%add_foreground

ninv_t = [np.load(inv_with_mask)[0]] + [mask]
ninv_Q = [[np.load(inv_with_mask)[1]] + [mask]]         

TEMP = '/pscratch/sd/j/jianyao/data_lensing/lenre_results/%s_cinv/cleaned_cmb_%s_lmin_%s_lmax_%s_%s_lmax_Bmode_%s'%(experiment, add_foreground, lmin_ivf, lmax_ivf, qe_keys_dir[0], lmax_Bmode)

print('run for dir %s'%TEMP)

libdir_cinvt = os.path.join(TEMP, 'cinv_t')
libdir_cinvp = os.path.join(TEMP, 'cinv_p')
libdir_ivfs  = os.path.join(TEMP, 'ivfs')

cinv_t = filt_cinv.cinv_t(libdir_cinvt, lmax_ivf,nside, cl_len, transf, ninv_t,
                        marge_monopole=True, marge_dipole=True, marge_maps=[])

cinv_p = filt_cinv.cinv_p(libdir_cinvp, lmax_ivf, nside, cl_len, transf, ninv_Q)

sims = cmb_noise(cmb_len(add_foreground=add_foreground, experiment = experiment), transf, nlev_t, nlev_p, nside)

ivfs_raw = filt_cinv.library_cinv_sepTP(libdir_ivfs, sims, cinv_t, cinv_p, cl_len)
ftl = np.ones(lmax_ivf + 1, dtype=float) * (np.arange(lmax_ivf + 1) >= lmin_ivf) # rescaling or cuts. Here just a lmin cut
fel = np.ones(lmax_ivf + 1, dtype=float) * (np.arange(lmax_ivf + 1) >= lmin_ivf)
fbl = np.ones(lmax_ivf + 1, dtype=float) * (np.arange(lmax_ivf + 1) >= lmin_ivf)
ivfs   = filt_util.library_ftl(ivfs_raw, lmax_ivf, ftl, fel, fbl)

if lmax_Bmode != lmax_ivf:
    # assert qe_keys[0] == 'p_p', 'not using polarization estimator'
    fbl[lmax_Bmode:] *= 0.

qlms_dd = qest.library_sepTP(os.path.join(TEMP, 'qlms_dd'), ivfs, ivfs, cl_len['te'], nside, lmax_qlm=lmax_qlm)
nhl_dd = nhl.nhl_lib_simple(os.path.join(TEMP, 'nhl_dd'), ivfs, cl_weight, lmax_qlm)

#---- N1 lensing bias library:
# libdir_n1_dd = os.path.join(TEMP, 'n1_test')
# n1_dd = n1.library_n1(libdir_n1_dd,cl_len['tt'],cl_len['te'],cl_len['ee'])

#---- QE response calculation library:
qresp_dd = qresp.resp_lib_simple(os.path.join(TEMP, 'qresp'), lmax_ivf, cl_weight, cl_len,
                                         {'t': ivfs.get_ftl(), 'e':ivfs.get_fel(), 'b':ivfs.get_fbl()}, lmax_qlm)

def run_multi(mc):
        

    qlm = qlms_dd.get_sim_qlm(qe_key, mc)                                     
    
    if mc%10 == 0:
        print('finish %s!'%mc)
                    

    # if mc >= 199:
    #     dir_mf = TEMP+'/mean_field_%s'%qe_keys[0] #once mf are saved, the 0:200 realizations can be left alone, when estimating errors
    #     if not os.path.exists(dir_mf):
    #         os.makedirs(dir_mf)
    #         hp.write_alm(dir_mf + '/mf_200_%s.fits'%(qe_keys[0]), np.mean(qlms[0:200], axis = 0))
    #         hp.write_alm(dir_mf + '/mf_100_%s.fits'%(qe_keys[0]), np.mean(qlms[0:100], axis = 0))

s = time.time()
pool = mp.Pool(processes=nproc)
res = pool.map(run_multi, np.arange(ranges[0], ranges[1]))

pool.close()
pool.join()
e = time.time()

print('Time cost: %s mins'%((e-s)/60))
 
 