import multiprocessing as mp

import sys
sys.path.append('/global/homes/j/jianyao/')
import os
from cmbdb import cmbdb
import numpy as np
import healpy as hp

tag = 'S4_LAT'
df = cmbdb.loc[cmbdb['experiment'].isin(tag.split())]
instrument= df.dropna(axis=1, how='all')

mask = '/pscratch/sd/j/jianyao/data_lensing/%s_mask_1024.fits'%tag

if mask is not None:
    mask_so = hp.read_map(mask)
    fsky = mask_so.sum()/mask_so.size

nside = 1024
data_dir = '/pscratch/sd/j/jianyao/data_lensing/simulations/cleaned_CMB/%s_MASK/'%tag
# cases = ['forse3_d9']
# cases = ['forse3_Gaussiand9']
# cases = ['d9']
cases = ['forse3_d10']
add_foreground = cases[0]
inv_path = data_dir + 'Inverse_noise_variance_map_%s.npy'%add_foreground
inv_with_mask = data_dir + 'Inverse_noise_variance_map_mask_zeros_%s.npy'%add_foreground

def run(mc):
    # for add_foreground in cases:


    if not os.path.exists(inv_path):
        noise_alm = hp.read_alm(data_dir + 'Noise_ilc_alms_from_%s_%s_HILC_lbins_42x50_lmax_2050_nside_1024_%04d.fits'%(tag, add_foreground, mc), hdu = [1, 2, 3])
        map_i = hp.alm2map(noise_alm, nside=nside)

    else:
        print('already exisis!')

    return map_i
            

pool = mp.Pool(processes=250)
res = pool.map(run, np.arange(500))
            
inv = 1/np.var(res, axis = 0)
np.save(inv_path, inv)

pool.close()
pool.join()
