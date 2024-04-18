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

cls_path = '/global/homes/j/jianyao/non_gau_lensing/theory/cls/'
w = lambda ell : ell ** 2 * (ell + 1.) ** 2 * 0.5 / np.pi * 1e7

cl_unl = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))
cl_len = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))
#: Fiducial unlensed and lensed power spectra used for the analysis.

cl_weight = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))
cl_weight['bb'] *= 0.
#: CMB spectra entering the QE weights (the spectra multplying the inverse-variance filtered maps in the QE legs) 

def run_qe(qe_keys, cases, nside = 1024, lmin_ivf = 100, lmax_ivf = 2000, lmax_qlm = 1000, nsim = 1, lmax_Bmode = None, from_fg_res = False, which_fg = None, 
          experiment = 'SO_LAT'):
    '''
    2024/4/4: now no_fore case will be compatible with other foreground cases, since no_fore will have the same sigma with 'PySM_d9' case. 
    Parameters
    ----------
    
    qe_keys: ['p_eb']; one key for now; 20240112
    lmax_Bmode: used to set fbl[lmax_Bmode:] = 0 to avoid the foreground residual in the B modes. for the estimator key p_p;
    return_results: Bool
                    return all the results inside a list
                    

    old_dir: If True, dir will be named without lmin and lmax;
    nsim: return many realizations of qlm if lager than 1, which can be used to get errors.
    
    mf: cached mean field; can be mean field from other case.
    from_fg_res: test the lensing estimator response from fg_res
    which_fg: Bool; If True, use the bias from another kind of dust 
    compare_mf: Bool; In the case of fg_res, if True, also return the mean-field from 300 realizations
    experiment: 'SO_LAT'/'S4_LAT' which will in fact be 'SO_LAT_MASK'/'S4_LAT_MASK'
    
    
    Returns
    -------
    
    returns are not cleaned with subtracting mean field, not normalized, nor with w(ell) weighted. But all the relevent numbers are returned
    '''
    
    # x_eb and p_eb results are saved simultaneously at the same time
    qe_keys_dir = [qe_key.replace('x_', 'p_') if qe_key.startswith('x_') else qe_key for qe_key in qe_keys]
    
    transf = np.ones(lmax_ivf + 1)
    nlev_t = 0; #2.16
    nlev_p = 0; #2.16
    
    if lmax_Bmode is None:
        lmax_Bmode = lmax_ivf

    results_phi = []
    results_n0 = []
    results_mf = []
    ells_binned = []
        
    results_qlms = []
    results_qnorms = []
    
    results_phi_fg = []
    results_n0_fg = []
    results_qnorms_fg = []
    
    mask = '/pscratch/sd/j/jianyao/data_lensing/%s_mask_1024.fits'%experiment
    
    for add_foreground in cases:

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

        sims = cmb_noise(cmb_len(add_foreground=add_foreground), transf, nlev_t, nlev_p, nside)

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
        
        # for fg_res
        if from_fg_res:
            
            if which_fg is not None:
                TEMP_fg = TEMP + '/fg_%s_res'%which_fg  
                sims_fg = cmb_noise(cmb_len(add_foreground=which_fg, from_fg_res=from_fg_res), transf, nlev_t, nlev_p, nside)
            else:
                TEMP_fg = TEMP + '/fg_res'
                sims_fg = cmb_noise(cmb_len(add_foreground=add_foreground, from_fg_res=from_fg_res), transf, nlev_t, nlev_p, nside)
                
            fl_EE = hp.alm2cl(sims_fg.sims_cmb_len.get_sim_elm(0))
            fl_BB = hp.alm2cl(sims_fg.sims_cmb_len.get_sim_blm(0))
            # fel_fg = utils.cli(fl_EE[:lmax_ivf + 1]); fel_fg[:lmin_ivf] *= 0.
            # fbl_fg = utils.cli(fl_BB[:lmax_ivf + 1]); fbl_fg[:lmin_ivf] *= 0.
            # ftl_fg = np.ones_like(fel_fg);            ftl_fg[:lmin_ivf] *= 0.
            fl_weight = {'tt':np.ones_like(fl_EE), 'ee':fl_EE, 'bb':fl_BB, 'te':np.ones_like(fl_EE)}
            
#             if lmax_Bmode != lmax_ivf:
#                 fbl_fg[lmax_Bmode:] *= 0.
                
            ftl_fg = ftl
            fel_fg = fel
            fbl_fg = fbl
            fl_len = cl_len
            # fl_weight = cl_weight #N0 bias for fg_res should consider Fl,  which are included in ivfs_fg
            
            # ivfs_fg = filt_simple.library_fullsky_sepTP(os.path.join(TEMP_fg, 'ivfs'), sims_fg, nside, transf, fl_len, ftl_fg, fel_fg, fbl_fg, cache=True)

            ivfs_fg = filt_simple.library_apo_sepTP(os.path.join(TEMP_fg, 'ivfs'), sims_fg, mask, fl_len, transf, ftl_fg, fel_fg, fbl_fg, cache=True)
        
            qlms_dd_fg = qest.library_sepTP(os.path.join(TEMP_fg, 'qlms_dd'), ivfs_fg, ivfs_fg, fl_len['te'], nside, lmax_qlm=lmax_qlm)
            nhl_dd_fg = nhl.nhl_lib_simple(os.path.join(TEMP_fg, 'nhl_dd'), ivfs_fg, fl_weight, lmax_qlm)
        
        #---- N1 lensing bias library:
        # libdir_n1_dd = os.path.join(TEMP, 'n1_test')
        # n1_dd = n1.library_n1(libdir_n1_dd,cl_len['tt'],cl_len['te'],cl_len['ee'])

        #---- QE response calculation library:
        qresp_dd = qresp.resp_lib_simple(os.path.join(TEMP, 'qresp'), lmax_ivf, cl_weight, cl_len,
                                         {'t': ivfs.get_ftl(), 'e':ivfs.get_fel(), 'b':ivfs.get_fbl()}, lmax_qlm)
        
        if isinstance(nsim, int):
            nsims = range(nsim)
        elif isinstance(nsim, list):
            nsims = range(nsim[0], nsim[1])
        for mc in tqdm(nsims):

            for qe_key in qe_keys:
                
                qlm = qlms_dd.get_sim_qlm(qe_key, mc)                                     
                qlms.append(qlm)
                
                if from_fg_res:
                    
                    qlm_fg = qlms_dd_fg.get_sim_qlm(qe_key, mc)
                    qlms_fg.append(qlm_fg)
                    
                    # if mc == nsims[0]:
                    nhl_data_fg = nhl_dd_fg.get_sim_nhl(mc, qe_key, qe_key)
                    nhl_datas_fg.append(nhl_data_fg)                

                if mc == nsims[0]:
                    # Lensing response according to the fiducial cosmology:
                    qresp_dat = qresp_dd.get_response(qe_key, 'p')
                    # Estimator normalization is the inverse response:
                    qnorm = utils.cli(qresp_dat)
                    qnorms.append(qnorm)   
                    
                    nhl_data = nhl_dd.get_sim_nhl(mc, qe_key, qe_key)
                    nhl_datas.append(nhl_data)
                    
        ell = np.arange(2 if qe_key[0] == 'x' else 2, lmax_qlm) # qnorms has very large number for \ell = 1; 2023/12/26
        
        if from_fg_res:
            
            # qresp_dd_fg = qresp.resp_lib_simple(os.path.join(TEMP_fg, 'qresp'), lmax_ivf, fl_len, fl_len, 
            #                                     {'t': ivfs_fg.get_ftl(), 'e':ivfs_fg.get_fel(), 'b':ivfs_fg.get_fbl()}, lmax_qlm)
            # qresp_dat_fg = qresp_dd_fg.get_response(qe_key[0], 'p')
            # qnorm_fg = utils.cli(qresp_dat_fg)
            
            qnorm_fg = qnorm
            print('using qnorm_cmb as qnorm_fg')
            
            # n0_fg = nhl_data_fg[ell] * qnorm_fg[ell] ** 2 * w(ell)
            # cleaned_fg = (hp.alm2cl(qlm_fg)[ell] - nhl_data_fg[ell])* qnorm_fg[ell] ** 2 / qlms_dd_fg.fsky12 * w(ell)
            
            results_n0_fg.append(nhl_datas_fg)
            results_phi_fg.append(qlms_fg)
            results_qnorms_fg.append(qnorm_fg)
            
        results_phi.append(qlms)
        results_n0.append(nhl_datas)
        ells_binned.append(ell)
        results_qnorms.append(qnorm)
        
        if mc >= 199:
            dir_mf = TEMP+'/mean_field_%s'%qe_keys[0] #once mf are saved, the 0:200 realizations can be left alone, when estimating errors
            if not os.path.exists(dir_mf):
                os.makedirs(dir_mf)
                hp.write_alm(dir_mf + '/mf_200_%s.fits'%(qe_keys[0]), np.mean(qlms[0:200], axis = 0))
                hp.write_alm(dir_mf + '/mf_100_%s.fits'%(qe_keys[0]), np.mean(qlms[0:100], axis = 0))
            
    if from_fg_res:
        
        return ells_binned, results_phi, results_n0, results_qnorms, qlms_dd.fsky12, results_phi_fg, results_n0_fg, results_qnorms_fg
    
    return ells_binned, results_phi, results_n0, results_qnorms, qlms_dd.fsky12
    
    
    
class cmb_len(object):

    def __init__(self, add_foreground, from_fg_res = False, experiment = 'SO_LAT_MASK'):
        
        if add_foreground != 'no_fore':

            dir_cleaned_cmb = '/pscratch/sd/j/jianyao/data_lensing/simulations/cleaned_CMB/%s_MASK/'%experiment # SO LAT
            
            # get an estimation of lensing field from foreground residuals: to test the estimator response to the fg_res
            if from_fg_res: 
                print('load fg_res')
                self.alms = dir_cleaned_cmb + 'FG_ilc_alms_from_SO_LAT_%s_HILC_lbins_42x50_lmax_2050_nside_1024'%(add_foreground)
                
            else:
                self.alms = dir_cleaned_cmb + 'CMB_alms_from_SO_LAT_%s_HILC_lbins_42x50_lmax_2050_nside_1024'%(add_foreground)
        
        elif add_foreground == 'no_fore':
            
            dir_cleaned_cmb = '/pscratch/sd/j/jianyao/data_lensing/simulations/cmb_noise_only/%s_MASK/'%experiment
            self.alms = dir_cleaned_cmb + 'CMB_noise_alms_from_%s_masked_%s_nside_1024'%(experiment, add_foreground)
        # self.dirs = os.environ["data"] + 'cmb_plus_%s'%self.add_foreground
        
    def hashdict(self):
        return {'sim_lib': 'ffp10 lensed scalar cmb inputs, freq 0'}

    # @staticmethod
    def get_sim_tlm(self, idx):
        
        # return hp.read_alm(opj(self.dirs, 'lensed_scl_cmb_plus_%s_alm_mc_%04d.fits'%(self.add_foreground, idx)), hdu=1)
        
        alms = self.alms + '_%04d.fits'%idx
        # print(alms)
        return hp.read_alm(alms, hdu = 1)
    
    # @staticmethod
    def get_sim_elm(self, idx):

        # return hp.read_alm(opj(self.dirs, 'lensed_scl_cmb_plus_%s_alm_mc_%04d.fits'%(self.add_foreground, idx)), hdu=2)
        
        alms = self.alms + '_%04d.fits'%idx
        # print(alms)
        return hp.read_alm(alms, hdu = 2)

    # @staticmethod
    def get_sim_blm(self, idx):

        # return hp.read_alm(opj(self.dirs, 'lensed_scl_cmb_plus_%s_alm_mc_%04d.fits'%(self.add_foreground, idx)), hdu=3)
        
        alms = self.alms + '_%04d.fits'%idx
        # print(alms)
        return hp.read_alm(alms, hdu = 3)
    
    
class cmb_noise(object):
    r"""CMB simulation library combining a lensed CMB library and a transfer function.

        Args:
            sims_cmb_len: lensed CMB library (e.g. *plancklens.sims.planck2018_sims.cmb_len_ffp10*)
            cl_transf: CMB temperature transfer function
            nside: healpy resolution of the maps. Defaults to 2048.
            lib_dir(optional): hash checks will be cached, as well as possibly other things for subclasses.
            cl_transf_P: CMB pol transfer function (if different from cl_transf)\
            eff_beam: fwhm; None or a number; if a number, apply the beam of this level to the input cleaned CMB map

    """
    def __init__(self, sims_cmb_len, cl_transf, nlev_t, nlev_p, nside=2048, cl_transf_P=None, lib_dir=None):
        if cl_transf_P is None:
            cl_transf_P = np.copy(cl_transf)

        self.sims_cmb_len = sims_cmb_len
        self.cl_transf_T = cl_transf
        self.cl_transf_P = cl_transf_P
        self.nside = nside
        self.nlev_t = nlev_t
        self.nlev_p = nlev_p
        self.vamin = np.sqrt(hp.nside2pixarea(self.nside, degrees=True)) * 60

        if lib_dir is not None:
            fn_hash = os.path.join(lib_dir, 'sim_hash.pk')
            if mpi.rank == 0 and not os.path.exists(fn_hash):
                pk.dump(self.hashdict(), open(fn_hash, 'wb'), protocol=2)
            mpi.barrier()
            hash_check(self.hashdict(), pk.load(open(fn_hash, 'rb')), fn=fn_hash)

    def hashdict(self):
        ret = {'sims_cmb_len':self.sims_cmb_len.hashdict(),'nside':self.nside,'cl_transf':clhash(self.cl_transf_T)}
        if not (np.all(self.cl_transf_P == self.cl_transf_T)):
            ret['cl_transf_P'] = clhash(self.cl_transf_P)
        return ret

    def get_sim_tmap(self,idx):
        """Returns temperature healpy map for a simulation

            Args:
                idx: simulation index

            Returns:
                healpy map

        """
        tlm = self.sims_cmb_len.get_sim_tlm(idx)
        # hp.almxfl(tlm,self.cl_transf_T,inplace=True)
        tmap = hp.alm2map(tlm,self.nside)
        return tmap + self.get_sim_tnoise(idx)

    def get_sim_pmap(self,idx):
        """Returns polarization healpy maps for a simulation

            Args:
                idx: simulation index

            Returns:
                Q and U healpy maps

        """
        elm = self.sims_cmb_len.get_sim_elm(idx)
        blm = self.sims_cmb_len.get_sim_blm(idx)
        
        if self.cl_transf_T[-1] != 1:
            # in the case of apply the effective beam to the input cleaned CMB map
            hp.almxfl(elm,self.cl_transf_P,inplace=True)
            hp.almxfl(blm, self.cl_transf_P, inplace=True)
        Q,U = hp.alm2map_spin([elm,blm], self.nside, 2,hp.Alm.getlmax(elm.size))
        del elm,blm
        return Q + self.get_sim_qnoise(idx),U + self.get_sim_unoise(idx)

    def get_sim_tnoise(self,idx):
        '''
        there are already noise in the clean CMB maps
        '''
        return 0 #self.nlev_t / self.vamin * np.load(opj(os.environ["data"],'noise/noise_1uK_amin_map_mc_%04d.npy'%idx))[0] # one frequency for now 11.15

    def get_sim_qnoise(self, idx):
        
        return 0 #self.nlev_p / self.vamin * np.load(opj(os.environ["data"],'noise/noise_1uK_amin_map_mc_%04d.npy'%idx))[1]

    def get_sim_unoise(self, idx):
        
        return 0 #self.nlev_p / self.vamin * np.load(opj(os.environ["data"],'noise/noise_1uK_amin_map_mc_%04d.npy'%idx))[2]