import numpy as np
import healpy as hp

from utility import bin_cell
from fgbuster.separation_recipes import harmonic_ilc as hilc

from fgbuster import (CMB, Dust, Synchrotron, AnalyticComponent,
                      basic_comp_sep, 
                      get_observation, get_instrument)

from plancklens.utils import camb_clfile
from lenspyx import synfast
from lenspyx.utils_hp import synalm, almxfl, alm2cl
import lenspyx

import pysm3
import pysm3.units as u

from common import convert_units

import os

cls_path = '/global/homes/j/jianyao/non_gau_lensing/theory/cls/'

class simulations:
    
    def __init__(self, nside, instrument, e_n_d = False, mask = None):
        '''
        un-lensed CMB, lensed-CMB, phi, noise are saved to disk
        '''
        
        self.nside = nside
        self.instrument = instrument
        self.fres = instrument['frequency']
        self.fwhms = instrument['fwhm']
        self.end = e_n_d
        
        self.tag = instrument['experiment'][0]
        
        if mask is None:
            mask = np.ones(12*nside**2)
            
        self.mask = mask
        
        if int(sum(mask)/len(mask)) == 1:
            self.name = ""
        else:
            self.name = "_masked"
        
    def get_all(self, add_foreground = 'd0', noise = 'alms', use_phi_alm = False, index = 0, fix_phi = False):

        '''
        return one realization of (lensed CMB + foreground) with the beam applied + noise, 
        for each frequency defined in the instrument. 
        noise: get noise alms or maps. 
        e_n_d: extra noise debias, to generate many noise realizations.
        index: index labelling the realizations
        '''

        cmb_dir = '/pscratch/sd/j/jianyao/data_lensing/simulations/cmb/'
        if fix_phi:
            cmb_name = 'lensed_CMB_map_fix_lens'
        else:
            cmb_name = 'lensed_CMB_map'
            
        if not os.path.exists(cmb_dir + '%s_%04d.fits'%(cmb_name, index)):
            
            cl_unl = camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))
            geom_info = ('healpix', {'nside':self.nside}) # Geometry parametrized as above, this is the default

            lmax_unl = mmax_unl = 3*self.nside
            epsilon = 1e-6

            if use_phi_alm is False:
                assert fix_phi is False, 'fix_phi shoule be False, but instead %s'%sfix_phi
                
                cmb_temp = synfast(cl_unl, lmax=lmax_unl, verbose=0, geometry=geom_info, alm = False)
                self.cmb_len = np.row_stack((cmb_temp['T'], cmb_temp['QU'])) # (3, 12*nside**2)

            elif use_phi_alm is True:
                
                if not os.path.exists(cmb_dir + 'unlensed_CMB_map_%04d.fits'%index):
                    tlm_unl = synalm(cl_unl['tt'], lmax=lmax_unl, mmax=mmax_unl)
                    elm_unl = synalm(cl_unl['ee'], lmax=lmax_unl, mmax=mmax_unl)
                    blm_unl = synalm(cl_unl['bb'], lmax=lmax_unl, mmax=mmax_unl)

                    cmb_unlen = hp.alm2map((tlm_unl, elm_unl, blm_unl), nside = self.nside)
                    hp.write_map(cmb_dir + 'unlensed_CMB_map_%04d.fits'%index, cmb_unlen) 
                else: 
                    cmb_unlen = hp.read_map(cmb_dir + 'unlensed_CMB_map_%04d.fits'%index, field = None)
                    tlm_unl, elm_unl, blm_unl = hp.map2alm(cmb_unlen, lmax= lmax_unl, mmax = mmax_unl)
                
                phi_dir = '/pscratch/sd/j/jianyao/data_lensing/simulations/phi_alms/'
                if fix_phi:
                    if index == 0:
                        print('phi field is fixed in the following simulations')
                    phi_name = phi_dir + 'phi_%04d.fits'%0
                    if os.path.exists(phi_name):
                        plm = hp.read_alm(phi_name)

                    else:
                        plm = synalm(cl_unl['pp'], lmax=lmax_unl, mmax=mmax_unl)
                        hp.write_alm(phi_name, plm)
                
                else:
                    phi_name = phi_dir + 'phi_%04d.fits'%index
                    if os.path.exists(phi_name):
                        plm = hp.read_alm(phi_name)

                    else:
                        plm = synalm(cl_unl['pp'], lmax=lmax_unl, mmax=mmax_unl)
                        hp.write_alm(phi_name, plm)
                    
                # We then transform the lensing potential into spin-1 deflection field, and deflect the temperature map.
                dlm = almxfl(plm, np.sqrt(np.arange(lmax_unl + 1, dtype=float) * np.arange(1, lmax_unl + 2)), None, False)

                Tlen, Qlen, Ulen = lenspyx.alm2lenmap([tlm_unl, elm_unl, blm_unl], dlm, geometry=geom_info, verbose=0, epsilon=epsilon)

                self.cmb_len = np.row_stack((Tlen, Qlen, Ulen)) 
                
                hp.write_map(cmb_dir + '%s_%04d.fits'%(cmb_name, index), self.cmb_len)
                
        else:
            self.cmb_len = hp.read_map(cmb_dir + '%s_%04d.fits'%(cmb_name, index), field = None)

        Nf = self.fres.size
        cmb_maps = np.repeat(self.cmb_len[np.newaxis, :, :], Nf, axis = 0)
        
        ########## simulate noise ##############        
        if add_foreground != 'no_fore' and add_foreground != 'no_fore_full': 
            noise_dir = '/pscratch/sd/j/jianyao/data_lensing/simulations/noise/%s/'%self.tag
            if not os.path.exists(noise_dir + 'noise_freq_%04d.npy'%index):
                noise_maps = get_noise_realization(self.nside, self.instrument, unit='uK_CMB')

                np.save(noise_dir + 'noise_freq_%04d.npy'%index, noise_maps)

            else:
                noise_maps = np.load(noise_dir + 'noise_freq_%04d.npy'%index)

            if self.end is True: 
                noise_maps_end = []
                N = 20
                for n in range(N):
                    noise_map_i = get_noise_realization(self.nside, self.instrument, unit='uK_CMB')
                    noise_maps_end.append(noise_map_i)

                self.extra_noise = noise_maps_end
                self.N = N
                
        elif add_foreground == 'no_fore': 
            # if no foreground, the noise level should be at the level of residual noise of the case with foreground, after component separation 
            # we could use the noise_ilc_maps directly!!! 2024-4-4
            noise_dir = '/pscratch/sd/j/jianyao/data_lensing/simulations/cleaned_CMB/%s_MASK/'%self.tag # this case for the SO_LAT_masked case
            noise_alm = hp.read_alm(noise_dir + 'Noise_ilc_alms_from_%s_d9_HILC_lbins_42x50_lmax_2050_nside_1024_%04d.fits'%(self.tag, index), hdu = [1, 2, 3])
        
        elif add_foreground == 'no_fore_full':
            # no_fore for full-sky case
            noise_dir = '/pscratch/sd/j/jianyao/data_lensing/simulations/noise/%s/'%self.tag
            noise_maps = np.load(noise_dir + 'noise_freq_%04d.npy'%index)
               
        ######## simulate foreground ############
        
        if add_foreground != 'no_fore' and add_foreground != 'no_fore_full':
            fg_dir = '/pscratch/sd/j/jianyao/data_lensing/simulations/foreground/%s/'%self.tag

            if 'Gaussian' in add_foreground:
                fg_name = fg_dir + 'fg_%s%s_freq_%04d.npy'%(add_foreground, self.name, index)
            else:
                fg_name = fg_dir + 'fg_%s_freq_%04d.npy'%(add_foreground, index)

            if not os.path.exists(fg_name): # for now, only have one realization for foreground

                if add_foreground.startswith('d'):
                    # for pysm3 pre-defined models
                    sky = pysm3.Sky(self.nside, preset_strings = [add_foreground])
                    foreground = get_observation(self.instrument, sky = sky, nside=self.nside, noise=False, unit='uK_CMB')

                elif add_foreground.startswith('forse'):

                    pysm_model = add_foreground.split('_')[1]

                    foreground = self.rescale_forse(self.fres, pysm_model = pysm_model)

                np.save(fg_name, foreground)

            else:
                foreground = np.load(fg_name)
                
            map_all = cmb_maps + foreground            
        
            noise_alms = []
            for i in range(Nf):

                # map_all[i] = pysm3.apply_smoothing_and_coord_transform(map_all[i], fwhm=self.fwhms[i]*u.arcmin) + self.noise_maps[i]
                map_all[i] = hp.smoothing(map_all[i], fwhm = self.fwhms[i]/180/60*np.pi, lmax = 2*self.nside) + noise_maps[i]
                
                if not hasattr(self, 'fg'):
                    # print('here')
                    foreground[i] = hp.smoothing(foreground[i], fwhm = self.fwhms[i]/180/60*np.pi, lmax = 2*self.nside)

                if noise == 'alms':
                    noise_alms.append(hp.map2alm(np.where(self.mask== 0, hp.UNSEEN, noise_maps[i])))

            if noise == 'maps':
                self.noise_maps = np.where(self.mask== 0, hp.UNSEEN, noise_maps)
            elif noise == 'alms':
                self.noise_alms = noise_alms

            self.map_all = np.where(self.mask== 0, hp.UNSEEN, map_all)
            
            if not hasattr(self, 'fg'):
                self.fg = np.where(self.mask== 0, hp.UNSEEN, foreground)
                    
        elif add_foreground == 'no_fore': 
              # for the case of no foreground, we can just use one pure lensed CMB map, rather than CMB maps at different frequency; 
              # and without beam effect then coadd wth the residual noise maps from other foreground case (which already experienced 
              #  deconvoling process) 
            map_all = self.cmb_len + hp.alm2map(noise_alm, nside=self.nside)
            self.map_all = np.where(self.mask== 0, hp.UNSEEN, map_all)
            
        elif add_foreground == 'no_fore_full':
            # print(noise_maps.shape)
            self.map_all = self.cmb_len + noise_maps[4]
        
    def rescale_forse(self, fres, pysm_model = 'd0'):
        '''
        Possible problem: hp.udgrade is used for nside changes; better choice: pysm3.apply_smoothing_and_coord_transform
        '''
                
        my_dust = self.model_forse(pysm_model)
            
        if fres.size > 1:
            _observations = np.zeros((len(fres), 3, 12*self.nside**2))
            for i in range(len(fres)):
                _observations[i] = my_dust.get_emission(fres[i] * u.GHz)*convert_units('uK_RJ', 'uK_CMB', fres[i])
                
        else:
            assert fres.size == 1
            _observations = my_dust.get_emission(fres * u.GHz)*convert_units('uK_RJ', 'uK_CMB', fres)
        
        return _observations
    
    def apply_hilc(self, components, lbins):
        
        '''
        apply fgbuster.hilc method to get clean CMB map
        '''
        
        results = hilc(components, self.instrument, self.map_all, lbins) 
        ### data are deconvolved when transforming to alms, 
        ### then using harmonic_ilc_alm to do the component separation
        weighted_alms = []
        noise_alms_ilc = []
        
        for i in range(self.fres.size):
            bl = hp.gauss_beam(np.radians(self.fwhms[i]/60.0), lmax = lbins[-1], pol=True)
            # all_alms = hp.map2alm(self.map_all[i], lmax = lbins[-1])
            fg_alms = hp.map2alm(self.fg[i], lmax = lbins[-1])
            noise_alm = hp.map2alm(self.noise_maps[i], lmax = lbins[-1])
            
            for j in range(3): # T, E, B
                # hp.almxfl(all_alms[j], results.W[j, :, 0, i]/bl[:, j], inplace = True)  ### to mimic the HILC method: simulated total maps are deconvolved
                hp.almxfl(fg_alms[j], results.W[j, :, 0, i]/bl[:, j], inplace = True)
                hp.almxfl(noise_alm[j], results.W[j, :, 0, i]/bl[:, j], inplace = True)  ### simulated ilc noise maps are not deconvolved during the component separation
                
            # weighted_alms.append(all_alms)
            weighted_alms.append(fg_alms)
            noise_alms_ilc.append(noise_alm)
            
        fg_ilc_alms = np.sum(weighted_alms, axis = 0)
        # cmb_alms = np.sum(weighted_alms, axis = 0)
        noise_ilc_alms = np.sum(noise_alms_ilc, axis = 0) 
        
        if self.end is True:
            nls = []
            for n in range(self.N):
                _noise_alms_ilc = []

                for i in range(self.fres.size):
                    bl = hp.gauss_beam(np.radians(self.fwhms[i]/60.0), lmax = lbins[-1], pol=True)

                    _noise_alm_i = hp.map2alm(self.extra_noise[n][i], lmax = lbins[-1])

                    for j in range(3):
                        hp.almxfl(_noise_alm_i[j], results.W[j, :, 0, i]/bl[:, j], inplace = True)

                    _noise_alms_ilc.append(_noise_alm_i)

                _noise_alms = np.sum(_noise_alms_ilc, axis = 0) 
                nls.append(hp.alm2cl(_noise_alms))

            return results, noise_ilc_alms, fg_ilc_alms, nls
        else:

            return results, noise_ilc_alms, fg_ilc_alms
    
    def model_forse(self, pysm_model):
        
        dust_dir = "/pscratch/sd/j/jianyao/data_lensing/processed_dust_maps/"
        if pysm_model == 'd0':
            my_dust = pysm3.ModifiedBlackBody(
                nside = self.nside,
                map_I = "pysm_2/dust_t_new.fits",
                map_Q = dust_dir + "forse_dust_Q_353GHz_deconvolved_lmax_4096_nside4096_uK_RJ.fits", #"forse_dust_Q_353GHz_3amin_nside4096_uK_RJ.fits",
                map_U = dust_dir + "forse_dust_U_353GHz_deconvolved_lmax_4096_nside4096_uK_RJ.fits",
                unit_I = "uK_RJ",
                unit_Q = "uK_RJ",
                unit_U = "uK_RJ",
                map_mbb_index = 1.54,
                map_mbb_temperature = 20,
                unit_mbb_temperature = "K",
                freq_ref_I = "545 GHz",
                freq_ref_P = "353 GHz"
            )
            
        if pysm_model == 'd1':
            my_dust = pysm3.ModifiedBlackBody(
                nside = self.nside,
                map_I = "pysm_2/dust_t_new.fits",
                map_Q = dust_dir + "forse_dust_Q_353GHz_deconvolved_lmax_4096_nside4096_uK_RJ.fits",
                map_U = dust_dir + "forse_dust_U_353GHz_deconvolved_lmax_4096_nside4096_uK_RJ.fits",
                unit_I = "uK_RJ",
                unit_Q = "uK_RJ",
                unit_U = "uK_RJ",
                map_mbb_index = "pysm_2/dust_beta.fits",
                map_mbb_temperature = "pysm_2/dust_temp.fits",
                unit_mbb_temperature = "K",
                freq_ref_I = "545 GHz",
                freq_ref_P = "353 GHz"
            )
            
        if pysm_model == 'd9':
            my_dust = pysm3.ModifiedBlackBody(
                nside = self.nside,
                map_I = "pysm_2/dust_t_new.fits",
                map_Q = dust_dir + "forse_dust_Q_353GHz_deconvolved_lmax_4096_nside4096_uK_RJ.fits",
                map_U = dust_dir + "forse_dust_U_353GHz_deconvolved_lmax_4096_nside4096_uK_RJ.fits",
                unit_I = "uK_RJ",
                unit_Q = "uK_RJ",
                unit_U = "uK_RJ",
                map_mbb_index = 1.48,
                map_mbb_temperature = 19.6,
                unit_mbb_temperature = "K",
                freq_ref_I = "353 GHz",
                freq_ref_P = "353 GHz"
            )

        if pysm_model == 'd10':
            my_dust = pysm3.ModifiedBlackBody(
                nside = self.nside,
                map_I = "pysm_2/dust_t_new.fits",
                map_Q = dust_dir + "forse_dust_Q_353GHz_deconvolved_lmax_4096_nside4096_uK_RJ.fits",
                map_U = dust_dir + "forse_dust_U_353GHz_deconvolved_lmax_4096_nside4096_uK_RJ.fits",
                unit_I = "uK_RJ",
                unit_Q = "uK_RJ",
                unit_U = "uK_RJ",
                map_mbb_index = "dust_gnilc/gnilc_dust_beta_nside{nside}_2023.06.06.fits",
                map_mbb_temperature = "dust_gnilc/gnilc_dust_Td_nside{nside}_2023.06.06.fits",
                unit_mbb_temperature = "K",
                freq_ref_I = "353 GHz",
                freq_ref_P = "353 GHz"
            )

        if pysm_model == 'Gaussiand0':
            # print('Gaussian d0 model!')
            my_dust = pysm3.ModifiedBlackBody(
                nside = self.nside,
                map_I = "pysm_2/dust_t_new.fits",
                map_Q = dust_dir + "Gaussian_forse_dust_Q_%s%s_353GHz_deconvolved_lmax_4096_nside4096_uK_RJ.fits"%(self.tag, self.name),
                map_U = dust_dir + "Gaussian_forse_dust_U_%s%s_353GHz_deconvolved_lmax_4096_nside4096_uK_RJ.fits"%(self.tag, self.name),
                unit_I = "uK_RJ",
                unit_Q = "uK_RJ",
                unit_U = "uK_RJ",
                map_mbb_index = 1.54,
                map_mbb_temperature = 20,
                unit_mbb_temperature = "K",
                freq_ref_I = "545 GHz",
                freq_ref_P = "353 GHz"
            )
            
        if pysm_model == 'Gaussiand1':
            # print('Gaussian d1 model!')
            my_dust = pysm3.ModifiedBlackBody(
                nside = self.nside,
                map_I = "pysm_2/dust_t_new.fits",
                map_Q = dust_dir + "Gaussian_forse_dust_Q_%s%s_353GHz_deconvolved_lmax_4096_nside4096_uK_RJ.fits"%(self.tag, self.name),
                map_U = dust_dir + "Gaussian_forse_dust_U_%s%s_353GHz_deconvolved_lmax_4096_nside4096_uK_RJ.fits"%(self.tag, self.name),
                unit_I = "uK_RJ",
                unit_Q = "uK_RJ",
                unit_U = "uK_RJ",
                map_mbb_index = "pysm_2/dust_beta.fits",
                map_mbb_temperature = "pysm_2/dust_temp.fits",
                unit_mbb_temperature = "K",
                freq_ref_I = "545 GHz",
                freq_ref_P = "353 GHz"
            )

        if pysm_model == 'Gaussiand9':
            my_dust = pysm3.ModifiedBlackBody(
                nside = self.nside,
                map_I = "pysm_2/dust_t_new.fits",
                map_Q = dust_dir + "Gaussian_forse_dust_Q_%s%s_353GHz_deconvolved_lmax_4096_nside4096_uK_RJ.fits"%(self.tag, self.name),
                map_U = dust_dir + "Gaussian_forse_dust_U_%s%s_353GHz_deconvolved_lmax_4096_nside4096_uK_RJ.fits"%(self.tag, self.name),
                unit_I = "uK_RJ",
                unit_Q = "uK_RJ",
                unit_U = "uK_RJ",
                map_mbb_index = 1.48,
                map_mbb_temperature = 19.6,
                unit_mbb_temperature = "K",
                freq_ref_I = "353 GHz",
                freq_ref_P = "353 GHz"
            )
            
        if pysm_model == 'Gaussiand10':
            # print('Gaussian d10 model!')
            my_dust = pysm3.ModifiedBlackBody(
                nside = self.nside,
                map_I = "pysm_2/dust_t_new.fits",
                map_Q = dust_dir + "Gaussian_forse_dust_Q_%s%s_353GHz_deconvolved_lmax_4096_nside4096_uK_RJ.fits"%(self.tag, self.name),
                map_U = dust_dir + "Gaussian_forse_dust_U_%s%s_353GHz_deconvolved_lmax_4096_nside4096_uK_RJ.fits"%(self.tag, self.name),
                unit_I = "uK_RJ",
                unit_Q = "uK_RJ",
                unit_U = "uK_RJ",
                map_mbb_index = "dust_gnilc/gnilc_dust_beta_nside{nside}_2023.06.06.fits",
                map_mbb_temperature = "dust_gnilc/gnilc_dust_Td_nside{nside}_2023.06.06.fits",
                unit_mbb_temperature = "K",
                freq_ref_I = "353 GHz",
                freq_ref_P = "353 GHz"
            )
            
        return my_dust