import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from plancklens import utils
import os

from binning import binning, multipole_binning

cls_path = '/global/homes/j/jianyao/non_gau_lensing/theory/cls/'
w = lambda ell : ell ** 2 * (ell + 1.) ** 2 * 0.5 / np.pi * 1e7
# w = lambda ell : ell ** 2 * (ell + 1.) ** 2*0.25

cl_unl = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))
cl_len = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))
#: Fiducial unlensed and lensed power spectra used for the analysis.

cl_weight = utils.camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))
cl_weight['bb'] *= 0.
#: CMB spectra entering the QE weights (the spectra multplying the inverse-variance filtered maps in the QE legs) 

def Ell(ell_list):
    ell = np.array(ell_list)
    return ell*(ell+1)/2/np.pi

def bin_cell(cls, lmax, bins, ell_2 = True):
    '''
    cls: list; cls including several XX or for only one XX, X means T, E, B
    bins: an integer or a list; if list, determines the edges of the bins
    lmax:
    
    Return
    ------
    ells_bin: binned ells, don't include multipoles beyond the last bin_edge defined by N*bin_width
    cls_bin: binned cell
    ell_2: Bool; If True, factor ell*(ell+1)/2/np.pi are applied to the output binned cell.
    
    '''
    
    if isinstance(bins, list):
        use_edge = True
        N = len(bins) - 1
        
    else:
        use_edge = False
        N = lmax//bins
    
    if len(cls) != 6:
        cls = [cls]
        
    cls_bin = []
    for s in range(len(cls)):
        cls_bin_i = np.zeros(N)
        
        if s == 0:
            ells = np.arange(len(cls[0]))
            ells_bin = []
        
        for i in range(N):
            
            if use_edge:
                if s == 0:
                    ells_bin.append(np.mean(ells[bins[i]:bins[i+1]]))
                    
                cls_bin_i[i] = np.mean(cls[s][bins[i]:bins[i+1]])    
                
            else:
                if s == 0:
                    ells_bin.append(np.mean(ells[i*bins:(i+1)*bins]))
                    
                cls_bin_i[i] = np.mean(cls[s][i*bins:(i+1)*bins])
                
        if ell_2:
            cls_bin_i *= Ell(ells_bin)
        
        cls_bin.append(cls_bin_i)
    
    return ells_bin, cls_bin

def plot_lensing(lmax_qlm, results, return_binned = False, qe_key = 'p_p', add_foreground = 'd9', conf = 'lmin = 100', from_fg_res = False):
    
    '''
    To show the result of one realization of reconstructed phi, after subtracting the mean field.
    lmax_qlm
    results: output from `run_qe`
    ells, results_phi, results_mf, results_n0

    # results[i][j][k]: j: cases; k: nsim
    # qlms: [i][j]: j: cases, for nsim together
    # qnorm: [i][j]: j: cases, each nsim is the same
    # nhl_data: each sim is the same
    '''
    
    ell, qlms, nhl_data, qnorm, fsky = results[0][0], results[1][0], results[2][0][0], results[3][0], results[4]
    
    if len(qlms) > 1:
        qlms_mf_300 = np.mean(qlms[1:200], axis = 0)
    else:
        qlms_mf_300 = np.zeros_like(qlms[0])
    
    weights_phi = qnorm[ell] ** 2 / fsky * w(ell)
    n0 = nhl_data[ell] * qnorm[ell] ** 2 * w(ell)
    
    mf = hp.alm2cl(qlms_mf_300)[ell] * weights_phi
    
    returned = (hp.alm2cl(qlms[0])[ell])* weights_phi - n0
    cleaned_mf_alm = (hp.alm2cl(qlms[0] - qlms_mf_300)[ell])* weights_phi - n0
    
    bw = 10
    ell_binned, n0_binned =  bin_cell(n0, lmax = lmax_qlm, bins=bw, ell_2=False)
    _, returned_binned =  bin_cell(returned, lmax = lmax_qlm, bins=bw, ell_2=False)
    _, mf_binned =  bin_cell(mf, lmax = lmax_qlm, bins=bw, ell_2=False)
    
    _, cleaned_mf_alm_binned = bin_cell(cleaned_mf_alm, lmax = lmax_qlm, bins=bw, ell_2=False)
    
    _, cl_pp = bin_cell(cl_unl['pp'][ell] *  w(ell), lmax = lmax_qlm, bins=bw, ell_2=False)

    if from_fg_res:
        qlms_fg = results[5][0] # also nsim together
        nhl_data_fg = results[6][0][0]
        
        n0_fg = nhl_data_fg[ell] * qnorm[ell] ** 2 * w(ell)
        returned_fg = (hp.alm2cl(qlms_fg[0])[ell])* weights_phi - n0_fg       
        
        _, returned_fg_binned = bin_cell(returned_fg, lmax = lmax_qlm, bins=bw, ell_2=False)
        _, n0_fg_binned =  bin_cell(n0_fg, lmax = lmax_qlm, bins=bw, ell_2=False)

        
    # make fig, axes = plt.subplots(1,2,figsize = (15, 6))
    plt.figure(figsize = (15, 6))
    # plt.title('%s for %s, lmin = %s, lmax = %s'%(qe_key, add_foreground, lmin_qlm, lmax_qlm))
    
    # label=r'$C_L^{\hat \phi \hat \phi}$' if qe_key[0] == 'p' else r'$C_L^{\hat \omega \hat \omega}$'
    plt.subplot(121)
    plt.title('%s for %s'%(qe_key, conf))
    plt.loglog(ell_binned, cl_pp[0], c='k', label=r'$C_L^{\phi\phi, \rm fid}$')
    plt.loglog(ell_binned, n0_binned[0], '--', label=r'$\hat N_L^{(0)}$')

    plt.loglog(ell_binned, mf_binned[0], label = 'mean_field')
    plt.loglog(ell_binned, returned_binned[0], label = r'%s $- N_L^{(0)}$'%add_foreground)

    if from_fg_res:
        
        plt.loglog(ell_binned, returned_fg_binned[0], label = r'%s fg res - $N_{L, fg}^{(0)}$'%add_foreground, ls = '--')
        plt.text(50, 1e-2, r'$N^{0}_{fg res} = %.1e$'%np.mean(n0_fg_binned[0]))
        # plt.loglog(ell_binned, n0_fg_binned[0], '--', label=r'$\hat N_{L, fg}^{(0)}$')
        
    plt.xlabel('$L$', fontsize=12)
    plt.ylabel('$L^2 (L + 1)^2 C_L^{\phi\phi}$  [$x10^7$]', fontsize=12)
    plt.ylim(1e-3,4e1)
    plt.xlim(3, )
    plt.legend(fontsize=12, loc='lower left', frameon = False)
    
    plt.subplot(122)
    plt.loglog(ell_binned, cl_pp[0], c='k', label=r'$C_L^{\phi\phi, \rm fid}$')
    
    if from_fg_res:
        plt.loglog(ell_binned, returned_binned[0] - returned_fg_binned[0], 'r--', label = 'Left_Green - pp_fg_res')
    plt.loglog(ell_binned, returned_binned[0] - mf_binned[0], '--', label = 'Left_Green - cl_mf', color = sns.color_palette()[1])
    
    plt.loglog(ell_binned, cleaned_mf_alm_binned[0], '--', label = 'Left_Green - alm_mf', color = sns.color_palette()[2])
        
    plt.ylim(1e-3,4e1)
    plt.legend(fontsize=12, loc='lower left', frameon = False)
    
def get_SNR(ells, cl_pp_input, n0_input, lmax):
    
    n0 = n0_input/w(ells) # input n0 is multiplied with the L^4 factor
    
    cl_pp = cl_pp_input[ells]
    snr2 = (cl_pp/(cl_pp + n0))**2*(ells + 0.5)
    # plt.loglog(snr2)
    return np.sqrt(sum(snr2[:lmax-2]))

def plot_snr(ells, cl_pp_input, results, labels):

    cl_pp = cl_pp_input[ells]
   
    lmaxs = np.arange(600, 1000, 20)
    
    for i in range(len(labels)):
        snrs = []
        n0 =  results[i]/w(ells)
        snr2 = (cl_pp/(cl_pp + n0))**2*(ells + 0.5)
        
        for lmax in lmaxs:
            snr_sum = np.sqrt(sum(snr2[:lmax-2]))
            
            snrs.append(snr_sum)
    
        plt.plot(lmaxs, snrs, label = labels[i])
    
    plt.legend()
    
dicts = {'no_fore':'No', 'd9':'P_d9', 'forse3_Gaussiand9':'G_d9', 'forse3_d9':'F_d9', 'no_fore_full':'No_full'} 

def plot_errors_from_mf(cases, results_all, lmaxBmode, qe_key = 'p_eb', experiment = 'SO_LAT_MASK', subtract_mf = 'cl', N_mf = 200, plot_mf_cl = False, 
                        from_fg_res = False, add_foreground = None, return_error = False, binning = ['planck', None], true_binned = None, debias_n1 = False, debias_mcmf = False, xlim = 1000):
    '''
    results_all: output from run_qe function for foreground cases, assumed the mean field is already calculated. 
    set nsim = [200, 500] for run_qe.
    plot_mf_cl only works for subtract_mf = cl case.
    add_foreground: **Mismatch** effect. set this to a foreground case to use this mf to be the mf in the results_all.
    N_mf: number of realizations to estimate the mean field; can be 100 or 200;
    '''
   
    if qe_key.startswith('x'):
        _plot_variance = plot_variance_x
    else:
        _plot_variance = plot_variance # o.w. python will set plot_variance as local variable and the global 'plot_variance' defined below shoule not be used.
    
    if experiment[-4:] == 'full':
        lenre_dir = '/pscratch/sd/j/jianyao/data_lensing/lenre_results/%s/'%experiment
        
    else:
        lenre_dir = '/pscratch/sd/j/jianyao/data_lensing/lenre_results/%s_cinv/'%experiment
    
    if add_foreground:
        TEMP = lenre_dir + 'cleaned_cmb_%s_lmin_%s_lmax_%s_%s_lmax_Bmode_%s'%(add_foreground, 100, 2000, 'p_eb', lmaxBmode)
    
    print('plot from dir: %s'%lenre_dir)
    results_all = list(results_all)
    fsky = results_all.pop(5)

    ell = results_all[0][0]
    data_errors = []; mf_cls = []; n0s = []; n1s = [] # for different fore cases
    labels = [dicts[case] for case in cases]

    case_i = 0
    data_errors = []
    for fore in list(zip(*results_all)):
        if not add_foreground:
            TEMP =  lenre_dir + 'cleaned_cmb_%s_lmin_%s_lmax_%s_%s_lmax_Bmode_%s'%(cases[case_i], 100, 2000, 'p_eb', lmaxBmode)
        
        error_mf = get_variance(fore, fsky, subtract_mf = subtract_mf, mf= TEMP+'/mean_field_%s/mf_%s_%s.fits'%(qe_key, N_mf, qe_key), Nsim_mf = 0, return_mf_cl = plot_mf_cl, binning = binning, debias_n1 = debias_n1, debias_mcmf = debias_mcmf)
        
        if plot_mf_cl:
            data_error, mf_cl, n0, n1, mc_mf = error_mf 
            mf_cls.append(mf_cl)
        else:
            data_error, n0, n1, mc_mf = error_mf
            
        data_errors.append(data_error)
        n0s.append(n0); n1s.append(n1)
        case_i += 1
    
    if not debias_n1:
        n1 = None
    if not debias_mcmf:
        mc_mf = None
    
    if plot_mf_cl:
        mf_data = 'mean_filed_lmin_%s_lmax_%s_%s_lmax_Bmode_%s.dat'%(100, 2000, qe_key, lmaxBmode)
        if not os.path.exists(mf_data):
            with open(mf_data, 'w') as f:
                for s in cases:
                    f.write(s + ' ')
                f.write('\n')
                for i in range(len(mf_cls[0])):
                    f.write(str(ell[i]) + ' ' + str(mf_cls[0][i]) + ' ' + str(mf_cls[1][i]) + ' ' + str(mf_cls[2][i]) + ' ' + str(mf_cls[3][i]) + '\n')
                
        _plot_variance(ell, data_errors, labels = labels, kind = 'bar', title = '%s, Subtracting MF at %s, lmaxBmode = %s'%(qe_key, subtract_mf, lmaxBmode), mf_cls = mf_cls, n0s = n0s, n1s = n1s, mc_mf = mc_mf, true_binned = true_binned, xlim = xlim)

    else:
        _plot_variance(ell, data_errors, labels = labels, kind = 'bar', title = '%s, Subtracting MF at %s, lmaxBmode = %s'%(qe_key, subtract_mf, lmaxBmode), n0s = n0s, n1s = n1s, mc_mf = mc_mf, true_binned = true_binned, xlim = xlim)
        
    if from_fg_res:
        data_errors = []
        for fore in list(zip(*results_all)):
            data_errors.append(get_variance(fore, fsky, Nsim_mf = 0, from_fg_res = True, binning = binning))
        _plot_variance(ell, data_errors, labels = labels, kind = 'bar', title = '%s, Subtracting FG res bias, lmaxBmode = %s'%(qe_key, lmaxBmode), true_binned = true_binned)

        data_errors_fg_res = []
        for fore in list(zip(*results_all)):
            data_errors_fg_res.append(get_variance_fg_res(fore, binning = binning), fsky)
        _plot_variance(ell, data_errors_fg_res, labels = labels, kind = 'bar', title = '%s, Cl_phiphi from fg_res, lmaxBmode = %s'%(qe_key, lmaxBmode), set_ylim=False, true_binned = true_binned)
        
    if return_error:
        return data_error


def get_variance(results, fsky, subtract_mf = None, mf = None, Nsim_mf = None, return_mf_cl = False, 
                 binning = ['my',None, None], from_fg_res = False, debias_n1 = False, debias_mcmf = False):
    '''
    Get errors for ONE foreground case
    
    Parameters
    ----------
    
    results: unbinned results from run_qe function 
    
    subtract_mf: 'alm' or 'cl' or None, whether subtracting the mean field or not; If not None, subtracting at the alm space or cl space
    mf: str, 0, or None; If str, input mean_field_alms will be loaded and to be subtracted;
    Nsim_mf: int or None, number of realizations in the resutls used to estimate mean field ; If int, the former Nsim_mf will be used ([0:Nsim_mf])
    binning: my simple binning scheme or planck binning scheme; 'my' or 'planck'. ['my', bin_cell, bin_edge] or ['planck', plk_binner, bin_edge], or ['toshiyan', binning, binner]; plk_binner is an instance of ffp10_binner; 2024/4/14; NOT implemented for fg_res yet
    
    '''
    
    ell, qlms, nhl_data, qnorm, n1_data = results[0], results[1], results[2][0], results[3], results[4][0]
    ell = np.arange(1000); # 2024/4/14: bin_cell starts from 2. so the input should have ell = 0 and 1.
    
    kswitch = w(ell)
    fsky = fsky
    weights_phi = qnorm[ell] ** 2 / fsky 
    n0 = nhl_data[ell] * qnorm[ell] ** 2
    n1 = n1_data[ell] * qnorm[ell] ** 2
    
    if not debias_n1:
        n1 = 0
    
    func = binning[1]
    
    if binning[0] == 'toshiyan':
        bins = binning[2].bp
        
    else:
        bins = binning[2]
        
    delta_ell = [(y - x)/2 - 0.5 for x, y in zip(bins[0:-1], bins[1:])]
    
    nstart = Nsim_mf
    if from_fg_res:
        qlms_fg = results[4] # also nsim together
        nhl_data_fg, qnorm_fg  = results[5][0], results[6]
        n0_fg = nhl_data_fg[ell] * qnorm_fg[ell] ** 2 
        
    else:
        # assert subtract_mf is not None, subtract_mf
        if isinstance(mf, str):
            mf_alm = hp.read_alm(mf)
            
        else: 
            mf_alm = np.mean(qlms[:Nsim_mf], axis = 0)
    
        mf_cl = (hp.alm2cl(mf_alm)[ell])*weights_phi
    
    cleaned_my = []; mc_mfs = []
    for i in range(nstart, len(qlms)):
        
        if subtract_mf == 'alm':
            cleaned_my_i = (hp.alm2cl(qlms[i] - mf_alm)[ell])*weights_phi - n0 - n1 # n0 and qnorm are the same for every realization
            returned_fg = 0
        else:
            cleaned_my_i = (hp.alm2cl(qlms[i])[ell])*weights_phi - n0 - n1

        if from_fg_res:
            assert subtract_mf is None, 'subtract_mf should be None while fg_res, but it is %s'%subtract_mf
            returned_fg = (hp.alm2cl(qlms_fg[i])[ell])* weights_phi - n0_fg 
            
        elif subtract_mf == 'cl':
            returned_fg = mf_cl
        
        elif subtract_mf is None: # not subtracting MF
            mf_cl = 0
            returned_fg = 0
            if i == (len(qlms) -1):
                print('Not subtracting MF!')
            
        mc_mf = (cleaned_my_i + n0)/200 # use theory cl_phiphi
        mc_mfs.append(mc_mf)
        
        if not debias_mcmf:
            mcmf = 0
            
        debiased = cleaned_my_i - returned_fg - mc_mf
        
        if binning[0] == 'my':
            ell_binned, cleaned_binned_i =  func((debiased)*kswitch, lmax = bins[-1], bins = bins, ell_2=False)  
            
        elif binning[0] == 'planck': # kswitch is already included in planck binner
            ell_binned, cleaned_binned_i =  func(debiased) 
        
        elif binning[0] == 'toshiyan':
            assert len(binning) == 3, len(binning)
            ell_binned, cleaned_binned_i =  binning[2].bc, func((debiased)*kswitch, binning[2]) 
            
        cleaned_my.append(cleaned_binned_i)
        
    cleaned_mean = np.mean(cleaned_my, axis = 0)
    cleaned_std = np.std(cleaned_my, axis = 0)

    xdata = np.array(ell_binned)
    # xerr = np.array([delta_ell, delta_ell])
    xerr = np.array([np.zeros_like(delta_ell), np.zeros_like(delta_ell)])
    
    if binning[0] == 'my':
        ydata = cleaned_mean[0]
        yerr = np.array([cleaned_std[0], cleaned_std[0]])
        
    else:
        ydata = cleaned_mean
        yerr = np.array([cleaned_std, cleaned_std])        
    
    # print(n1*kswitch)
    if return_mf_cl:
        assert not from_fg_res, 'mean field and fg_res should not be estimated simultaneously.'
        return [xdata, ydata, xerr, yerr], mf_cl*kswitch, n0*kswitch, n1*kswitch, mc_mf*kswitch
    
    return [xdata, ydata, xerr, yerr], n0*kswitch, n1*kswitch, mc_mf*kswitch

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def get_variance_fg_res(results, fsky):
    '''
    Get errors for ONE foreground case
    
    Parameters
    ----------
    
    results: unbinned results from run_qe function 
    
    subtract_mf: 'alm' or 'cl' or None, whether subtracting the mean field or not; If not None, subtracting at the alm space or cl space
    mf: str, 0, or None; If str, input mean_field_alms will be loaded and to be subtracted;
    Nsim_mf: int or None, number of realizations in the resutls used to estimate mean field ; If int, the former Nsim_mf will be used ([0:Nsim_mf])
    
    '''
    
    ell, qnorm = results[0], results[3]
    
    fsky = fsky
    weights_phi = qnorm[ell] ** 2 / fsky * w(ell)
    
    nstart = 0
    qlms_fg = results[4] # also nsim together
    nhl_data_fg, qnorm_fg  = results[5][0], results[6]
    n0_fg = nhl_data_fg[ell] * qnorm_fg[ell] ** 2 * w(ell)    
    
    bins = [2, 21, 40, 66, 101, 145, 199, 264, 339, 426, 526, 638, 763, 1000]
    
    cleaned_my = []
    for i in range(nstart, len(qlms_fg)):

        returned_fg = (hp.alm2cl(qlms_fg[i])[ell])* weights_phi - n0_fg 
        ell_binned, cleaned_binned_i =  bin_cell(returned_fg, lmax = 1000, bins = bins, ell_2=False)  
        cleaned_my.append(cleaned_binned_i)
        
    cleaned_mean = np.mean(cleaned_my, axis = 0)
    cleaned_std = np.std(cleaned_my, axis = 0)

    delta_ell = [(y - x)/2 - 0.5 for x, y in zip(bins[0:-1], bins[1:])]
    # 0.5 for visuallization 

    xdata = np.array(ell_binned)
    ydata = cleaned_mean[0]
    
    yerr = np.array([cleaned_std[0], cleaned_std[0]])
    xerr = np.array([delta_ell, delta_ell])
    
    return [xdata, ydata, xerr, yerr]

def plot_variance(ell, data_error, labels, kind = 'box', title = 'Title', xscale = 'log', yscale = 'log', set_ylim = True, mf_cls = None, n0s = None, n1s = None, mc_mf = None, 
                  true_binned = None, xlim = 1000):
    '''
    data_error: [[xdata, ydata, xerr, yerr]]
    kind: box or bar
    '''
    # Create figure and axes
    fig, axes = plt.subplots(2, 1, sharex=True, figsize = (10, 8),  gridspec_kw={'height_ratios':[3,1]})
    axes[0].plot(ell, cl_unl['pp'][ell]* w(ell), 'k--')
    axes[0].plot(data_error[0][0], true_binned, 'k-')
    
    if kind == 'box':
        edgecolor='none'
        alpha=0.5

        boxes = []
        for i in range(len(data_error)):

            facecolor = colors[i]

            # Loop over data points; create box from errors at each point
            errorboxes = [Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum(), label = '%s'%labels[i], facecolor=facecolor)
                      for x, y, xe, ye in zip(data_error[i][0], data_error[i][1], data_error[i][2].T, data_error[i][3].T)]

            boxes.append(errorboxes[0])

            # Create patch collection with specified colour/alpha
            pc = PatchCollection(errorboxes, alpha=alpha, facecolor=facecolor, 
                             edgecolor=edgecolor)

            # Add collection to axes
            axes[0].add_collection(pc)

            axes[0].scatter(data_error[i][0], data_error[i][1], s = 20, c = facecolor, alpha = 0.8)

        axes[0].legend(handles = boxes, frameon = False)
        
    if kind == 'bar':
        for i in range(len(data_error)):
            facecolor = colors[i]
             
            axes[0].errorbar(data_error[i][0] + i*0.5, data_error[i][1], yerr = data_error[i][3][0], 
                          ecolor = colors[i], label = '%s'%labels[i], fmt='o') #xerr = data_error[i][2][0],
            
            if mf_cls:
                axes[0].semilogx(ell, mf_cls[i][ell], color = facecolor, ls = '--')
            
            if n0s is not None:
                axes[0].semilogx(ell, n0s[i][ell], color = facecolor, ls = '-', alpha = 0.5)
            if n1s is not None:
                axes[0].semilogx(ell, n1s[i][ell], color = facecolor, ls = '--', alpha = 0.5)
            if mc_mf is not None:
                axes[0].semilogx(ell, mc_mf[ell], color = facecolor, ls = '-', alpha = 0.2)
            
            axes[1].scatter(data_error[i][0]  + i*0.5, (data_error[i][1] - true_binned)/data_error[i][3][0], s = 20, c = facecolor, alpha = 0.8)
            
            # axes[1].errorbar(data_error[i][0] + i*3, (data_error[i][1] - true_binned[0])/true_binned[0]*100, xerr = data_error[i][2][0], 
                             # yerr = data_error[i][3][0]/true_binned[0]*100, ecolor = colors[i], ls='none')
            
        axes[0].legend(frameon = False)
    axes[0].set_xlim(2, xlim)
    axes[1].set_xlim(2, xlim)
    if set_ylim:
        axes[0].set_ylim(1e-3, 10)
        # axes[0].set_ylim(1e-10, 1e-6)
        # axes[0].set_ylim(1e-3,4e1)
    axes[0].set_yscale(yscale)
    axes[0].set_xscale(xscale)
    # axes[0].set_yticks(np.arange(0.2, 1.8, 0.2))
    axes[0].set_ylabel('$L^2 (L + 1)^2 C_L^{\phi\phi}$  [$x10^7$]', fontsize=12)
    # axes[0].set_ylabel('$1/4 L^2 (L + 1)^2 C_L^{\phi\phi}$', fontsize=12)
    axes[1].set_xlabel('$L$', fontsize=12)
    axes[0].set_title(title)
    axes[1].set_ylabel(r'$(C_{\ell}^{O} - C_{\ell}^{I})/\sigma$', fontsize=12)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha = 0.4)
    axes[1].axhline(y=-3, color='k', linestyle='--', alpha = 0.4)
    axes[1].axhline(y=-1, color='k', linestyle='--', alpha = 0.4)
    axes[1].axhline(y=1, color='k', linestyle='--', alpha = 0.4)
    axes[1].axhline(y=3, color='k', linestyle='--', alpha = 0.4)
    axes[1].set_yticks(np.array((-3, -1, 0, 1, 3)))
    axes[1].set_ylim(-3, 3)
    fig.subplots_adjust(hspace=0)
    
    
def plot_variance_x(ell, data_error, labels, kind = 'box', title = 'Title', xscale = 'log', yscale = 'linear', set_ylim = True, mf_cls = None):
    '''
    for curl part.
    data_error: [[xdata, ydata, xerr, yerr]]
    kind: box or bar
    '''
    # Create figure and axes
    fig, axes = plt.subplots(2, 1, sharex=True, figsize = (10, 8),  gridspec_kw={'height_ratios':[3,1]})
    
    if kind == 'box':
        edgecolor='none'
        alpha=0.5

        boxes = []
        for i in range(len(data_error)):

            facecolor = colors[i]

            # Loop over data points; create box from errors at each point
            errorboxes = [Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum(), label = '%s'%labels[i], facecolor=facecolor)
                      for x, y, xe, ye in zip(data_error[i][0], data_error[i][1], data_error[i][2].T, data_error[i][3].T)]

            boxes.append(errorboxes[0])

            # Create patch collection with specified colour/alpha
            pc = PatchCollection(errorboxes, alpha=alpha, facecolor=facecolor, 
                             edgecolor=edgecolor)

            # Add collection to axes
            axes[0].add_collection(pc)

            axes[0].scatter(data_error[i][0], data_error[i][1], s = 20, c = facecolor, alpha = 0.8)

        axes[0].legend(handles = boxes, frameon = False)
        
    if kind == 'bar':
        for i in range(len(data_error)):
            facecolor = colors[i]
            axes[0].errorbar(data_error[i][0] + i*3, data_error[i][1], xerr = data_error[i][2][0], yerr = data_error[i][3][0], 
                          ecolor = colors[i], ls='none', label = '%s'%labels[i])
            
            if mf_cls:
                axes[0].semilogx(ell, mf_cls[i], color = facecolor, ls = '--')
            
            axes[1].scatter(data_error[i][0]  + i*3, (data_error[i][1])/data_error[i][3][0], s = 20, c = facecolor, alpha = 0.8)
            
        axes[0].legend(frameon = False)
    axes[0].set_xlim(8, 1000)
    # if set_ylim:
        # axes[0].set_ylim(1e-3, 10)
        # axes[0].set_ylim(1e-3,4e1)
    axes[0].axhline(y=0, color='k', linestyle='--', alpha = 0.4)
    axes[0].set_yscale(yscale)
    axes[0].set_xscale(xscale)
    # axes[0].set_yticks(np.arange(0.2, 1.8, 0.2))
    axes[0].set_ylabel('$L^2 (L + 1)^2 C_L^{\phi\phi}$  [$x10^7$]', fontsize=12)
    axes[1].set_xlabel('$L$', fontsize=12)
    axes[0].set_title(title)
    axes[1].set_ylabel(r'$(C_{\ell}^{O} - C_{\ell}^{I})/\sigma$', fontsize=12)
    # axes[1].axhline(y=0, color='k', linestyle='--', alpha = 0.4)
    # axes[1].axhline(y=-3, color='k', linestyle='--', alpha = 0.4)
    # axes[1].axhline(y=-1, color='k', linestyle='--', alpha = 0.4)
    # axes[1].axhline(y=1, color='k', linestyle='--', alpha = 0.4)
    # axes[1].axhline(y=3, color='k', linestyle='--', alpha = 0.4)
    axes[1].set_yticks(np.array((-3, -1, 0, 1, 3)))
    # axes[1].set_ylim(-10, 100)
    fig.subplots_adjust(hspace=0)