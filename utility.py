import numpy as np


Ell = lambda ell: ell*(ell+1)/2/np.pi

def bin_cell(cls, lmax, bin_width, ell_2 = True):
    '''
    cls: list; cls including several XX or for only one XX, X means T, E, B
    bin_width: 
    lmax:
    
    Return
    ------
    ells_bin: binned ells, don't include multipoles beyond the last bin_edge defined by N*bin_width
    cls_bin: binned cell
    ell_2: Bool; If True, factor ell*(ell+1)/2/np.pi are applied to the output binned cell.
    
    '''
    N = lmax//bin_width
    
    if len(cls) != 6:
        cls = [cls]
        
    cls_bin = []
    for s in range(len(cls)):
        cls_bin_i = np.zeros(N)
        
        if s == 0:
            ells = np.arange(len(cls[0]))
            ells_bin = []
        
        for i in range(N):
            if s == 0:
                ells_bin.append(np.mean(ells[i*bin_width:(i+1)*bin_width]))
            
            if ell_2:
                cls_bin_i[i] = Ell(ells_bin[i])*np.mean(cls[s][i*bin_width:(i+1)*bin_width])
                
            else: 
                cls_bin_i[i] = np.mean(cls[s][i*bin_width:(i+1)*bin_width])
        
        cls_bin.append(cls_bin_i)
    
    return ells_bin, cls_bin