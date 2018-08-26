import numpy as np 
import mkl_fft
from functools import partial
import copy

from .filters import *
from .convolution import *


class ScatOptions():
    def __init__(self, path_margin = 0, oversampling = 1,
                 x_resolution = 0, psi_mask = None, M = None):
        self.path_margin = path_margin
        self.oversampling = oversampling
        self.x_resolution = x_resolution
        self.psi_mask = psi_mask  
        self.M = M
        
class MetaPhi():
    def __init__(self, j = None, bandwidth = None, resolution = None):
        self.j = j
        self.bandwidth = bandwidth
        self.resolution = resolution
    
class MetaPsi():
    def __init__(self, count):
        self.j = -1*np.ones(count)
        self.bandwidth = -1*np.ones(count)
        self.resolution = -1*np.ones(count)


class MetaU():
    def __init__(self, bandwidth = None, resolution = None, j = None):
        self.bandwidth = bandwidth
        self.resolution = resolution
        self.j = j

class LayerU():
    def __init__(self, metaU = None, signal = None):
        self.meta = metaU
        self.signal = signal



def wavelet_1d(x, filters, options = None):
    '''
    Input    
        x: The signal to be transformed. Input x must be 3-d array Nx1xK,
            where K is the number of signals
        options:
            x_resolution: The resolution of the input signal x as
                a power of 2, representing the downsampling of the signal
                with respect to the finest resolution (default 0)
            oversampling: The oversampling factor (as a power of 2) 
                with respect to the critical bandwidth when calculating
                convolutions (default 1).
                psi_mask: Specifies the wavelet filters in filters.psi
                for which the transform is to be calculated (default all).
    Output
        x_phi: x filtered by the lowpass filter filters.phi.
        x_psi: cell array of x filtered by the wavelets filters.psi.
        meta_phi, meta_psi: meta information on x_phi and x_psi, respectively
        
    '''
    #уточнить как   заполнять
    if options is None:
        options = ScatOptions()
        options.psi_mask = np.ones(len(filters.psi.filter))
    
    N = x.shape[0]
    
    center_psi, bw_psi, bw_phi = morlet_freq_1d(filters.meta)
    
    j0 = options.x_resolution
    
    N_padded = filters.meta.filter_size/(2**j0)
    
    # Convert x to the shape N_padded x K to perform faster FT

    x = x[:,-1,:]
    x = pad_signal(x, N_padded,
                   boundary = filters.meta.boundary, center = False)
    
    # mklt_fft.fft make fft along columns
    x_fft = mkl_fft.fft(x.T).T
    # print('inside wavelet_1d: x_fft shape:', x_fft.shape)
    # print(x_fft[:10,0])
    
    ds = np.ceil(np.log2(2*np.pi/bw_phi)) -j0 - options.oversampling
    ds = np.max(ds, 0)
    # ds = 0
    # print('inside wavelet_1d: ds:', ds)
    # print('inside wavelet_1d: filter shape:', filters.phi.filter[0].shape)

    
    x_phi = np.real(conv_sub_1d(x_fft, filters.phi.filter[0], ds))
    # in matlab code they forgot to mention center parameter!
    # print(x_phi)
    x_phi = unpad_signal(x_phi, N, resolution = ds,
                         center = False)
    
    # print(x_phi)
    x_phi = x_phi[:, np.newaxis,:]
    meta_phi = MetaPhi(j = -1, bandwidth = bw_phi, resolution = j0 + ds)
    # print('inside wavelet_1d: x_phi shape:', x_phi.shape)
    # print('inside wavelet_1d: x_phi meta:', vars(meta_phi))
    
    
    x_psi = np.empty(len(filters.psi.filter), dtype = object)
    meta_psi = MetaPsi(len(filters.psi.filter))
    
    for p1 in np.nonzero(options.psi_mask)[0]:
        
        ds = np.round(np.log2(2*np.pi/bw_psi[p1]/2))-j0-options.oversampling
        ds = max(ds, 0)
        # ds = 0
        # print('Inside wavelet_1d: p1 ', p1)
        # print('Inside wavelet_1d: ds before conv_sub_1d ', ds)
        x_psi[p1] = conv_sub_1d(x_fft, filters.psi.filter[p1], ds)
        x_psi[p1] = unpad_signal(x_psi[p1], N, resolution = ds,
                                 center = False)
        x_psi[p1] = x_psi[p1][:, np.newaxis, :]
        
        meta_psi.j[p1] = p1
        meta_psi.bandwidth[p1] = bw_psi[p1]
        meta_psi.resolution[p1] = j0+ds
    
    # print('inside wavelet_1d: x_psi shape:', x_psi.shape)
    # print('inside wavelet_1d: x_psi meta:', vars(meta_psi))
    
    return x_phi, x_psi, meta_phi, meta_psi


def wavelet_layer_1d(U, filters, scat_opt = None):
    '''
    Compute the one-dimensional wavelet transform from
    the modulus wavelet coefficients of the previous layer.
    
    Input
        U: The input layer to be transformed.
        filters: The filters of the wavelet transform.
        scat_opt: The options of the wavelet layer. Some are used in the
           function itself, while others are passed on to the
           wavelet transform. The parameters used by wavelet_layer_1d are:
              
              path_margin: The margin used to determine wavelet decomposition
                  scales with respect to the bandwidth of the signal.
                  If the bandwith of a signal in U is bw, only wavelet filters
                  of center frequency less than bw*2^path_margin are applied
                  (default 0). 
        wavelet: the wavelet transform function (default wavelet_1d).  
    Output
        U_phi The coefficients of in, lowpass-filtered (scattering coefficients).
        U_psi: The wavelet transform coefficients.
        
    Description
        This function has a pivotal role between wavelet_1d (which computes
        a single wavelet transform), and wavelet_factory_1d (which creates
        the whole cascade). Given inputs modulus wavelet coefficients
        corresponding to a layer, wavelet_layer_1d computes the wavelet
        transform coefficients of the next layer using wavelet_1d. 

    '''    
    if scat_opt is None:
        scat_opt = ScatOptions()
    scat_opt.path_margin = 0
    
    if U.meta.bandwidth is None:
        U.meta.bandwidth = [2*np.pi]
    if U.meta.resolution is None :
        U.meta.resolution = [0]
        
    center_psi, bw_psi, bw_phi = morlet_freq_1d(filters.meta) 
        
    U_phi = LayerU(metaU = MetaU(bandwidth = [],
                                  resolution = [],
                                  j = []),
                   signal = [])
    
    U_psi = LayerU(metaU = MetaU(bandwidth = [],
                                  resolution = [],
                                  j = []),
                   signal = [])
 
    r = 0
    for p1 in range(len(U.signal)):
        
        current_bw = U.meta.bandwidth[p1] * 2**scat_opt.path_margin
        psi_mask = (current_bw > center_psi) 
        
        scat_opt.x_resolution = U.meta.resolution[p1]
        scat_opt.psi_mask = psi_mask

        x_phi, x_psi, meta_phi, meta_psi = wavelet_1d(U.signal[p1],
                                                      filters,
                                                      scat_opt)
        # проверить все ниже!
        U_phi.signal.append(x_phi)
        
        U_phi.meta.j.append(U.meta.j[:, p1:p1+1])
        U_phi.meta.bandwidth.append(meta_phi.bandwidth)
        U_phi.meta.resolution.append(meta_phi.resolution)      
              
        ind = np.arange(r, r + psi_mask.sum())
        U_psi.signal.extend(x_psi[psi_mask])

        U_psi.meta.bandwidth.extend(meta_psi.bandwidth[psi_mask])
        U_psi.meta.resolution.extend(meta_psi.resolution[psi_mask])

         
        U_psi.meta.j.append(np.concatenate([np.repeat(U.meta.j[:, p1:p1+1], len(ind), axis = 1),
                                            [meta_psi.j[psi_mask]]]
                                          ))
        
        r += len(ind)
    
    if len(U_phi.meta.j) > 0:
        U_phi.meta.j = np.concatenate(U_phi.meta.j, axis = 1)
    else:
        U_phi.meta.j = U.meta.j
        
    U_psi.meta.j = np.concatenate(U_psi.meta.j, axis = 1)
        
    return U_phi, U_psi    


def modulus_layer(W):
    U = copy.deepcopy(W)
    U.signal = list(map(lambda w: np.abs(w), W.signal))
    return U



def wavelet_factory_1d(N, filt_opt_bank, scat_opt = None):
    bank_filters = filter_bank(N, filt_opt_bank)
    
    if scat_opt is None:
        scat_opt = ScatOptions()
        
    if scat_opt.M is None:
        scat_opt.M = 2
    
    Wop = []
    for m in range(scat_opt.M + 1):
        filt_ind = min(len(bank_filters)-1, m)

        Wop.append(partial(wavelet_layer_1d,
                           filters = bank_filters[filt_ind],
                           scat_opt = scat_opt))
        
    return Wop, bank_filters


def scat(x, Wop):
    '''
    Input
        x: The input signal.
        Wop: Linear operators used to generate a new layer from the previous one.

    Output
        S: The scattering representation of x.
        U: Intermediate covariant modulus coefficients of x.

    Description
        The signal x is decomposed using linear operators Wop and modulus 
        operators, creating scattering invariants S and intermediate covariant
        coefficients U. 

        Each element of the Wop array is a function handle of the signature
        [A, V] = Wop{m+1}(U),
        where m ranges from 0 to M (M being the order of the transform). The 
        outputs A and V are the invariant and covariant parts of the operator.

        The variables A, V and U are all of the same structure, that of a network
        layer. Specifically, the have one field, signal, which is an array
        corresponding to the constituent signals, and another field, meta, which
        contains various information on these signals.

        The scattering transform therefore initializes the first layer of U using
        the input signal x, then iterates on the following transformation
        [S{m+1}, V] = Wop{m+1}(U{m+1});
        U{m+2} = modulus_layer(V);
        The invariant part of the linear operator is therefore output as a scat-
        tering coefficient, and the modulus covariant part V is assigned to the 
        next layer of U.
    '''
    Us = [None]*len(Wop)
    
    Us[0]= LayerU(metaU = MetaU(j = np.empty((0,1))),
                  signal = [x])
    
    S = [None]*len(Wop)
    
    for m in range(len(Wop)):
        if m < len(Wop) - 1:
            S[m], V = Wop[m](Us[m])
            Us[m+1] = modulus_layer(V)
        else:
            S[m], _ = Wop[m](Us[m])
        
    return S, Us


