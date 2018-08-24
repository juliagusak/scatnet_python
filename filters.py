import numpy as np 

class FiltOptions(object):
    '''
    Fields:
    T: Size of averaging interval
    Q: Number of wavelets per octave
    B: Reciprocal per-octave bandwidth of the wavelets
    phi_bw_multiplier: Ratio between the bandwidth of the lowpass filter phi
        and lowest-frequency wavelet
    filter_type: Can be 'morlet_1d' or 'gabor_id'
    boundary: Sets the boundary conditions of the wavelet transform. 
        If 'symm', symmetric boundaries will be used, 
        if 'per', periodic boundaries will be used (default 'symm')
    precision: Either 'float64', or 'float32'. Determines the precision of the filters
        stored and consequently that of the resulting
        wavelet and scattering transform (default 'float64')
    phi_dirac: 
    sigma0: 
    center_psi: Center frequency of the mother wavelet
    sigma_psi: Standard deviation of the mother wavelet in space
    sigma:phi: Standard deviation of the scaling function in space

    J: Number of logarithmically spaced wavelets
    P: Number of linearly spaced band-pass filters. 
        If sigma_psi is smaller than a certaind threshhold,
        a number of linearly-spaced constant-bandwidth filters are added
    '''

    def __init__(self, T, Q = None, B = None,
                 phi_bw_miltiplier = None, sigma0 = None,
                 boundary = 'symm', filter_size = None,
                 precision = 'float64', filter_type = 'morlet_1d',
                 phi_dirac = 0):
        self.T = T
       
        if Q is None:
            self.Q = 1
        else:
            self.Q = Q
        
        if B is None:
            self.B = self.Q
        else:
            self.B = B
        
        if  phi_bw_miltiplier is None:
            self.phi_bw_miltiplier = 1 + (self.Q == 1)
        else:
            self.phi_bw_miltiplier = phi_bw_miltiplier

        if sigma0 is None:
            self.sigma0 = 2/np.sqrt(3)
        else:
            self.sigma0 = sigma0
            
         # mean and standard deviation of mother wavelet
        self.center_psi0 = self.count_center_psi0()
        self.sigma_psi0 = self.count_sigma_psi0()
        self.sigma_phi0 = self.count_sigma_phi0()

        self.J = self.T_to_J()
        self.P = self.count_P()
         
        self.boundary = boundary
                
        self.precision = precision
        self.filter_type = filter_type
        self.phi_dirac = phi_dirac
    
    
    def count_center_psi0(self):
        return np.pi/2*(2**(-1/self.Q) + 1)
    
    
    def count_sigma_psi0(self):
        return 0.5*self.sigma0/(1 - 2**(-1/self.B))

    def count_sigma_phi0(self):
        return  self.sigma_psi0/self.phi_bw_miltiplier  
    
    def T_to_J(self):
        return int(1 + round(np.log2(self.T*self.phi_bw_miltiplier/4/self.B))*self.Q) 
    
    def count_P(self):
        return int(round((2**(-1/self.Q) - 0.25*self.sigma0/self.sigma_phi0)/(1-2**(-1/self.Q))))
      
 
class Filters(object):
    def __init__(self, T, **kwargs):
        self.meta = FiltOptions(T, **kwargs) 
        self.psi = FilterPsi(self.meta.J, self.meta.P)
        self.phi = FilterPhi()
        
class FilterPsi(object):
    def __init__(self, J, P):
        self.meta = np.empty(J + P, dtype = object)
        self.filter = np.empty(J + P, dtype = object)

        
class FilterPhi(object):
    def __init__(self):
        self.meta = np.empty(1, dtype = object)
        self.filter = np.empty(1, dtype = object)


def morlet_freq_1d(filt_opt):
    '''
    Input
        filt_opt: The parameters defining the filter bank. See 
           MORLET_FILTER_BANK_1D for details.

    Output
        center_psi: The center frequencies of the wavelet filters.
        bw_psi: The bandwidths of the wavelet filters.
        bw_phi: The bandwidth of the lowpass filter

    Description
        Compute the center frequencies and bandwidth for the wavelets and lowpass
        filter of the one-dimensional Morlet/Gabor filter bank.
    ''' 

    center_psi = np.zeros(filt_opt.J + filt_opt.P)
    sigma_psi = np.zeros(filt_opt.J + filt_opt.P)
        
    # Calculate logarithmically spaced, band-pass filters.
    center_psi[:filt_opt.J] = filt_opt.center_psi0 * np.float_power(2, -np.arange(0, filt_opt.J)/filt_opt.Q)
    sigma_psi[:filt_opt.J] = filt_opt.sigma_psi0 * 2**(np.arange(0, filt_opt.J)/filt_opt.Q)
    
    # Calculate linearly spaced band-pass filters so that they evenly
    # cover the remaining part of the spectrum


    if filt_opt.P > 0:
    	step = np.pi * 2**(-filt_opt.J/filt_opt.Q)*(1 - 0.25*filt_opt.sigma0/filt_opt.sigma_phi0*2**(1/filt_opt.Q))/filt_opt.P
    else:
    	step = 0
    center_psi[filt_opt.J : filt_opt.J+filt_opt.P] = filt_opt.center_psi0 * 2**(-(filt_opt.J-1)/filt_opt.Q) - step*np.arange(1, filt_opt.P+1)
    sigma_psi[filt_opt.J : filt_opt.J + filt_opt.P] = filt_opt.sigma_psi0 * 2**((filt_opt.J-1)/filt_opt.Q)
    
    # Calculate low-pass filter
    sigma_phi = filt_opt.sigma_phi0 * 2**((filt_opt.J-1)/filt_opt.Q)
    
    # Convert (spatial) sigmas to (frequential) bandwidths
    bw_psi = np.pi/2 * filt_opt.sigma0/sigma_psi
    
    if not filt_opt.phi_dirac:
        bw_phi = np.pi/2 * filt_opt.sigma0/sigma_phi
    else:
        bw_phi = 2*np.pi
    
    return center_psi, bw_psi, bw_phi


def gabor(N, center, sigma, precision = 'float64'):
    '''
    Output
        f: Fourier transform of the filter
    '''
    # extent of periodization, the higher the better
    extent = 1
    
    f = np.zeros(N).astype(precision)
    
    # Calculate the 2*pi-periodization of the filter over 0 to 2*pi*(N-1)/N
    # For the filter's Fourier transform  use the formula mentioned above
    
    # Summation for k = -1, 0, 1  means that we include components with
    # frequencies wc+2pi, wc, wc-2pi, to the filter frequency representation
    
    # (Adding to the filter impulse response components with frequencies wc + 2pi*k
    # allow us to catch power spectrum of harmonics, corresponding to wc?)

    # center/ 2pi*center - angular velocity/frequency of the filter
    for k in range(-extent, extent+2):
        f = f + np.exp(-sigma**2*((np.arange(N)-k*N)/N*2*np.pi - center)**2/2)    
    return f


def morletify(f, sigma):
    '''
        f: Fourier transform of the filter
    '''
    f0 = f[0]
    
    f = f - f0*gabor(len(f), 0, sigma, f.dtype)
    return f


def optimize_filter(f, is_lowpass, filt_opt):
    '''
        f: Fourier transform of the filter
        is_lowpass: if True, f contains lowpass filter
    '''
    return f


def morlet_filter_bank_1d(signal_length, filt_opt):
    '''
    MORLET_FILTER_BANK_1D Create a Morlet/Gabor filter bank%
    Input
        sz: The size of the input data.
        options (struct, optional): Filter parameters, see below.

    Output
        filters: The Morlet/Gabor filter bank corresponding to the data 
           size sz and the filter parameters in options.

    Description
    Depending on the value of options.filter_type, the functions either
    creates a Morlet filter bank (for filter_type 'morlet_1d') or a Gabor
    filter bank (for filter_type 'gabor_1d'). The former is obtained from 
    the latter by, for each filter, subtracting a constant times its enve-
    lopes uch that the mean of the resulting function is zero.

    The following parameters can be specified in options:
       options.filter_type (char): See above (default 'morlet_1d').
       options.Q (int): The number of wavelets per octave (default 1).
       options.B (int): The reciprocal per-octave bandwidth of the wavelets 
          (default Q).
       options.J (int): The number of logarithmically spaced wavelets. For  
          Q=1, this corresponds to the total number of wavelets since there 
          are no  linearly spaced ones. Together with Q, this controls the  
          maximum extent the mother wavelet is dilated to obtain the rest of 
          the filter bank. Specifically, the largest filter has a bandwidth
          2^(J/Q) times that of the mother wavelet (default 
          T_to_J(sz, options)).
       options.phi_bw_multiplier (numeric): The ratio between the bandwidth 
          of the lowpass filter phi and the lowest-frequency wavelet (default
           2 if Q = 1, otherwise 1).
       options.boundary, options.precision, and options.filter_format: 
          See documentation for the FILTER_BANK function.
        signal_length - almost number of points to evaluate filter at 
    '''
        
    sigma0 = filt_opt.sigma0
    
    
    # N - number of points to evaluate filter at
    if filt_opt.boundary == 'symm':
        N = 2*signal_length
    else:
        N = signal_length
        
    # Increase N to be the power of 2. Then resolution = log2(N) 
    N = int(2**np.ceil(np.log2(N)))
    
    do_gabor = (filt_opt.filter_type == 'gabor_1d')
    
    filters = Filters(T=filt_opt.T, Q = filt_opt.Q)
#     filters.meta = filt_opt
    filters.meta.filter_size = N
    
    
    center_psi, bw_psi, bw_phi = morlet_freq_1d(filters.meta)
    
    sigma_psi = sigma0*np.pi/2./bw_psi
    sigma_phi = sigma0*np.pi/2/bw_phi
    
    # Calculate normalization of filters so that sum of squares does not
    # exceed 2. This guarantees that the scattering transform is contractive.
    
    # As it occupies a larger portion of the spectrum, it is more
    # important for the logarithmic portion of the filter bank to be
    # properly normalized, so we only sum their contributions.
    
    Sum = np.zeros(N).astype(filt_opt.precision)
    
    for j1 in range(filt_opt.J):        
        temp = gabor(N, center_psi[j1], sigma_psi[j1], filt_opt.precision)

        if not do_gabor:
            temp = morletify(temp, sigma_psi[j1])
        
        Sum += np.abs(temp)**2
    psi_ampl = np.sqrt(2/np.max(Sum))   
    
    
    # Apply the normalization factor to the filters.

    for j1 in range(len(filters.psi.filter)):
        temp = gabor(N, center_psi[j1], sigma_psi[j1], filt_opt.precision)
        
        if not do_gabor:
            temp = morletify(temp, sigma_psi[j1])
        
        filters.psi.filter[j1] = optimize_filter(psi_ampl*temp, 0, filt_opt)
        filters.psi.meta[j1] = j1 
        
    # Calculate the associated low-pass filter
    
    if not filt_opt.phi_dirac:
        filters.phi.filter[0] = gabor(N, 0, sigma_phi, filt_opt.precision)
    else:
        filters.phi.filter = np.ones(N).astype(filt_opt.precision)
    
    filters.phi.filter[0] = optimize_filter(filters.phi.filter[0], 1, filt_opt)
    filters.phi.meta[0] = filt_opt.J + filt_opt.P
    
    return filters  


def filter_bank(signal_length, filt_opt_bank):
    '''
    FILTER_BANK Create a cell array of filter banks

    Input
        N: The size of the input data.
        options: Filter parameters, see below.

    Output
        filters (struct): A cell array of filter banks corresponding to the
          data size N and the filter parameters in options.

    Description
    The behavior of the function depends on the value of options.filter_type,
    which can have the following values:
       'morlet_1d', 'gabor_1d': Calls MORLET_FILTER_BANK_1D.
       'spline_1d': Calls SPLINE_FILTER_BANK_1D.
       'selesnick_1d': Calls SELESNICK_FILTER_BANK_1D.
    The filter parameters in options are then passed on to these functions.
    If multiple filter banks are desired, multiple parameters can be supplied
    by providing a vector of parameter values instead of a scalar (in the 
    case of filter_type and filter_format, these have to be cell arrays).
    The function will then split the options structure up into several 
    parameter sets and call the appropriate filter bank function for each one
    returning the result as a cell array. If a given field does not have 
    enough values to specify parameters for all filter banks, the last ele-
    ment is used to extend the field as needed.

    The specific parameters vary between filter bank types, but the follow-
    ing are common to all types:
       options.filter_type: Can be 'morlet_1d', 'gabor_1d', 'spline_1d',
          or 'selesnick_1d' (default 'morlet_1d').
       options.boundary: Sets the boundary conditions of the wavelet trans-
          form. If 'symm', symmetric boundaries will be used, if 'per', per-
          iodic boundaries will be used (default 'symm').
       options.precision: Either 'double', or 'single'. Determines the preci-
          sion of the filters stored and consequently that of the resulting
          wavelet and scattering transform (default 'double').
       options.filter_format: Specifies the format in which the filters are 
          stored. Three formats are available:
              'fourier': Filters are stored as Fourier transforms defined
                 over the whole frequency domain of the signal.
             'fourier_multires': Filters are stored as Fourier transforms 
                 defined over the frequency domain of the signal at all
                 resolutions. This requires much more memory, but speeds up
                 calculations significantly.
              'fourier_truncated': Stores the Fourier coefficients of each
                 filter on the support of the filter, reducing memory
                 consumption and increasing speed of calculations (default).

    '''
    
    filters = np.empty(len(filt_opt_bank), dtype = object)
    
    for k in range(len(filt_opt_bank)):
        options_k = filt_opt_bank[k]
        
        # Calculate the k-th filter bank
        filter_type = options_k.filter_type
        
        if filter_type == 'morlet_1d':
            filters[k] = morlet_filter_bank_1d(signal_length, options_k)
        else:
            # can be any other filter bank
            filters[k] = morlet_filter_bank_1d(signal_length, options_k)
        
    return filters


