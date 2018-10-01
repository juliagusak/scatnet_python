import numpy as np

import matlab
import matlab.engine


def get_scattering_coefficients(path_to_scatnet, np_arr_batch, N, T, M = 2,
                                Q = [8, 1], renorm = False, log = False):
    '''  
    args:
        path_to_scatnet - path to the folder with matlab scattering implementataion,
        np_arr_batch - batch of signals, where one signal is represented with  1d numpy array of size (1, length) or (length, 1),
        N - signal length,
        T - filter length,  
        Q - factors for constant-Q filter banks,
        M - highest order of scattering coefficients we want to count (linear operators that make up the layers of the scattering network)
        
    return:
         S_table_count - batch of P-by-Z numpy arrays, where P is the total number of scattering coefficients
         (all orders combined) and Z is the number of time points. One array corresponds to one signal;
         
         time_count - Z, number of time points, i.e. number of signal windows (consecutive slices) on which scattering coefficients were counted;
         
         coeffs_count - list of length M, where coeffs_count[m] is equal to the number of scattering coefficients of order m and sum(coeffs_count) = P.
    '''
    
    # start matlab engine
    eng = matlab.engine.start_matlab()
    
    # add pathes to matlab scripts
    eng.eval("addpath {}; addpath_scatnet;".format(path_to_scatnet), nargout = 0)
       
    # create matlab values N, T, M of type double
    eng.eval("N = {:d}; T = {:d}; M = {:d};".format(N, T, M), nargout = 0)
    
    
    # set filter options, get wavelet filter bank
#     eng.eval("".join(["filt_opt = default_filter_options('audio', T);",
#                       "scat_opt.M = M;",                          
#                       "[Wop, filters] = wavelet_factory_1d(N, filt_opt, scat_opt)"]), nargout = 0)
    eng.eval("".join(["filt_opt.Q = {};".format(str(Q)),
                      "filt_opt.J = T_to_J(T, filt_opt);",
                      "scat_opt.M = M;",
                      "[Wop, filters] = wavelet_factory_1d(N, filt_opt, scat_opt);"]), nargout = 0)
    
    
    S_table_batch = []
    for np_arr in np_arr_batch:
        # convert numpy array to matlab array
        # "A.'" -  A transposed in matlab    
        mat_arr = matlab.double(np_arr.tolist())
        eng.eval("y = {}.';".format(mat_arr), nargout = 0)

        # count 0,1,2,..,M-order scattering coefficients 
        eng.eval('S = scat(y, Wop);', nargout = 0)

        if renorm:
            eng.eval("S = renorm_scat(S);", nargout = 0)

        if log:
            eng.eval("S = log_scat(S);", nargout = 0)

        # assemple scattering coefficients in vector form
        eng.eval("[S_table, meta] = format_scat(S);", nargout = 0)
        S_table = eng.eval('S_table')
        
        S_table_batch.append(S_table)
        
    # Get number of scattering coefficients
    time_count = np.array(S_table).shape[1]
    
    coeffs_count = [eng.eval("size(S{}.signal, 2);".format('{'+str(i+1)+'}'))
                    for i in range(M+1)]
    
    return  np.array(S_table_batch), time_count, coeffs_count