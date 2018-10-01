import numpy as np
import librosa

import scattering


if __name__=="__main__":
    
    path_to_scatnet = '/home/julia/Downloads/scatnet-master/'
    
    # create input batch
    signal_path = "/home/julia/DeepVoice_data/samples/clear.wav"
    signal_wav, _ = librosa.load(signal_path)
    
    np_arr_batch = np.array([signal_wav]*2)
    N = np_arr_batch.shape[1]
    T = 1024

    # Count scattering coefficients
    S_table_batch, time_count, coeffs_count = scattering.get_scattering_coefficients(path_to_scatnet, np_arr_batch, N, T, M = 2, Q = [8,1], renorm = True, log = True)
    
    print(S_table_batch.shape, time_count, coeffs_count)