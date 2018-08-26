import numpy as np 
import mkl_fft

from .filters import *

# could be extendede to pad multiple 1d arrays with different lengths
# https://web.eecs.umich.edu/~fessler/course/451/l/pdf/updown.pdf
    
def pad_signal(x, N_padded, boundary = 'symm', center = False):
    '''
    Input
        x: The signal to be padded, 2-d array N_original x K,
            where K is the number of signals 
        N_padded: The desired size of the padded output signal.
        boundary: The boundary condition of the signal, one of:
            'symm': Symmetric boundary condition with half-sample symmetry,
                for example, [1 2 3 4]' -> [1 2 3 4 4 3 2 1]' for N_padded = 8
            'per': Periodic boundary with half-sample symmetry, for example
                [1 2 3 4]' -> [1 2 3 4 1 2 3 4]' for N_padded = 8
            'zero': Zero boundary, for example
                [1 2 3 4]' -> [1 2 3 4 0 0 0 0]' for N_padded = 8
            (default 'symm')
        center: If true, the signal x is centered in the output y, 
            otherwise it is located in the (upper) left corner (default false).

    Output
        y (numeric): The padded signal of size N_padded x K
    
    Description
        The input signal x is padded to give a signal of the size N_padded
        using the boundary conditions specified in boundary.
        This has the advantage of reducing boundary effects when
        computing convolutions by specifying boundary to be 'symm' or 'zero',
        depending on the signal.
        It also allows for convolutions to be calculated on signals smaller
        than the filter was originally defined for.
        Specifically, N_paded does not need to be a multiple of size(x).
        Indeed, if x = [1 2 3 4]', we can have N_padded = 11, which gives
        y = [1 2 3 4 4 3 2 4 3 2 1]'. There is a discontinuity since Npadded
        is not a multiple of 4, but this discontinuity occurs as far from the
        original signal, located in indices 1 through 4, as possible.

    '''
    N_original = x.shape[0]
    
    # Should I do smth spesific with imagenary part during padding?
    has_imag = np.linalg.norm(np.imag(x)) > 0
    
    delta = (N_padded - N_original)
    if center:        
        margins = (int(delta//2), int(delta - delta//2))
    else:
        margins = (0, int(delta))
        
    
    if boundary == 'zero':
        y = np.pad(x, pad_width = [margins, (0,0)], mode = 'constant')
    elif boundary == 'symm':
        y = np.pad(x, pad_width = [margins, (0,0)], mode = 'symmetric')
    elif boundary == 'per':
        y = np.pad(x, pad_width = [margins, (0,0)], mode = 'wrap')
    else:
        print('Padding need to be implemented')
        raise NotImplementedError
    
    return y

def unpad_signal(x, N_unpadded, resolution = 0, center = False):
    '''
    Input
        x: The signal to be unpadded, 2-d array N_padded x K
        resolution: The resolution of the signal (as a power of 2), with
            respect to the original, unpadded version.
        N_unpadded: The size of the original, unpadded version. Combined
            with resolution, the size of the output x_unpadded is given by
            N_unpadded*2*(-resolution) x K
        center: If True, extracts the center part of x_unpadded,
            otherwise extracts the (upper) left corner (default False).
    Output
        x_unpadded: The extracted unpadded signal
        
    Description
        To handle boundary conditions, a signal is often padded using
        pad_signal() before being convolved with conv_sub_1d().
        After this, the padding needs to be removed to recover a regular
        signal.
        This is achieved using unpad_signal(), which takes the padded,
        convolved signal x as input, as well as its resolution relative to
        the original, unpadded version, and the size of this original version.
        Using this, it extracts the coefficients in x that correspond to
        the domain of the original signal.
        If the center flag was specified during pad_signal(), it is specified
        here again in order to extract the correct part.
    '''
    N_padded = x.shape[0]
    
    if not center:
        x_unpadded = x[:int(N_unpadded//(2**resolution)),:]
        
    else:
        delta = (N_padded - N_unpadded)
        margins = (int(delta//2), int(delta - delta//2))
        
        delta2 = N_unpadded - N_unpadded//(2**resolution)
        margins2 = (int(delta2//2), int(delta2 - delta2//2))
        
        x_unpadded = x[margins[0]+margins2[0]:-(margins[1] + margins2[1]),:]
        
    return x_unpadded




def downsample_filter(f, ds_factor = 1):
    '''
    f - fft transform, 2-d array of shape N x K,
        where K is the number of signals 
    ds_factor - The downsampling factor as a power of 2 with respect to f
    '''
    N, signal_count = f.shape
    f = f.T
    
    f_ds = f.reshape((signal_count, int(2**ds_factor), int(N//(2**ds_factor)))
                    ).sum(axis = 1)
    f_ds = f_ds.T
    
    return f_ds



def conv_sub_1d(x_fft, f, ds):
    '''
    Input
        x_fft: The Fourier transform of the signals to be convolved, 2-d array N x K,
            where K is number of signals.
        f: The filter in the frequency domain (= Fourier transform of the filter).
        ds: The downsampling factor as a power of 2 with respect to x_fft
    Output
        y_ds: the filtered, downsampled signalm in the time domain
    '''

    
    # periodize filter so that it has the correct resolution
    # j0 = int(np.log2(f.shape[0]/x_fft.shape[0]))
    # f = f.reshape((f.shape[0]//2**j0, 2**j0)).sum(axis = 1)

    sig_length = x_fft.shape[0]
    end = len(f)
    f = np.concatenate([f[:sig_length//2],
    	[f[sig_length//2]/2 + f[end- sig_length//2]/2],
    	f[end - sig_length//2+1:]])
    
    # represent filter in matrix form
    f = f[:, np.newaxis] + 0*1j
    # print('Inside conv_sub_1d: filter')
    # print(f.shape)
    # print(f[:10,0])

    
    # Calculate Fourier coefficients
    y_fft = f * x_fft
    # print('Inside conv_sub_1d: y_fft')
    # print(y_fft.shape)
    # print(y_fft[:10,0])

    
    # Calculate downsampling factor with respect to y_fft
    # Пока в имплементации у нас всегда y_fft.shape[0]=x_fft.shape[0]
    ds_factor = ds + np.log2(y_fft.shape[0]/sig_length)
    # print('Inside conv_sub_1d: dsj: ', ds_factor)
    
    if ds_factor > 0:
        y_fft = downsample_filter(y_fft, ds_factor = ds_factor)
    elif ds_factor < 0:
        print('Downsampling for ds_factor > 0 need to be implemented')
        raise NotImplementedError
    # print('Inside conv_sub_1d; downsampled y_fft: ')
    # print(y_fft.shape)
    # print(y_fft[:4,0])
    
    # Calculate inverse Fourier transform
    # Use .T to apply transform along columns (coefficients of 1-d signals)
    # Divede to the 2**(ds_factor/2) to get the same signal magnitude in time domain
    # after downsampling in frequency domain (CHECK TAHT WE SHOULD DEVIDE ifft NOT fft)

    x_filtered_ds = mkl_fft.ifft(y_fft.T).T/(2**(ds_factor/2))
    # print('Inside conv_sub_1d; x_filtered_ds: ')
    # print(x_filtered_ds.shape)
    # print(x_filtered_ds[:4,0])
    return x_filtered_ds
