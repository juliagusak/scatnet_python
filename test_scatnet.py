import numpy as np 
import mkl_fft
from functools import partial

from filters import *
from convolution import *
from core import *
from scatutils import *

import matplotlib.pylab as plt


def radsec_to_hertz(values = [1], vice_versa = False):
    converter_mode = 'Hz to rad/sec' if vice_versa else 'rad/sec to Hz'
    print('\n', converter_mode)
    
    
    for v in values:
        if not vice_versa:
            w = v
            freq = w/2/np.pi
        else:
            w = 2*np.pi*v
            freq = v
        k  = w//(2*np.pi)
        print("{} rad/sec ~ {}*pi + 2pi*{}, = {} Hz".format(
        np.round(w,2), np.round((w - 2*np.pi*k)/np.pi, 2),
        k,  np.round(freq, 2)))   

def test_morlet_freq_1d(T = 2**10, Q = 8):
    filt_opt = FiltOptions(Q = Q, T = T,
                                  filter_type = 'morlet_1d',
                                  boundary = 'symm')
    
    print(morlet_freq_1d(filt_opt))


def test_gabor_morletify(N = 128,
                         center = np.array([0, 0.3, np.pi/2, np.pi]),
                         sigma = [3, 3, 5, 0.8]):

    for c, s in zip(np.round(center,2), np.round(sigma, 1)):
        g = gabor(N, c, s)
        plt.plot(g, label = 'gabor, center={}, sigma={}'.format(np.round(c,2), s))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.xticks(np.array([0, np.pi/2, np.pi, 3/2*np.pi, 2*np.pi])/2/np.pi*N,
           [0, '$\pi/2$', '$\pi(-\pi)$', '$3/2\pi(-\pi/2)$', '$2\pi(0)$'])
    plt.title('Filters in Frequency domain')
    plt.show()
    
    row_count = len(center)//2 + len(center)%2
    idxs = [(i,j) for i in range(row_count) for j in range(2)]

    _, ax = plt.subplots(row_count, 2, figsize = (14, 3*row_count))
    
    for (i,j), c, s in zip(idxs, center, sigma):
        g = gabor(N, c, s)
        f = morletify(g, s)
        ax[i][j].plot(g, label = 'gabor, center={}, sigma={}'.format(np.round(c,2), s))
        ax[i][j].plot(f, label = 'morlet, center={}, sigma={}'.format(np.round(c,2), s))
        ax[i][j].legend(bbox_to_anchor=(1, 0.8), loc=4)
        ax[i][j].set_xticks(np.array([0, np.pi/2, np.pi, 3/2*np.pi, 2*np.pi])/2/np.pi*N,
                        [0, '$\pi/2$', '$\pi(-\pi)$', '$3/2\pi(-\pi/2)$', '$2\pi(0)$'])
    plt.show()


def test_downsample_filter(N = 128, ds_factor = 1,
                           center = np.array([0, 0.3, np.pi/2, np.pi]),
                           sigma = [3, 3, 5, 0.8],
                           filter_type = 'gabor_1d'):
    
    row_count = len(center)//2 + len(center)%2
    idxs = [(i,j) for i in range(row_count) for j in range(2)]

    fig, ax = plt.subplots(row_count, 2, figsize = (14, 3*row_count))
    fig.suptitle('{}'.format(filter_type))

    
    for (i,j), c, s in zip(idxs, center, sigma):
        g = gabor(N, c, s)
        if filter_type == 'morlet_1d':
            g = morletify(g, s)
            
        f = downsample_filter(g[:, np.newaxis], ds_factor = ds_factor)
        f = f.ravel()
        
        ax[i][j].plot(g, label = 'filter, center={}, sigma={}'.format(np.round(c,2), s))
        ax[i][j].plot(f, label = 'downsampled filter')
        ax[i][j].legend(bbox_to_anchor=(1, 0.8), loc=4)
        ax[i][j].set_xticks(np.array([0, np.pi/2, np.pi, 3/2*np.pi, 2*np.pi])/2/np.pi*N,
                        [0, '$\pi/2$', '$\pi(-\pi)$', '$3/2\pi(-\pi/2)$', '$2\pi(0)$'])
    plt.show()



def test_filter_bank(return_bank = False):
    T = 2**10
    signal_length = 2**11
    
    filt_opt_bank = [FiltOptions(Q = 8, T = T,
                                  filter_type = 'morlet_1d',
                                  boundary = 'symm'),
                     FiltOptions(Q = 1, T = T,
                                  filter_type = 'morlet_1d',
                                  boundary = 'symm')
                    ]
    
    bank = filter_bank(signal_length, filt_opt_bank)
    
    if return_bank:
        return filt_opt_bank, bank
    else:
        print(bank[0].psi.meta, bank[1].psi.meta)     


def plot_filter_bank(bank=None, filt_opt_bank = None):
    if bank is None:
        filt_opt_bank, bank = test_filter_bank(return_bank=True)

    for k in range(len(bank)):
        print('Bank  {}'.format(k))
        J = bank[k].meta.J
        P = bank[k].meta.P
        N = bank[k].meta.filter_size
        
        center_psi, bw_psi, bw_phi  = morlet_freq_1d(filt_opt_bank[k])        
    
        # plot psi filters
        plt.figure(figsize=(20,4))
        for j1 in range(J):
            f = bank[k].psi.filter[j1]
            plt.plot(f)

        plt.title('Psi filters, number of filters = {} \
            \n (logarithmically spaced band-pass filters)'.format(J), size = 'xx-large')
#         plt.xticks([N/2] + list(center_psi*N/2/np.pi) + [N],
#                    ['$\pi$'] + list(np.round(center_psi,1)) + ['2$\pi$'], size = 'large')
        plt.xticks(np.round(center_psi*N/2/np.pi,1), size = 'large')
        plt.xlabel("Hz", size = 'xx-large')
        plt.show()

        # plot phi filters
        if P > 0:
            f = bank[k].phi.filter[0]

            plt.figure(figsize=(20,4))
            plt.plot(f[:10])
            plt.title('Phi filters, number of filters = {} \
                \n (linearly spaced band-pass filters, cover the remaining part of the spectrum)'.format(P), size = 'xx-large')
            M = N//10
            plt.xticks(np.arange(0, M, M//5),
                       np.round(np.arange(0, M, M//5)*M/2/np.pi, 1), size = 'large')
    #         plt.xticks(np.arange(0, N + N/4, N/4),
    #                    ['{}$\pi$'.format(i) for i in np.arange(0,2.5,0.5)], size = 'large')
            plt.xlabel("Hz", size = 'xx-large')
            plt.show()


def plot_filters(bank=None, filt_opt_bank = None):
    if bank is None:
        filt_opt_bank, bank = test_filter_bank(return_bank=True)
        
    for k in range(2):
        print('Bank {}'.format(k))
        center_psi, bw_psi, bw_phi  = morlet_freq_1d(filt_opt_bank[k])  

        sigma0 = bank[k].meta.sigma0
        sigma_psi = sigma0*np.pi/2./bw_psi
        sigma_phi = sigma0*np.pi/2./bw_phi

        J = bank[k].meta.J
        P = bank[k].meta.P
        print('J = {}, P = {}'.format(J, P))

        for j in range(J)[::J//5]:

            f = bank[k].psi.filter[j]
#             f = np.concatenate([f[:len(f)//2][::-1], f[:len(f)//2]])

            ifft_f = mkl_fft.ifft(f)
            ifft_f = np.real(ifft_f)

            fig, ax = plt.subplots(1, 2, figsize = (24, 4))
            ax[0].plot(np.concatenate([ifft_f[-int(3*sigma_psi[-1]):],
                                       ifft_f[:int(3*sigma_psi[-1])]]))
#             ax[0].plot(ifft_f)

            ax[1].plot(f)
#             ax[1].plot(np.concatenate([f[:len(f)//2][::-1], f[:len(f)//2]]))

            fig.suptitle('Psi filter, j{} = {}'.format(k+1, j), size = 'xx-large')
            ax[0].set_title('Time domain')
            ax[1].set_title('Frequency domain')

            plt.show()
            
        
        if P > 0:
            f = bank[k].phi.filter[0]

            ifft_f = mkl_fft.ifft(f)
            ifft_f = np.real(ifft_f)

            fig, ax = plt.subplots(1, 2, figsize = (24, 4))
            ax[0].plot(np.concatenate([ifft_f[-int(3*sigma_psi[-1]):],
                                       ifft_f[:int(3*sigma_psi[-1])]]))
#             ax[0].plot(ifft_f)
            ax[1].plot(f)
            
            fig.suptitle('Phi filter', size = 'xx-large')
            ax[0].set_title('Time domain')
            ax[1].set_title('Frequency domain')
            
            plt.show()
            


def test_pad_signal():    
    signal = np.array([np.arange(1,4), 10*np.arange(1,4)]).T
    N_padded = 2**3
    
    print('Initial signal: {}, padded signal: {}'.format(
        signal.shape, (N_padded, signal.shape[1])))
    print(signal)
    
    for pad in ['zero', 'symm', 'per']:
        signal_padded = pad_signal(signal, N_padded,
                                   boundary=pad, center = True)
        print('\n {} padding'.format(pad))
        print(signal_padded)


def test_unpad_signal():
    signal = np.array([np.arange(16), 10*np.arange(16)]).T
    N_unpadded = 11
    
    print('\n Initial signal: {}'.format(signal.shape))
    print(signal)
    
    for (res, center) in [(0, 0), (0, 1), (1, 1)]:
        signal_unpadded = unpad_signal(signal, N_unpadded,
                                       resolution = res, center = center)
        print('\n Unpadded signal: {}, N_unpadded = {}, center = {}, resolution = {}'.format(
            signal_unpadded.shape, N_unpadded, center, res))
        print(signal_unpadded)


def test_conv_sub_1d(ds=1):
    
    N = 2**12
    
    # signal - 2-d array N x K, where K - number of signals
    signal = np.random.normal(size = (N, 5))
    signal_fft = mkl_fft.fft(signal.T).T
    
    f = gabor(N, 1, 1)
    
    signal_filtered_ds = conv_sub_1d(signal_fft, f, ds = ds)
    
    _, ax = plt.subplots(1, 2, figsize = (10, 4))
    ax[0].plot(signal[:,0])
    ax[0].set_title('Initial signal: {}'.format(signal.shape))
    
    ax[1].plot(signal_filtered_ds[:,0])
    ax[1].set_title('Filtered downsampled signal: {}'.format(
        signal_filtered_ds.shape))
    plt.show() 


def test_wavelet_1d(T=2**10, N = None, plot = False):
    K = 12
    if N is None:
        N = T*2**3
    filt_opt = FiltOptions(T = T, Q = 8)
    filters =  morlet_filter_bank_1d(N, filt_opt)

   
    for signal, name in zip([np.random.normal(size = (N, K))[:, np.newaxis,:],
                             np.sin([100 + np.arange(N) for k in range(K)]).T[:, np.newaxis,:]],
                            ['Random noise', 'Sine']):
        delim = '---------------------'
        print('{} \n {} signal \n {} \n'.format(delim, name, delim))

        x_phi, x_psi, meta_phi, meta_psi = wavelet_1d(signal, filters)

        print('Signal filtered by  Psi filters and downsampled \n')
        print(meta_psi.j,  type(x_psi))
        x_psi_count = len(x_psi)

        if plot:
            plt.plot(signal[:,:,0])
            plt.ylim(-3,3)
            plt.show()

            _, ax = plt.subplots(x_psi_count//4 + (x_psi_count%4 != 0), 4, figsize = (20,10))
            for i in range(x_psi_count):
                ax[i//4][i%4].plot(x_psi[i][:,:,0])
            plt.show()

            print('Signal filtered by Phi filter and downsampled \n')
            print(meta_phi.j, type(x_phi))
            plt.figure(figsize=(3, 2))
            plt.plot(x_phi[:,:,0])
            plt.ylim(-3,3)
            plt.show()


def test_wavelet_layer_1d(T = 2**10, N = None):
    if N is None:
        N = T*2**3
    filt_opt_bank = [FiltOptions(Q = 8, T = T,
                              filter_type = 'morlet_1d',
                              boundary = 'symm'),
                 FiltOptions(Q = 1, T = T,
                              filter_type = 'morlet_1d',
                              boundary = 'symm')
                ]
    
    bank = filter_bank(N, filt_opt_bank)

    x = np.random.normal(size = (N,1,4))

    U = LayerU(metaU = MetaU(j = np.empty((0,1))), signal = [x])

    U_phi, U_psi = wavelet_layer_1d(U, bank[0])
    
    print('\n Layer 1')
    print('Phi filters output:\n j:{}, number of signals: {}'.format(
        U_phi.meta.j, len(U_phi.signal)))
    print('Psi filters output:\n j:{}, number of signals:{}'.format(
        U_psi.meta.j, len(U_psi.signal)))
    
    
    U_phi2, U_psi2 = wavelet_layer_1d(modulus_layer(U_psi), bank[1])
    
    print('\n Layer 2')
    print('Phi filters output:\n j:{}, number of signals:{}'.format(
        U_phi2.meta.j, len(U_phi2.signal)))
    print('Psi filters output:\n j:{}, number of signals:{}'.format(
        U_psi2.meta.j, len(U_psi2.signal)))

    return U_phi, U_psi, U_phi2, U_psi2


def test_wavelet_factory_1d(T = 2**10, N = None, K = None):
    if N is None:
        N = T*2**3
    if K is None:
        K = 4
        
    filt_opt_bank = [FiltOptions(Q = 8, T = T,
                              filter_type = 'morlet_1d',
                              boundary = 'symm'),
                 FiltOptions(Q = 1, T = T,
                              filter_type = 'morlet_1d',
                              boundary = 'symm')
                ]
    
    scat_opt = ScatOptions()
    Wop, bank  = wavelet_factory_1d(N, filt_opt_bank, scat_opt)    
    
    x = np.random.normal(size = (N, 1, K))
    U= LayerU(metaU = MetaU(j = np.empty((0,1))),
              signal = [x])
    
    for i in range(len(Wop)):
        Uphi, Upsi = Wop[i](U)
        
        print('\n Filter bank {}'.format(i))
        print('Signals filtered with Phi filters: {}')
        print([uphi.shape for uphi in Uphi.signal])
        
        print('Signals filtered with Psi filters: {}')
        print([upsi.shape for upsi in Upsi.signal])



def test_scat(T = 2**10, x = None):
    if x is None:
        N = T*2**3
        K = 40
        x = np.random.normal(size = (N,1,K))

    else:
        N = x.shape[0]
        K = x.shape[2]

    filt_opt_bank = [FiltOptions(Q = 8, T = T,
                              filter_type = 'morlet_1d',
                              boundary = 'symm'),
                 FiltOptions(Q = 1, T = T,
                              filter_type = 'morlet_1d',
                              boundary = 'symm')
                ]
    Wop, bank  = wavelet_factory_1d(N, filt_opt_bank)    
    print(vars(filt_opt_bank[0]))
    print(vars(filt_opt_bank[1]))

    S, Us = scat(x, Wop)
    return S


if __name__=='__main__':
	# test_filter_bank()
	# test_wavelet_1d(plot = False)
	# U_phi, U_psi, U_phi2, U_psi2 = test_wavelet_layer_1d()
	S = test_scat()