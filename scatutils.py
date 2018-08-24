import numpy as np 
import copy

from core import *

class MetaFlattenScat():
    def __init__(self, bandwidth = None, resolution = None,
                 j = None, order = None):
        self.bandwidth = bandwidth
        self.resolution = resolution
        self.j = j
        self.order = order

class FlattenScat():
    def __init__(self, meta = None, signal = None):
        self.meta = meta
        self.signal = signal

def flatten_scat(S):
    '''
    Put scattering coefficients of all layers together

    Input
        S: A scattering representation (list of U_phi outputs from all layers).

    Output
        S: The same scattering representation, but flattened into one layer.
            As a result, meta fields from different orders are concatenated.
            Since different orders have meta fields of different sizes,
            such as meta.j,the meta fields are filled with -1's
            where necessary.
    '''
    Y = FlattenScat(meta = MetaFlattenScat())
    
    Y.meta.bandwidth = np.concatenate([s.meta.bandwidth for s in S])
    Y.meta.resolution = np.concatenate([s.meta.resolution for s in S])
    
    
    M = len(S)-1
    Y.meta.order = np.concatenate([[m]*len(S[m].signal) for
                                   m in range(M+1)])
    
    Y.signal = np.concatenate([s.signal for s in S])
    
    Y.meta.j = []
    for m in range(M+1):
        if m == 0:
            Y.meta.j.append(-1*np.ones((M,len(S[0].signal))))
        elif m < M:
            Y.meta.j.append(np.concatenate([S[m].meta.j,
                                            -1*np.ones((M-m,len(S[m].signal)))
                                           ]))
        else:
            Y.meta.j.append(S[m].meta.j)

    Y.meta.j = np.concatenate(Y.meta.j, axis = 1)
    
    return Y


def format_scat(S, format_type = 'table'):
    
    '''
    Input
        S: The scattering representation to be formatted.
        format_type: The desired format. Can be either 'raw',
        'order_table' or 'table' (default 'table').
    Output
        out: The scattering representation in the desired format (see below).
        meta: Properties of the scattering nodes in out.

    Description
        Three different formats are available for the scattering transform:
           'raw': Does nothing, just return S. The meta structure is empty.
           'order_table': For each order, creates a table of scattering
              coefficients with scattering index running along the first dimen-
              sion, time/space along the second, and signal index along the
              third. The out variable is then a cell array of tables, while
              the meta variable is a cell array of meta structures, each
              corresponding to the meta structure for the given order.
           'table': Same as 'order_table', but with the tables for each order
              concatenated into one table, which is returned as out. Note that
              this requires that each order is of the same resolution, that is
              that the lowpass filter phi of each filter bank is of the same
              bandwidth. The meta variable is one meta structure formed by con-
              catenating the meta structure of each order and filling out with
              -1 where necessary (the j field, for example).
    '''
    
    if format_type == 'raw':
        out = S
        meta = []
        
    elif format_type == 'table':
        S = flatten_scat(S)
        out = S.signal
        meta = S.meta
        
    else:
        out = [np.array(s.signal) for s in S]
        meta = [s.meta for s in S]
        
    return out, meta


    # Do we need to normalize S[0], S[1]?
def renorm_scat(S, epsilon=2**(-20), min_order=2):
    '''
    Input
        S: A scattering transform.

    Output
        S: The scattering transform with second- and higher-order coefficients
           divided by their parent coefficients.
    '''
    Y = copy.deepcopy(S)
    
    for m in np.arange(len(Y)-1, min_order-1, -1):
        for p2 in range(len(Y[m].signal)):
            j = Y[m].meta.j[:, p2]
            
            # p1 - index of the path from the layer S{m-1}, such that this path
            # represents the begining of path j from the layer S{m} 
            p1 = np.where(np.sum([Y[m-1].meta.j[l] == j[l] for l in range(m-1)],
                                 axis = 0)==m-1)[0][0]
            sub_multiplier = 2**(Y[m].meta.resolution[p2]/2)
            
            ds = np.log2(len(S[m-1].signal[p1])/len(S[m].signal[p2]))

            if ds >= 0:
                parent = Y[m-1].signal[p1][::int(2**ds)]*2**(ds/2)
            else:
                print('The case (ds < 0) is not implemented yet')
                raise NotImplementedError
                
            Y[m].signal[p2] = Y[m].signal[p2]/(parent + epsilon*sub_multiplier)
    return Y      
    

def log_scat(S, epsilon = 2**(-20)):
    Y = copy.deepcopy(S)
    for m in range(len(Y)):
        for p1 in range(len(Y[m].signal)):
            res = Y[m].meta.resolution[p1]
            
            sub_multiplier = 2**(res/2)
            Y[m].signal[p1] = np.log(np.abs(Y[m].signal[p1]) + epsilon*sub_multiplier)
    return Y  



def concatenate_freq(S, format_type = 'table'):
    '''
    Concatenates first frequencies into tables
    
    Input
        X (struct or cell): The scattering layer to process, or a cell array of
           such scattering layers. Often S or U outputs of SCAT.
        fmt (char, optional): Either 'table' or 'cell'. Describes how grouped
           coefficients are assembled. See Description for more details (default
           'table').
    Output
        Y (struct or cell): The same scattering layers, with all coefficients 
           that only differ by first frequency lambda1 grouped together.
    Description
        In order to perform operations along frequency, or in the time-frequency
        plane, scattering coefficients need to be grouped together and concate-
        nated along the first frequency axis, lambda1. Specifically, all coef-
        ficients that have the same frequencies lambda2, lambda3, etc are grouped
        together. For the first order, this means that we only have one group,
        whereas in the second order, we have one group for each lambda2, and so
        on.

        Each signal in the input is of the form Nx1xK, where N is the number of
        time samples and K is the number of signals that are processed simulta-
        neously. Consider one group as described above, containing P coeffi-
        cients, with all coefficients having the same number of time samples. 
        This is the case, for example, when the input X is a scattering transform
        output S. Here, the coefficients can be concatenated into a single table
        of dimension NxPxK. If the fmt parameter is set to 'table', this is in-
        deed what happens. The P coefficients in X are therefore replaced by
        one table in Y of the dimension described above. If fmt equals 'cell',
        a cell array is created instead of a table, containing each of the sig-
        nals in the group. In both cases, frequencies lambda1 are arranged in 
        order of decreasing frequency (increasing scale).

        To preserve the meta fields of the original coefficients, they are copied
        into the output structure. They are ordered in order of increasing group
        index, so that if the first group contains P coefficients, the first
        P columns of a given meta field corresponds to the coefficients of the 
        first group, and so on. Within each group, the columns are ordered in 
        order of decreasing lambda1, just like within the groups themselves.
    '''
    Y = [None]*len(S)
    
    for m in range(len(S)):
        if m == 0:
            Y[0] = LayerU(metaU = MetaU(bandwidth = S[0].meta.bandwidth,
                                  resolution = S[0].meta.resolution,
                                  j = S[0].meta.j),
                          signal = S[0].signal)
        
        else:
            print('m: ', m)
            Y[m] = LayerU(metaU = MetaU(bandwidth = [],
                                  resolution = [],
                                  j = []),
                          signal = [])

            # Form equivalences between coefficients with the same j1 and group
            # them together
    #         if 'fr_j' not in vars(S[m].meta).keys():
    #             idx_set = set([tuple(row) for row in S[m].meta.j[1:,:].T])
    #         else:
    #             idx_set = set(np.concatenate([Y[m].meta.j[1:,:],
    #                                           S[m].meta.j[1:,:]]).T)

            # convert tuples of indexes to 1-d indexes
            idx_set = sorted(set([tuple(row) for row in S[m].meta.j[1:,:].T]))
            idx_mapping = dict(zip(idx_set, np.arange(len(idx_set))))

            assigned_idxs = np.array(list(
                map(lambda row: idx_mapping[tuple(row)], S[m].meta.j[1:,:].T)
            ))
#             print(assigned_idxs)

            # 1D coefficients are of the form Nx1xK, where N is the number of 
            # time samples and K is the number of signals. Permute them so we
            # have  NxKx1, instead, this simplifies creating tables later.


            signals = np.array(S[m].signal)[:, :, -1, :]

            for k in range(np.max(assigned_idxs)+1):
                size_original = signals[0].shape[:-1]

                # Select the coefficients belonging to the current group
                ind = np.where(assigned_idxs == k)[0]
                print('k: ', k)

                # Each 'signal' being a table of size NxK, as described above, and
                # the cell array being arranged horizontally, the following con-
                # catenates all the signals of indices ind into a table of size
                # Nx(KP), where P is the number of coefficients in the current 
                # group.
                # Put the frequency in the first dimension and the signal index 
                # in the last dimension, giving PxNxK.

                if format_type == 'table':
                    nsignal = signals[ind]
                else:
                    nsignal = signals[ind]

                print(nsignal.shape)
                print(S[m].meta.j[:,ind])
                Y[m].signal.extend(nsignal)

                Y[m].meta.bandwidth.extend(np.array(S[m].meta.bandwidth)[ind])
                Y[m].meta.resolution.extend(np.array(S[m].meta.resolution)[ind])
                Y[m].meta.j.append(np.concatenate([S[m].meta.j[:,ind]]))

            Y[m].meta.j = np.concatenate(Y[m].meta.j, axis = 1)
    return Y  