# -*- coding: utf-8 -*-
from __future__ import absolute_import

from keras.layers.convolutional import Convolution1D
from keras.models import Sequential
from keras.layers import Input, Lambda, merge
from keras.models import Model
from keras import backend as K
# imports for backwards namespace compatibility
import numpy as np
import pdb
import time

import librosa


def get_kernels(n_dft, n_hop=None):
    assert n_dft > 1 and ((n_dft & (n_dft - 1)) == 0), \
        ('n_dft should be >1 and power of 2, but n_dft == %d' % n_dft)
    if n_hop is None:
        n_hope = n_dft / 2

    nb_filter = n_dft/2 + 1
    # prepare DFT filters
    timesteps = range(n_dft)
    w_ks = [2*np.pi*k/float(n_dft) for k in xrange(n_dft)]
    dft_real_kernels = np.array([[np.cos(w_k*n) for n in timesteps] for w_k in w_ks])
    dft_imag_kernels = np.array([[np.sin(w_k*n) for n in timesteps] for w_k in w_ks])

    dft_real_kernels = dft_real_kernels[:nb_filter]
    dft_imag_kernels = dft_imag_kernels[:nb_filter]    
    # reshape filter e.g. (5, 8) --> (5, 1, 8, 1) (todo; is this theano convention?)
    dft_real_kernels = dft_real_kernels[:, np.newaxis, :, np.newaxis]
    dft_imag_kernels = dft_imag_kernels[:, np.newaxis, :, np.newaxis]    
    return dft_real_kernels, dft_imag_kernels


def logam(x):
    # using librosa default values.
    # in many cases, they should be batch-normalized per frequency band anyway.
    # also the implementation is ported from 

    ref_power = 1.0
    amin = 1e-10
    top_db = 80.0

    max_db = 10.0 * K.log(K.max(x))/np.log(10)

    log_spec = 10.0 * K.log(K.maximum(x, amin))/np.log(10)
    log_spec = log_spec - max_db
    return log_spec


def Spectrogram(n_dft, n_hop=None,  border_mode='same', input_shape=None,
                logamplitude=True):
    ''' Spectrogram using STFT - using two conv1d layers

    n_dft : length of DFT
    n_hop : hop length of STFT
    input_shape : input_shape, same as keras layers
    amplitude : 'linear': no pre-processing,
                'decibel': 

    '''
    # get DFT kernels  
    dft_real_kernels, dft_imag_kernels = get_kernels(n_dft, n_hop)
    nb_filter = n_dft/2 + 1

    # layers - one for the real, one for the imaginary
    x = Input(shape=input_shape, name='audio_input', dtype='float32')

    STFT_real = Convolution1D(nb_filter, n_dft,
                              subsample_length=n_hop,
                              border_mode=border_mode,
                              weights=[dft_real_kernels],
                              bias=False,
                              name='dft_real',
                              input_shape=input_shape)(x)

    STFT_imag = Convolution1D(nb_filter, n_dft,
                              subsample_length=n_hop,
                              border_mode=border_mode,
                              weights=[dft_imag_kernels],
                              bias=False,
                              name='dft_imag',
                              input_shape=input_shape)(x)
    
    STFT_real.trainable = False
    STFT_imag.trainable = False

    STFT_real = Lambda(lambda x: x ** 2)(STFT_real)
    STFT_imag = Lambda(lambda x: x ** 2)(STFT_imag)

    STFT_magnitude = merge([STFT_real, STFT_imag], mode='sum') # magnitude
    
    if logamplitude:
        # STFT_magnitude = Lambda(lambda x: np.maximum(x, ))(STFT_magnitude)
        STFT_log = Lambda(lambda x: 10*np.log10(x + 1e-10))(STFT_magnitude)
        return Model(input=x, output=STFT_log)
       
    else:
        return Model(input=x, output=STFT_magnitude)


if __name__ == '__main__':

    len_src = 12000*8
    specgram = Spectrogram(n_dft=512, n_hop=128, input_shape=(len_src, 1))
    
    src, sr = librosa.load('src/bensound-cute.mp3', sr=12000, duration=8.0)  # whole signal    
    src = src[:len_src]
    src = src[:, np.newaxis]
    srcs = np.zeros((16, len_src, 1))

    for ind in range(srcs.shape[0]):
        srcs[ind] = src

    print srcs.shape
    start = time.time()
    outputs = specgram.predict(srcs)
    print "Prediction is done. It took %5.3f seconds." % (time.time()-start)
    np.save('outputs.npy', outputs)








