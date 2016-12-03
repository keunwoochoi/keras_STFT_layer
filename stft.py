# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np
import scipy.signal
from keras.layers.convolutional import Convolution1D
from keras.layers import Input, Lambda, merge, Permute, Reshape
from keras.models import Model
from keras import backend as K


def _get_stft_kernels(n_dft, keras_ver='new'):
    '''Return dft kernels for real/imagnary parts assuming
        the input signal is real.
    An asymmetric hann window is used (scipy.signal.hann).

    Parameters
    ----------
    n_dft : int > 0 and power of 2 [scalar]
        number of dft components.

    keras_ver : string, 'new' or 'old'
        It determines the reshaping strategy.

    Returns
    -------
    dft_real_kernels : np.ndarray [shape=(nb_filter, 1, 1, n_win)]
    dft_imag_kernels : np.ndarray [shape=(nb_filter, 1, 1, n_win)]

    * nb_filter = n_dft/2 + 1
    * n_win = n_dft

    '''
    assert n_dft > 1 and ((n_dft & (n_dft - 1)) == 0), \
        ('n_dft should be > 1 and power of 2, but n_dft == %d' % n_dft)

    nb_filter = n_dft / 2 + 1

    # prepare DFT filters
    timesteps = range(n_dft)
    w_ks = [(2 * np.pi * k) / float(n_dft) for k in xrange(n_dft)]
    dft_real_kernels = np.array([[np.cos(w_k * n) for n in timesteps]
                                  for w_k in w_ks])
    dft_imag_kernels = np.array([[np.sin(w_k * n) for n in timesteps]
                                  for w_k in w_ks])

    # windowing DFT filters
    dft_window = scipy.signal.hann(n_dft, sym=False)
    dft_window = dft_window.reshape((1, -1))
    dft_real_kernels = np.multiply(dft_real_kernels, dft_window)
    dft_imag_kernels = np.multiply(dft_imag_kernels, dft_window)

    if keras_ver == 'old':  # 1.0.6: reshape filter e.g. (5, 8) -> (5, 1, 8, 1)
        dft_real_kernels = dft_real_kernels[:nb_filter]
        dft_imag_kernels = dft_imag_kernels[:nb_filter]
        dft_real_kernels = dft_real_kernels[:, np.newaxis, :, np.newaxis]
        dft_imag_kernels = dft_imag_kernels[:, np.newaxis, :, np.newaxis]
    else:
        dft_real_kernels = dft_real_kernels[:nb_filter].transpose()
        dft_imag_kernels = dft_imag_kernels[:nb_filter].transpose()
        dft_real_kernels = dft_real_kernels[:, np.newaxis, np.newaxis, :]
        dft_imag_kernels = dft_imag_kernels[:, np.newaxis, np.newaxis, :]

    return dft_real_kernels, dft_imag_kernels


def Logam_layer(name='log_amplitude'):
    '''Return a keras layer for log-amplitude.
    The computation is simplified from librosa.logamplitude by
        not having parameters such as ref_power, amin, tob_db.

    Parameters
    ----------
    name : string
        Name of the logamplitude layer

    Returns
    -------
    a Keras layer : Keras's Lambda layer for log-amplitude-ing.
    '''
    def logam(x):
        log_spec = 10 * K.log(K.maximum(x, 1e-10))/K.log(10)
        log_spec = log_spec - K.max(log_spec)  # [-?, 0]
        log_spec = K.maximum(log_spec, -80.0)  # [-80, 0]
        return log_spec

    def logam_shape(shapes):
        '''shapes: shape of input(s) of the layer'''
        # print('output shape of logam:', shapes)
        return shapes

    return Lambda(lambda x: logam(x), name=name,
        output_shape=logam_shape)


def get_spectrogram_tensors(n_dft, input_shape, trainable=False, 
                            n_hop=None, border_mode='same', 
                            logamplitude=True):
    '''Returns two tensors, x as input, stft_magnitude as result.
        x(input) and STFT_magnitude(tensor) (#freq, #time shape)
    These tensors can be use to build a Keras model 
        using Functional API, 
        `e.g., model = keras.models.Model(x, STFT_magnitude)`
        to build a model that does STFT.
    It uses two `Convolution1D` to compute real/imaginary parts of
        STFT and sum(real**2, imag**2). 

    Parameters
    ----------
    n_dft : int > 0 and power of 2 [scalar]
        number of dft components.

    input_shape : tuple (length=2),
        Input shape of raw audio input.
        It should (num_audio_samples, 1), e.g. (441000, 1)

    trainable : boolean
        If it is `True`, the STFT kernels (=weights of two 1d conv layer)
        is set as `trainable`, therefore they are initiated with STFT 
        kernels but then updated. 

    n_hop : int > 0 [scalar]
        number of samples between successive frames.
    
    border_mode : 'valid' or 'same'.
        if 'valid' the edges of input signal are ignored.

    logamplitude : boolean
        whether logamplitude to stft or not


    this is then used in Keras - Functional model API
    STFT_real and STFT_imag is set as non_trainable

    Returns
    -------
    x : input tensor

    STFT_magnitude : STFT magnitude [shape=(None, n_freq, n_frame)]
    '''

    assert trainable in (True, False)

    if n_hop is None:
        n_hop = n_dft / 2

    # get DFT kernels  
    dft_real_kernels, dft_imag_kernels = _get_stft_kernels(n_dft)
    nb_filter = n_dft / 2 + 1

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
    
    STFT_real.trainable = trainable
    STFT_imag.trainable = trainable
    
    STFT_real = Lambda(lambda x: x ** 2, name='real_pow')(STFT_real)
    STFT_imag = Lambda(lambda x: x ** 2, name='imag_pow')(STFT_imag)

    STFT_magnitude = merge([STFT_real, STFT_imag], mode='sum', name='sum')

    if logamplitude:
        STFT_magnitude = Logam_layer()(STFT_magnitude)
    
    # output: (#sample, freq, time)
    STFT_magnitude = Permute((2, 1))(STFT_magnitude) 

    return x, STFT_magnitude


def Spectrogram(n_dft, input_shape, trainable=False, n_hop=None, 
                border_mode='same', logamplitude=True):
    '''A keras model for Spectrogram using STFT

    Parameters
    ----------
    n_dft : int > 0 and power of 2 [scalar]
        number of dft components.

    input_shape : tuple (length=2),
        Input shape of raw audio input.
        It should (num_audio_samples, 1), e.g. (441000, 1)

    trainable : boolean
        If it is `True`, the STFT kernels (=weights of two 1d conv layer)
        is set as `trainable`, therefore they are initiated with STFT 
        kernels but then updated. 

    n_hop : int > 0 [scalar]
        number of audio samples between successive frames.
    
    border_mode : 'valid' or 'same'.
        if 'valid' the edges of input signal are ignored.

    logamplitude : boolean
        whether logamplitude to stft or not

    Returns
    -------
    A keras model that has output shape of (None, n_freq, n_frame)

    '''
    x, STFT_magnitude = get_spectrogram_tensors(n_dft, input_shape=input_shape,
                                                trainable=trainable,
                                                n_hop=n_hop, 
                                                border_mode=border_mode,
                                                logamplitude=logamplitude)
    model = Model(input=x, output=STFT_magnitude)
    model.trainable = trainable
    return model
