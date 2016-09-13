# -*- coding: utf-8 -*-
from __future__ import absolute_import

from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.layers import Input, Lambda, merge, Reshape, Permute
from keras.models import Model
from keras import backend as K
# imports for backwards namespace compatibility
import numpy as np
import pdb
import time

import librosa

import stft
from stft import Spectrogram, get_spectrogram_tensors
from stft import Logam_layer


def mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0):
    """Compute the center frequencies of mel bands.(said Librosa)
    `htk` is removed.
    """
    def mel_to_hz(mels):
        """Convert mel bin numbers to frequencies
        """
        mels = np.atleast_1d(mels)

        # Fill in the linear scale
        f_min = 0.0
        f_sp = 200.0 / 3
        freqs = f_min + f_sp * mels

        # And now the nonlinear scale
        min_log_hz = 1000.0                         # beginning of log region (Hz)
        min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
        logstep = np.log(6.4) / 27.0                # step size for log region
        log_t = (mels >= min_log_mel)

        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))

        return freqs

    def hz_to_mel(frequencies):
        """Convert Hz to Mels
        """
        frequencies = np.atleast_1d(frequencies)

        # Fill in the linear part
        f_min = 0.0
        f_sp = 200.0 / 3

        mels = (frequencies - f_min) / f_sp

        # Fill in the log-scale part

        min_log_hz = 1000.0                         # beginning of log region (Hz)
        min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
        logstep = np.log(6.4) / 27.0                # step size for log region

        log_t = (frequencies >= min_log_hz)
        mels[log_t] = min_log_mel + np.log(frequencies[log_t]/min_log_hz) / logstep

        return mels

    ''' mel_frequencies body starts '''
    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = hz_to_mel(fmin)
    max_mel = hz_to_mel(fmax)

    mels = np.linspace(min_mel, max_mel, n_mels)

    return mel_to_hz(mels)


def dft_frequencies(sr=22050, n_dft=2048):
    '''Alternative implementation of `np.fft.fftfreqs` (said Librosa)
    `htk` is removed.
    '''
    return np.linspace(0,
                       float(sr) / 2,
                       int(1 + n_dft//2),
                       endpoint=True)


def mel(sr, n_dft, n_mels, fmin, fmax):
    ''' create a filterbank matrix to combine stft bins into mel-frequency bins
    use Slaney
    librosa.filters.mel
    
    n_mels: numbre of mel bands
    fmin : lowest frequency [Hz]
    fmax : highest frequency [Hz]
        If `None`, use `sr / 2.0`
    '''
    if fmax is None:
        fmax = float(sr) / 2

    # init
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_dft // 2)))

    # center freqs of each FFT bin
    dftfreqs = dft_frequencies(sr=sr, n_dft=n_dft)

    # centre freqs of mel bands
    freqs = mel_frequencies(n_mels + 2,
                            fmin=fmin,
                            fmax=fmax)
    # Slaney-style mel is scaled to be approx constant energy per channel
    enorm = 2.0 / (freqs[2:n_mels+2] - freqs[:n_mels])

    for i in range(n_mels):
        # lower and upper slopes qfor all bins
        lower = (dftfreqs - freqs[i]) / (freqs[i+1] - freqs[i])
        upper = (freqs[i+2] - dftfreqs) / (freqs[i+2] - freqs[i+1])

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper)) * enorm[i]

    return weights


def Melspectrogram(n_dft, input_shape, n_hop=None, border_mode='same', 
                   logamplitude=True, sr=22050, n_mels=128, fmin=0.0, fmax=None):
    ''' Mel-spectrogram keras layer
    sr: sampling rate (used to compute mel-frequency filter banks)
    '''
    if input_shape is None:
        raise RuntimeError('specify input shape')
    Melgram = Sequential()
    # Prepare STFT.
    x, STFT_magnitude = get_spectrogram_tensors(n_dft, 
                                                n_hop=n_hop, 
                                                border_mode=border_mode, 
                                                input_shape=input_shape,
                                                logamplitude=False) 
    # output: (sample, freq (height), time (width))
    stft_model = Model(input=x, output=STFT_magnitude, name='stft') 
    Melgram.add(stft_model)

    # Convert to a proper 2D representation
    if K.image_dim_ordering() == 'th':
        Melgram.add(Reshape((1,) + stft_model.output_shape[1:]))
    else:
        Melgram.add(Reshape(stft_model.output_shape[1:] + (1,)))

    # build a Mel filter
    mel_basis = mel(sr, n_dft, n_mels, fmin, fmax) # (128, 1025) (mel_bin, n_freq)
    n_freq = mel_basis.shape[1]
    mel_basis = mel_basis[:, np.newaxis, :, np.newaxis] # TODO: check if it's a theano convention?
    
    stft2mel = Convolution2D(n_mels, n_freq, 1, border_mode='valid', bias=False, name='stft2mel',
                             weights=[mel_basis])
    stft2mel.tranable = False

    Melgram.add(stft2mel) #output: (None, 128, 1, 375) if theano.
    Melgram.add(Logam_layer())
    # i.e. 128ch == 128 mel-bin, for 375 time-step, therefore,
    if K.image_dim_ordering() == 'th':
        Melgram.add(Permute((2, 1, 3)))
    else:
        Melgram.add(Permute((1, 3, 2)))
    # output dot product of them
    return Melgram


if __name__ == '__main__':

    len_src = 12000*8
    melgram = Melspectrogram(n_dft=512, n_hop=256, input_shape=(len_src, 1), sr=12000)
    
    src, sr = librosa.load('src/bensound-cute.mp3', sr=12000, duration=8.0)  # whole signal    
    src = src[:len_src]
    src = src[:, np.newaxis]
    srcs = np.zeros((16, len_src, 1))

    for ind in range(srcs.shape[0]):
        srcs[ind] = src

    print srcs.shape
    start = time.time()
    outputs = melgram.predict(srcs)
    print "Prediction is done. It took %5.3f seconds." % (time.time()-start)
    print outputs.shape
    pdb.set_trace()
    np.save('melgram_outputs.npy', outputs)











