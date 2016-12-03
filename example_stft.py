'''Example for keras stft layer'''
import time
import stft
import numpy as np
import librosa


len_src = 12000*10

src, sr = librosa.load('src/bensound-cute.mp3', sr=12000, duration=10.0)
src = src[:len_src]
src = src[:, np.newaxis]
src = np.hstack((src, src))
print(src.shape)
srcs = np.zeros((16, ) + src.shape)

for ind in range(srcs.shape[0]):
    srcs[ind] = src

specgram = stft.Spectrogram(n_dft=512, input_shape=src.shape, n_hop=256)
print('Source shape: ', srcs.shape)
start = time.time()
outputs = specgram.predict(srcs)
print("Prediction is done. It took %5.3f seconds." % (time.time()-start))
print('Computed STFT shape: ', outputs.shape)
np.save('stft_outputs.npy', outputs)
