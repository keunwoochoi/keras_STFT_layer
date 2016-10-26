import time
import melgram
import numpy as np
import librosa


len_src = 12000*8
melgram = melgram.Melspectrogram(n_dft=512,
                                 input_shape=(len_src, 1), 
                                 trainable=False,
                                 sr=12000)

src, sr = librosa.load('src/bensound-cute.mp3', sr=12000, duration=8.0)
src = src[:len_src]
src = src[:, np.newaxis]
srcs = np.zeros((16, len_src, 1))

for ind in range(srcs.shape[0]):
    srcs[ind] = src

print('Source shape: ', srcs.shape)
start = time.time()
outputs = melgram.predict(srcs)
print("Prediction is done. It took %5.3f seconds." % (time.time()-start))
print('Computed melgram shape: ', outputs.shape)
np.save('melgram_outputs.npy', outputs)

