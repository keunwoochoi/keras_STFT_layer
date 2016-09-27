# keras_STFT_layer
Do STFT & friends in Keras! For less hassle pre-processing
 * [x] STFT
 * [x] melgram
 * [ ] CQT

## why
Because I am planning to compare the performance of some Convnets while changing parameters of STFT and storing all of them doesn't seem to make sense.

## how
```python
from stft import Spectrogram
import keras

len_src = 12000 * 8 # 8-second signal is your input
model = keras.Sequential()
specgram = Spectrogram(n_dft=512, n_hop=128, input_shape=(len_src, 1))
model.add(specgram)
model.add(BatchNormalization(axis=time_axis)) # recommended
model.add(your_awesome_network)
...

```

## More info
* [See the results - STFT](https://github.com/keunwoochoi/keras_STFT_layer/blob/master/stft.ipynb)
* [See the results - Melgram](https://github.com/keunwoochoi/keras_STFT_layer/blob/master/melgram.ipynb)
