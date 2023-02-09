'''Readme,
this is the python program for part5 and all you need is change the paths in line8, 59-64, 86-91
to run it and a 6*6 confusion matrix will be shown'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import matplotlib.pylab as pl
frequency_sampling, audio_signal = wavfile.read("D:/CMSC5707/s9A.wav")

audio_signal = audio_signal[:150000]

features_mfcc = mfcc(audio_signal, frequency_sampling)
#5a///
print(features_mfcc,"features_mfcc",len(features_mfcc))
print('\nMFCC:\nNumber of windows =', features_mfcc.shape[0])
print('Length of each feature =', features_mfcc.shape[1])

features_mfcc = features_mfcc.T
plt.matshow(features_mfcc)
plt.title('MFCC')

filterbank_features = logfbank(audio_signal, frequency_sampling)

print('\nFilter bank:\nNumber of windows =', filterbank_features.shape[0])
print('Length of each feature =', filterbank_features.shape[1])

'''filterbank_features = filterbank_features.T
plt.matshow(filterbank_features)
plt.title('Filter bank')
plt.show()'''

def myDCW(F, R):

    r1, c1 = F.shape
    r2, c2 = R.shape
    distence = np.zeros((r1, r2))
    for n in range(r1):
        for m in range(r2):
            FR = np.power(F[n, :] - R[m, :], 2)
            distence[n, m] = np.sqrt(np.sum(FR)) / c1

    D = np.zeros((r1 + 1, r2 + 1))
    D[0, :] = np.inf
    D[:, 0] = np.inf
    D[0, 0] = 0
    D[1:, 1:] = distence


    for i in range(r1):
        for j in range(r2):
            dmin = min(D[i, j], D[i, j + 1], D[i + 1, j])
            D[i + 1, j + 1] += dmin

    cost = D[r1, r2]
    return cost

#5b
#Getting soundX
frequency_sampling1A, audio_signal1A = wavfile.read("D:/CMSC5707/s1A.wav")
frequency_sampling2A, audio_signal2A = wavfile.read("D:/CMSC5707/s2A.wav")
frequency_sampling5A, audio_signal5A = wavfile.read("D:/CMSC5707/s5A.wav")
frequency_sampling7A, audio_signal7A = wavfile.read("D:/CMSC5707/s7A.wav")
frequency_sampling8A, audio_signal8A = wavfile.read("D:/CMSC5707/s8A.wav")
frequency_sampling9A, audio_signal9A = wavfile.read("D:/CMSC5707/s9A.wav")
audio_signal1A = audio_signal1A[:150000]
audio_signal2A = audio_signal2A[:150000]
audio_signal5A = audio_signal5A[:150000]
audio_signal7A = audio_signal7A[:150000]
audio_signal8A = audio_signal8A[:150000]
audio_signal9A = audio_signal9A[:150000]
features_mfcc1A = mfcc(audio_signal1A, frequency_sampling1A)
print(features_mfcc1A,"features_mfcc1A",len(features_mfcc1A))
features_mfcc2A = mfcc(audio_signal2A, frequency_sampling2A)
print(features_mfcc2A,"features_mfcc2A",len(features_mfcc2A))
features_mfcc5A = mfcc(audio_signal5A, frequency_sampling5A)
print(features_mfcc5A,"features_mfcc5A",len(features_mfcc5A))
features_mfcc7A = mfcc(audio_signal7A, frequency_sampling7A)
print(features_mfcc7A,"features_mfcc7A",len(features_mfcc7A))
features_mfcc8A = mfcc(audio_signal8A, frequency_sampling8A)
print(features_mfcc8A,"features_mfcc8A",len(features_mfcc8A))
features_mfcc9A = mfcc(audio_signal9A, frequency_sampling9A)
print(features_mfcc9A,"features_mfcc9A",len(features_mfcc9A))
'''print('\nMFCC:\nNumber of windows =', features_mfccA.shape[0])
print('Length of each feature =', features_mfccA.shape[1])'''
#Getting soundY
frequency_sampling1B, audio_signal1B = wavfile.read("D:/CMSC5707/s1B.wav")
frequency_sampling2B, audio_signal2B = wavfile.read("D:/CMSC5707/s2B.wav")
frequency_sampling5B, audio_signal5B = wavfile.read("D:/CMSC5707/s5B.wav")
frequency_sampling7B, audio_signal7B = wavfile.read("D:/CMSC5707/s7B.wav")
frequency_sampling8B, audio_signal8B = wavfile.read("D:/CMSC5707/s8B.wav")
frequency_sampling9B, audio_signal9B = wavfile.read("D:/CMSC5707/s9B.wav")
audio_signal1B = audio_signal1B[:150000]
audio_signal2B = audio_signal2B[:150000]
audio_signal5B = audio_signal5B[:150000]
audio_signal7B = audio_signal7B[:150000]
audio_signal8B = audio_signal8B[:150000]
audio_signal9B = audio_signal9B[:150000]
features_mfcc1B = mfcc(audio_signal1B, frequency_sampling1B)
print(features_mfcc1B,"features_mfcc1B",len(features_mfcc1B))
features_mfcc2B = mfcc(audio_signal2B, frequency_sampling2B)
print(features_mfcc2B,"features_mfcc2B",len(features_mfcc2B))
features_mfcc5B = mfcc(audio_signal5B, frequency_sampling5B)
print(features_mfcc5B,"features_mfcc5B",len(features_mfcc5B))
features_mfcc7B = mfcc(audio_signal7B, frequency_sampling7B)
print(features_mfcc7B,"features_mfcc7B",len(features_mfcc7B))
features_mfcc8B = mfcc(audio_signal8B, frequency_sampling8B)
print(features_mfcc8B,"features_mfcc8B",len(features_mfcc8B))
features_mfcc9B = mfcc(audio_signal9B, frequency_sampling9B)
print(features_mfcc9B,"features_mfcc9B",len(features_mfcc9B))
A=[features_mfcc1A,features_mfcc2A,features_mfcc5A,features_mfcc7A,features_mfcc8A,features_mfcc9A]
B=[features_mfcc1B,features_mfcc2B,features_mfcc5B,features_mfcc7B,features_mfcc8B,features_mfcc9B]
#Op score= dist
dist = np.zeros((6,6))
for i in range(0,6):
    for j in range(0, 6):
        dist[i][j] = myDCW(A[i], B[j])
#plotting 5c

import matplotlib.pyplot as plt
import numpy as np
dist = np.rint(dist)

confusion = np.array(dist)

plt.imshow(confusion, cmap=plt.cm.Blues)

indices = range(len(confusion))

plt.xticks(indices, ['1', '2', '5', '7', '8', '9'])
plt.yticks(indices, ['1', '2', '5', '7', '8', '9'])

plt.colorbar()
plt.xlabel('')
plt.ylabel('')
plt.title('')

for first_index in range(len(confusion)):
    for second_index in range(len(confusion[first_index])):
        plt.text(first_index, second_index, confusion[first_index][second_index]).set_fontsize(5)

plt.show()