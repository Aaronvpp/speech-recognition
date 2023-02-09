'''readme: for running part 4, you need to change all the paths in line 69 which read the .wav file in this file
you can run it after importing the needed packages mentioned below. Don't change the path in line 106'''

import math
import wave
import pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import stdio as stdio
from numpy.fft._pocketfft import _get_forward_norm
from scipy import signal
from scipy.io import wavfile
from scipy.io.wavfile import read
from numpy.core import integer, empty, arange, asarray, roll
from numpy.core.overrides import array_function_dispatch, set_module
import functools
from numpy.core.multiarray import normalize_axis_index
from numpy.core import overrides
from cmath import sqrt

def fft_frequency(n, d=1.0):
    val = 1.0 / (n * d)
    results = empty(n, int)
    N = (n-1)//2 + 1
    p1 = arange(0, N, dtype=int)
    results[:N] = p1
    p2 = arange(-(n//2), 0, dtype=int)
    results[N:] = p2
    return results * val
def ZeroCR(waveData,frameSize,overLap):
    wlen = len(waveData)
    step = frameSize - overLap
    frameNum = math.ceil(wlen/step)
    zcr = np.zeros((frameNum,1))
    for i in range(frameNum):
        curFrame = waveData[np.arange(i*step,min(i*step+frameSize,wlen))]
        curFrame = curFrame - np.mean(curFrame) # zero-justified
        zcr[i] = sum(curFrame[0:-1]*curFrame[1::]<=0)
    return zcr
def Energy(waveData,frameSize,overLap):
    wlen1 = len(waveData)
    step1 = frameSize - overLap
    frameNum1 = math.ceil(wlen1/step1)
    ene = np.zeros((frameNum1,1))
    for i in range(frameNum1):
        curFrame1 = waveData[np.arange(i*step1,min(i*step1+frameSize,wlen1))]
        curFrame1 = curFrame1 - np.mean(curFrame1)
        ene[i] = sum(curFrame1[0:-1]*curFrame1[0:-1])
    return ene
def findMaxEnergy(waveData,frameSize,overLap):
    wlen1 = len(waveData)
    step1 = frameSize - overLap
    frameNum1 = math.ceil(wlen1/step1)
    ene = np.zeros((frameNum1,1))
    for i in range(frameNum1):
        curFrame1 = waveData[np.arange(i*step1,min(i*step1+frameSize,wlen1))]
        curFrame1 = curFrame1 - np.mean(curFrame1)
        ene[i] = sum(curFrame1[0:-1]*curFrame1[0:-1])
    index_max = np.argmax(ene)
    for i in range(frameNum1):
        if(i==(index_max)):
            Segment1=waveData[np.arange(i*step1,min(i*step1+frameSize,wlen1))]



    return index_max, Segment1


fs, data = wavfile.read('D:/CMSC5707/s9A.wav')            # reading the file

wavfile.write('s1A_1.wav', fs, data[:, 0])   # saving first column which corresponds to channel 1
wavfile.write('s1A_2.wav', fs, data[:, 1])
samplerate, data = read('s1A_1.wav')

duration = len(data)/samplerate
time = np.arange(0,duration,1/samplerate) #time vector
data = data/10000
plt.plot(time,data)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Point3.wav')
plt.show()
#energy of the data

#energy=data*data
#plt.plot(time,abs(energy))
#plt.xlabel('Time [s]')
#plt.ylabel('Energy')
#plt.title('Point4.wav')
#plt.show()
#data=data*10000

#zero crossing point calculation
'''zerocrossing = 0
for i in range(1, len(data)):
    if ((data[i-1]) > 0 and data[i] < 0):
        zerocrossing += 1
    if ((data[i-1]) < 0 and data[i] > 0):
        zerocrossing += 1
    if zerocrossing > 6:
        print(i)
        break'''


# read wave file and get parameters.
fw = wave.open('s1A_1.wav','rb')#don't change the path here
params = fw.getparams()

nchannels, sampwidth, framerate, nframes = params[:4]
str_data = fw.readframes(nframes)
wave_data = np.fromstring(str_data, dtype=np.short)
wave_data.shape = -1, 1
#wave_data = wave_data.T
fw.close()

# calculate Zero Cross Rate
frameSize = 882
overLap = 441
zcr = ZeroCR(wave_data,frameSize,overLap)
ene = Energy(wave_data,frameSize,overLap)

# plot the wave
time1 = np.arange(0, len(wave_data)) * (1.0 / framerate)
time2 = np.arange(0, len(zcr)) * (len(wave_data)/len(zcr) / framerate)
time3 = np.arange(0, len(ene)) * (len(wave_data)/len(ene) / framerate)


#find the starting point of an audio signal

#for x, y in zcr,ene:
#    if (x > 30):
#        print(idx, x)
#        break
for indexp, (i,j) in enumerate(zip(zcr,ene)):
    if (j > 1.88776508e+08 and i > 8) and (ene[indexp+1] > 1.88776508e+08 and zcr[indexp+1] > 8) and (ene[indexp+2] > 1.88776508e+08 and zcr[indexp+2] > 8):
        print(i,j)

        break
'''print(indexp,"indexp")'''
'''for index, j in enumerate(ene):
    if (j > 1.88776508e+08):
        print(index,j)
        pointer = index+1
        if(ene[pointer]>1.88776508e+08):
            print(pointer, ene[pointer])
            pointer = pointer + 1
            if (ene[pointer] > 1.88776508e+08):
                print(pointer, ene[pointer])
                for index2, i in enumerate(zcr, start=index):
                    if (i > 8):
                        print(index2, i)
                        pointer2 = index2 + 1
                        if (zcr[pointer2] > 8):
                            print(pointer2, zcr[pointer2])
                            pointer2 = pointer2 + 1
                            if (zcr[pointer2] > 8):
                                print(pointer2, zcr[pointer2])
                                break
                            else
                    else:break
                break'''


'''print(i)'''
'''print(j)'''
for idx, p in enumerate(ene):
    if (p==j):
        #print(idx, p)
        break
#print(idx)
#find the ending point of an audio signal
#print(indexp)
indexp1=indexp+1
#print(indexp1,"indexp1")
for indexp1, (c,k) in enumerate(zip(zcr,ene),start=80):
    if (k < 1.88776508e+08 and c < 50) and (ene[indexp1+1] < 1.88776508e+08 and zcr[indexp1+1] < 50) and (ene[indexp1+2] < 1.88776508e+08 and zcr[indexp1+2] < 50):
        print(c,k)
        break
#print(indexp1,"indexp1")
#print(c)
#print(k)
#for idx1, p1 in enumerate(ene):
#    if (p1==k):
#        print(idx1, p1)
#        break
idx1=indexp1
#print(idx1)
findmaxenergy,Segment1=findMaxEnergy(wave_data,frameSize,overLap)
print(findmaxenergy,Segment1,len(Segment1),"findmaxindex","Seg1","Length of the segment1")
#plotting
pl.subplot(311)
pl.plot(time1, wave_data)
plt.vlines([time3[idx]], -25000,25000,linestyles='dashed', colors='green' )
plt.vlines([time3[idx1]], -25000,25000,linestyles='dashed', colors='red' )
plt.vlines([time3[findmaxenergy]], -25000,25000,linestyles='dashed', colors='yellow' )
plt.vlines([time3[findmaxenergy+1]], -25000,25000,linestyles='dashed', colors='pink' )
pl.ylabel("Amplitude")
pl.subplot(312)
pl.plot(time2, zcr)
plt.vlines([time2[idx]], 0,250,linestyles='dashed', colors='green')
plt.vlines([time2[idx1]], 0,250,linestyles='dashed', colors='red')
pl.ylabel("ZCR")
pl.xlabel("time (seconds)")
pl.subplot(313)
pl.plot(time3, ene)
plt.vlines([time3[idx]], 0,1.34985821e+11,linestyles='dashed', colors='green' )
plt.vlines([time3[idx1]], 0,1.34985821e+11,linestyles='dashed', colors='red' )
pl.xlabel('Time [s]')
pl.ylabel('Energy')
pl.show()
#sample_rate, samples = wavfile.read('D:/CMSC5707/s1A.wav')
#frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
#print(1)

#plt.pcolormesh(times, frequencies, spectrogram)
#plt.imshow(spectrogram)
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [sec]')
#plt.show()
#testing for fft
#4c

def fft_own(segment1):
    n = int(len(segment1))
    x = segment1
    y = [0] * (n)
    # define j so that it can be used later
    j = sqrt(-1)
    for k in range(0, n):
        for l in range(0, n):
            # Carry out the summation
            y[k] += x[l] * (np.exp((-np.pi * j * l * k * 2) / n))

    for i in range(0, n):
        print(f"y{i} = {y[i]}")
    return y
# Number of sample points
N = len(Segment1)
# sample spacing
T = samplerate
x = np.linspace(0.0, N*T, N, endpoint=False)
y = Segment1
yf = fft_own(y)
print(yf,"yfyfyf")
xf = fft_frequency(N, T)[:N//2]
import matplotlib.pyplot as plt
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.show()

#4d  pre-emphasis signal (Pem_Seg1) of Seg1
time4=np.arange(0, len(Segment1))
input=Segment1
output = [None]*(len(Segment1)+2)
print(input[0])
output[0]=input[0]
for k in range(1, (len(input)-1)):
     output[k]=input[k] - 0.95*input[k-1]
print(output,"output")
time5=np.arange(0, len(output))
pl.subplot(211)
pl.plot(time4, Segment1)
pl.title("Segment1")
pl.xlabel('Time [s]')
pl.ylabel('Amplitude')
pl.subplot(212)
pl.plot(time5, output)
pl.title("Pem_Seg1")
pl.xlabel('Time [s]')
pl.ylabel('Amplitude')
pl.show()

#4e Find the LPC-10 parameters
from pynverse import inversefunc
ORDER = 10
WINDOW=len(Segment1)
auto_coeff = np.zeros((ORDER+1,ORDER+1))
sig = Segment1

for i in range(0,ORDER+1):
    auto_coeff[i]=0.0
    for j in range(i,(WINDOW)):
       auto_coeff[i]+= sig[j]*sig[j-i]
auto_coeff=auto_coeff[:,1]
print(auto_coeff,"r0...r10")
matrixcoeff = np.array([[auto_coeff[0],auto_coeff[1],auto_coeff[2],auto_coeff[3],auto_coeff[4],auto_coeff[5],auto_coeff[6],auto_coeff[7],auto_coeff[8],auto_coeff[9]],
             [auto_coeff[1],auto_coeff[0],auto_coeff[1],auto_coeff[2],auto_coeff[3],auto_coeff[4],auto_coeff[5],auto_coeff[6],auto_coeff[7],auto_coeff[8]],
             [auto_coeff[2],auto_coeff[1],auto_coeff[0],auto_coeff[1],auto_coeff[2],auto_coeff[3],auto_coeff[4],auto_coeff[5],auto_coeff[6],auto_coeff[7]],
             [auto_coeff[3],auto_coeff[2],auto_coeff[1],auto_coeff[0],auto_coeff[1],auto_coeff[2],auto_coeff[3],auto_coeff[4],auto_coeff[5],auto_coeff[6]],
             [auto_coeff[4],auto_coeff[3],auto_coeff[2],auto_coeff[1],auto_coeff[0],auto_coeff[1],auto_coeff[2],auto_coeff[3],auto_coeff[4],auto_coeff[5]],
             [auto_coeff[5],auto_coeff[4],auto_coeff[3],auto_coeff[2],auto_coeff[1],auto_coeff[0],auto_coeff[1],auto_coeff[2],auto_coeff[3],auto_coeff[4]],
             [auto_coeff[6],auto_coeff[5],auto_coeff[4],auto_coeff[3],auto_coeff[2],auto_coeff[1],auto_coeff[0],auto_coeff[1],auto_coeff[2],auto_coeff[3]],
             [auto_coeff[7],auto_coeff[6],auto_coeff[5],auto_coeff[4],auto_coeff[3],auto_coeff[2],auto_coeff[1],auto_coeff[0],auto_coeff[1],auto_coeff[2]],
             [auto_coeff[8],auto_coeff[7],auto_coeff[6],auto_coeff[5],auto_coeff[4],auto_coeff[3],auto_coeff[2],auto_coeff[1],auto_coeff[0],auto_coeff[1]],
             [auto_coeff[9],auto_coeff[8],auto_coeff[7],auto_coeff[6],auto_coeff[5],auto_coeff[4],auto_coeff[3],auto_coeff[2],auto_coeff[1],auto_coeff[0]]
             ])

print(matrixcoeff,"matrixcoeff")
rightmatrix=auto_coeff[0:10]
print(rightmatrix,"rightmarix")
print(np.linalg.inv(matrixcoeff))
LPC_matrix=np.matmul((np.linalg.inv(matrixcoeff)),rightmatrix)
print("LPC_matrix=",LPC_matrix)
'''coeff=auto_coeff
a = np.zeros((ORDER+1,ORDER+1))
print(a[1,1])
#if ( coeff[0] == 0) : coeff[0] = 1.0E-30
E = coeff[0]
for i in range(1,ORDER+1):
    sum=0.0
    for j in range(1,i):
        sum+= a[j,i-1]*coeff[i-j]
    K=(coeff[i]-sum)/E
    a[i,i]=K
    E*=(1-K*K)
    for j in range(1,i):
        a[j,i]=a[j,i-1]-K*a[i-j,i-1]

for i in range(1,ORDER+1):
    coeff[i]=a[i,ORDER]
print(coeff)'''




