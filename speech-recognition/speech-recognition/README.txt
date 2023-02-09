python 3.9 is used.

This program can be called as a LPC processor for a 10-word isolated speech recognition system and mainly follows the 5 steps and realized by the author myself.
1.End-point detection
2.Pre-emphasis -- high pass filtering 
(3a) Frame blocking and (3b) Windowing
4.Feature extraction
(4a)Find Cepstral coefficients by LPC
(i)Auto-correlation analysis
(ii)LPC analysis,
(iii)Find Cepstral coefficients, 
(4b)or find Cepstral coefficients directly
5.Distortion measure calculations


The main.py includes part3-4 and you can directly open it in Pycharm and change the path mentioned in the "readme" inside the python program

Also, you have to install the package mentioned in the "import" section. 

For part4a, you can see the T1 and T2 are in green and red.
For part4b, you can see the Seg1 is between the yellow line and the pink line.

For part 5, you can run the Part5.py and change the path mentioned in the "readme" inside the python program

There are comments in the program files for you to know which part those codes are indicating.

If you need to run lpc10.txt and pre-em, you can find them in the main.py to run.

If you have any questions, please email me directly aaronyuen1222@gmail.com. Thank you.
 #package dependencies： 
import matplotlib.pyplot as plt
from pynverse import inversefunc
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
