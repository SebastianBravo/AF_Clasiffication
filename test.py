import sys
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import pan_tompkins as pt
import neurokit2 as nk
import random
import pywt
from scipy.signal import find_peaks
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.metrics import binary_accuracy
from sklearn.metrics import confusion_matrix
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn import svm
from sklearn.metrics import accuracy_score

# Importar base de datos
# Importar etiquetas 
labels = np.genfromtxt('data/REFERENCE-original.csv',delimiter=',', dtype='str')

# Importar señales con ritmo sinusal normal y AF
normal_idx = np.where(labels[:,1]=='N')[0] # índices ECGs sinusales normales
af_idx = np.where(labels[:,1]=='A')[0] # índices ECGs AF

normal_signals = []
for i in normal_idx:
	normal_signals.append(scipy.io.loadmat(f'data/{labels[i][0]}' + '.mat')['val'].T)

af_signals = []
for i in af_idx:
	af_signals.append(scipy.io.loadmat(f'data/{labels[i][0]}' + '.mat')['val'].T)

# Unir todos los datos
data = normal_signals + af_signals

# Frecuencia de muestreo
fs = 300
signal = data[0]
coeff = pywt.wavedec(signal.reshape(-1,), 'db4', level=6)

plt.figure(0)
plt.plot(signal)
plt.plot(range(len(coeff[6]))*(len(signal)/len(coeff[6])),coeff[6])

fig, axs = plt.subplots(2)
fig.suptitle('Vertically stacked subplots')
axs[0].plot(signal)
axs[1].plot(coeff[6][:,0])
axs[0].grid()
axs[1].grid()