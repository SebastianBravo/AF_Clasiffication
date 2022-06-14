import sys
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import pan_tompkins as pt
import neurokit2 as nk
import random
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

# Balancear clases:
# Se tienen 771 muestras de AF y 5154 muestras de Sinusal Normal
# Se temona 771 muestras ritmo Sinusal normal
# random.shuffle(normal_signals)
#normal_signals = normal_signals[0:len(af_signals)]

# Unir todos los datos
data = normal_signals + af_signals

# Frecuencia de muestreo
fs = 300
signal = data[3]
_, rpeaks = nk.ecg_peaks(signal.reshape(-1,), method='nabian2018', sampling_rate=fs)
r_times = rpeaks['ECG_R_Peaks']
r_times = r_times[~np.isnan(r_times)]

plt.plot(signal)
plt.plot(r_times,signal[r_times.astype(int)], "x")
plt.xlim(0,len(signal))
plt.grid()
plt.show()

# Extracción de características 
# [Covarianza intervalo RR, Desviación estándar intervalo RR, diferencia RR max y RR min, Relación número ondas p y ondas R]
X = np.zeros((len(data),4))

for i in range(len(data)):
	signal = data[i].reshape(-1,)

	# Extracción tiempo picos ondas R, tiempo picos ondas P, inicio intervalo PR y fin intervalo PR 
	_, rpeaks = nk.ecg_peaks(signal, sampling_rate=fs, method='nabian2018')

	# Tiempos ondas R
	r_times = rpeaks['ECG_R_Peaks']
	r_times = r_times[~np.isnan(r_times)]

	# Intervalos RR
	rr_times = (r_times[1:]-r_times[:-1])

	# Covaraianza intervalo RR
	rr_cov = np.cov(rr_times/fs)

	# Desviación estandar intervalo RR
	rr_std = np.std(rr_times/fs)

	# Diferencia del promedio de los 5 RR max y el promedio de los 5 RR min
	rr_diff =(np.mean(np.sort(rr_times/fs)[-5:])) - (np.mean(np.sort(rr_times/fs)[:5]))

	# Tiempos ondas P
	p_times = np.zeros(len(rr_times),)

	for j in range(len(r_times)-1):
		inicio = int(r_times[j+1] - 0.3*rr_times[j])
		fin = int(r_times[j+1] - 0.1*rr_times[j])

		signal_p = np.zeros(len(signal),)
		signal_p[inicio:fin] = signal[inicio:fin]

		p_times[j] = np.argmax(signal_p) if signal[np.argmax(signal_p)]>3*np.mean(signal) else None

	p_times = p_times[~np.isnan(p_times)]

	# Relación número ondas p y ondas R
	# p_r_rel = len(p_times)/len(r_times)
	p_r_rel = len(r_times)

	X[i] = np.array([rr_cov,rr_std,rr_diff,p_r_rel])

# Creacíon matrices de entrenamiento y prueba
# Etiquetas 0 == Ritmo Sinusal Normal, 1 == AF
y = np.concatenate((np.zeros(len(normal_signals),), np.ones(len(af_signals))),axis=0)

# Conctanear etiquetas:
X = np.concatenate((X,y.reshape(-1,1)), axis=1)

# Segmentación de datos de datos en entrenamiento (80%) y prueba (20%)
training_matrix, testing_matrix = model_selection.train_test_split(X,test_size = int(0.2*len(X)),train_size = int(0.8*len(X)))

# Etiquitas conjuntos de entrenamiento y prueba:
y_train = training_matrix[:,4]
y_test = testing_matrix[:,4]

# Remoción de etiquetas de conjuntos de entrenamiento y prueba:
training_matrix = training_matrix[:,:-1]
testing_matrix = testing_matrix[:,:-1]

# Número de atributos
d = np.size(training_matrix, axis=1)

# Validación cruzada para tener (Validación (20% del total de datos o 25 % del conjunto de entrenamiento))
cv = KFold(n_splits=4, random_state=1, shuffle=True)

# Arquitecturas de MLP
# d-8-1
dnn_1 = Sequential()
dnn_1.add(Dense(d, activation='sigmoid', input_shape=(d,)))
dnn_1.add(Dense(8, activation='sigmoid'))
dnn_1.add(Dense(1, activation='sigmoid'))

# d-16-8-1
dnn_2 = Sequential()
dnn_2.add(Dense(d, activation='sigmoid', input_shape=(d,)))
dnn_2.add(Dense(16, activation='sigmoid'))
dnn_2.add(Dense(8, activation='sigmoid'))
dnn_2.add(Dense(1, activation='sigmoid'))

# d-24-16-1
dnn_3 = Sequential()
dnn_3.add(Dense(d, activation='sigmoid', input_shape=(d,)))
dnn_3.add(Dense(24, activation='sigmoid'))
dnn_3.add(Dense(16, activation='sigmoid'))
dnn_3.add(Dense(1, activation='sigmoid'))

# Configuración de hiperparámetros adicionales: 
dnn_1.compile(optimizer = 'adam', metrics=[binary_accuracy], loss = 'mean_squared_error')
dnn_2.compile(optimizer = 'adam', metrics=[binary_accuracy], loss = 'mean_squared_error')
dnn_3.compile(optimizer = 'adam', metrics=[binary_accuracy], loss = 'mean_squared_error')

#Construcción del modelo con transformación OVA
svm_ova_1 = svm.SVC(gamma = 'auto',degree = 2,kernel = 'poly',decision_function_shape='ovr',verbose=1)
svm_ova_2 = svm.SVC(gamma = 'auto',degree = 3,kernel = 'poly',decision_function_shape='ovr',verbose=1)
svm_ova_3 = svm.SVC(gamma = 'auto',degree = 4,kernel = 'poly',decision_function_shape='ovr',verbose=1)


# Desempeños
acc_dnn_1 = []
acc_dnn_2 = []
acc_dnn_3 = []

acc_svm_1 = []
acc_svm_2 = []
acc_svm_3 = []

# Indices de los conjuntos con que se entrena
train_idx = []
val_idx = []

#Entrenamiento del modelo con validación cruzada
for train_indices, val_indices in cv.split(training_matrix):
	train_idx.append(train_indices)
	val_idx.append(val_indices)
    
    # Entrenamiento
    # DNN
	history_dnn_1 = dnn_1.fit(training_matrix[train_indices], y_train[train_indices], epochs = 300, 
								verbose = 1, workers = 8, use_multiprocessing = True,
								validation_data = (training_matrix[val_indices], y_train[val_indices]))
	history_dnn_2 = dnn_2.fit(training_matrix[train_indices], y_train[train_indices], epochs = 300, 
								verbose = 1, workers = 8, use_multiprocessing = True,
								validation_data = (training_matrix[val_indices], y_train[val_indices]))
	history_dnn_3 = dnn_3.fit(training_matrix[train_indices], y_train[train_indices], epochs = 300, 
								verbose = 1, workers = 8, use_multiprocessing = True,
								validation_data = (training_matrix[val_indices], y_train[val_indices]))

	# SVM
	svm_ova_1.fit(training_matrix[train_indices], y_train[train_indices])
	svm_ova_2.fit(training_matrix[train_indices], y_train[train_indices])
	svm_ova_3.fit(training_matrix[train_indices], y_train[train_indices])
	
	# Validación
	y_out_svm_1 = svm_ova_1.predict(training_matrix[val_indices]) 
	y_out_svm_2 = svm_ova_2.predict(training_matrix[val_indices]) 
	y_out_svm_3 = svm_ova_3.predict(training_matrix[val_indices]) 

	# Errores por cada validación realizada
	acc_dnn_1.append(history_dnn_1.history['val_binary_accuracy'][-1])
	acc_dnn_2.append(history_dnn_2.history['val_binary_accuracy'][-1])
	acc_dnn_3.append(history_dnn_3.history['val_binary_accuracy'][-1])
	
	acc_svm_1.append(accuracy_score(y_train[val_indices], y_out_svm_1))
	acc_svm_2.append(accuracy_score(y_train[val_indices], y_out_svm_2))
	acc_svm_3.append(accuracy_score(y_train[val_indices], y_out_svm_3))



# Mejor rendimiento obtenido con DNN_3 (Grupo training 3) y SVM_1 (Grupo training 3)
# Entrenamiento final
dnn_3.fit(training_matrix[train_idx[2]], y_train[train_idx[2]], epochs = 300, 
							verbose = 1, workers = 8, use_multiprocessing = True,
							validation_data = (training_matrix[val_idx[2]], y_train[val_idx[2]]))

svm_ova_1.fit(training_matrix[train_idx[2]], y_train[train_idx[2]])

# Prueba:
y_hat_dnn = dnn_3.predict(testing_matrix)
y_out_dnn = y_hat_dnn.round()

y_out_svm = svm_ova_1.predict(testing_matrix)

# Métricas de desempeño: 
c_dnn = confusion_matrix(y_test, y_out_dnn)
acc_dnn = 100*((c_dnn[0,0] + c_dnn[1,1])/np.sum(c_dnn))
se_dnn = 100*c_dnn[0,0]/(c_dnn[0,0] + c_dnn[0,1])
sp_dnn = 100*c_dnn[1,1]/(c_dnn[1,1] + c_dnn[1,0])

c_svm = confusion_matrix(y_test, y_out_svm)
acc_svm = 100*((c_svm[0,0] + c_svm[1,1])/np.sum(c_svm))
se_svm = 100*c_svm[0,0]/(c_svm[0,0] + c_svm[0,1])
sp_svm = 100*c_svm[1,1]/(c_svm[1,1] + c_svm[1,0])

plt.plot(signal)
plt.plot(r_times,signal[r_times.astype(int)], "x")
plt.xlim(0,len(signal))
plt.grid()
plt.show()
