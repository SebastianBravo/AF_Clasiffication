import numpy as np
from scipy.signal import find_peaks
import neurokit2 as nk

def normalize(signal):
	return (signal-min(signal))/(max(signal)-min(signal))

def low_pass_filter(signal):
	# y(nT) = 2y(nT - T) - y(nT - 2 T) + x(nT) - 2x(nT- 6T) + x(nT- 12T) 

	# Inicializar salida en 0:
	y = np.zeros((len(signal),1))

	# Filtro pasa bajos:
	for i in range(len(signal)):
		y[i] = signal[i] # x(nT)

		if (i >= 1):
			y[i] += 2*y[i-1] # + 2y(nT - T)

		if (i >= 2):
			y[i] -= y[i-2] # - y(nT - 2T)

		if (i >= 6):
			y[i] -= 2*signal[i-6] # - 2x(nT - 6T)

		if (i >= 12):
			y[i] += signal[i-12]  # + x(nT - 12T)

	return y/max(abs(y)) # Salida normalizada

def high_pass_filter(signal):	
	# y(nT) = 32x(nT - 16 T) - [y(nT - T) + x(nT) - x(nT - 32 T)]
	
	# Inicializar salida en 0:
	y = np.zeros((len(signal),1))

	# Filtro pasa altos:
	for i in range(len(signal)):
		y[i] = -signal[i] # - x(nT)

		if (i >= 1):
			y[i] -= y[i-1] # - y(nT - T)

		if (i >= 16):
			y[i] += 32*signal[i-16] # + 32x(nT - 16T)

		if (i >= 32):
			y[i] += signal[i-32] # + x(nT - 32T)
	
	return y/max(abs(y)) # Salida normalizada

def derivative_filter(signal,fs):
	# y(nT) = (1/8 T) [-x(nT - 2 T) - 2x(nT - T) + 2x(nT + T) + x(nT+ 2T)]

	# Inicializar salida en 0:
	y = np.zeros((len(signal),1))

	# Filtro pasa altos:
	for i in range(len(signal)):
		if (i >= 1):
			y[i] -= 2*signal[i-1] # - 2x(nT - T)

		if (i >= 2):
			y[i] -= signal[i-2] # -x(nT - 2 T)

		if (i <= len(signal)-2):
			y[i] += 2*signal[i+1] # + 2x(nT + T)

		if (i <= len(signal)-3):
			y[i] += signal[i+2] # + x(nT+ 2T)

	y = fs*y/8 # (1/8 T)

	return y

def square_signal(signal):
	y = np.square(signal)

	return y


def window_integration(signal, fs):
	# y(nT) = (1/N) [x(nT- (N - 1) T) +x(nT- (N - 2) T) + ... + x(nT)]
	
	# Tamaño de la ventana
	#N = int(fs*0.1)
	N = 30

	# Inicializar salida en 0:
	y = np.zeros((len(signal),1))

	for i in range(len(signal)):
		if i>=N:
			y[i] = np.sum(signal[i-N:i])/N

	return y

def peak_finder(signal, fs):  
    min_distance = int(0.25*fs)

    signal_peaks = [0]
    noise_peaks = []

    SPKI = 0.0
    NPKI = 0.0

    threshold_I1 = 0.0
    threshold_I2 = 0.0

    RR_missed = 0
    index = 0
    indexes = []

    missed_peaks = []
    peaks = []

    for i in range(len(signal)):

        if i>0 and i<len(signal)-1:
            if signal[i-1]<signal[i] and signal[i+1]<signal[i]:
                peak = i
                peaks.append(i)

                if signal[peak]>threshold_I1 and (peak-signal_peaks[-1])>0.3*fs:
                        
                    signal_peaks.append(peak)
                    indexes.append(index)
                    SPKI = 0.125*signal[signal_peaks[-1]] + 0.875*SPKI
                    if RR_missed!=0:
                        if signal_peaks[-1]-signal_peaks[-2]>RR_missed:
                            missed_section_peaks = peaks[indexes[-2]+1:indexes[-1]]
                            missed_section_peaks2 = []
                            for missed_peak in missed_section_peaks:
                                if missed_peak-signal_peaks[-2]>min_distance and signal_peaks[-1]-missed_peak>min_distance and signal[missed_peak]>threshold_I2:
                                    missed_section_peaks2.append(missed_peak)

                            if len(missed_section_peaks2)>0:           
                                missed_peak = missed_section_peaks2[np.argmax(signal[missed_section_peaks2])]
                                missed_peaks.append(missed_peak)
                                signal_peaks.append(signal_peaks[-1])
                                signal_peaks[-2] = missed_peak   

                else:
                    noise_peaks.append(peak)
                    NPKI = 0.125*signal[noise_peaks[-1]] + 0.875*NPKI

                threshold_I1 = NPKI + 0.25*(SPKI-NPKI)
                threshold_I2 = 0.5*threshold_I1

                if len(signal_peaks)>8:
                    RR = np.diff(signal_peaks[-9:])
                    RR_ave = int(np.mean(RR))
                    RR_missed = int(1.66*RR_ave)

                index = index+1      
    
    signal_peaks.pop(0)

    return signal_peaks

def pan_tompkins_detection(signal, fs):
	result = signal.copy()
	result = normalize(result)
	result = low_pass_filter(result)
	result = high_pass_filter(result)
	result = derivative_filter(result,fs)
	result = square_signal(result)
	result = window_integration(result, fs)
	
	return peak_finder(result, fs)

