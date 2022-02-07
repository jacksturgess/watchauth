#python filter.py [userID(s)] [param(s)] [activity] [is_overwritenongestures] [is_displayfiltergraphs]
# - userID(s) (may be a list)
# - param(s): f<windowsize>o<offset> or u<windowsize>o<offset> (may be a list)
# - activity: g (gestures only), n (non-gestures only), or a (all)
# - is_overwritenongestures: true of false (for whether or not to re-process an existing nongestures file)
# - is_displayfiltergraphs: true or false (for whether or not to show unfiltered/filtered graph comparison for each sensor of each gesture)
#
#opens the file <userID>-gestures.csv, grabs data from a time window of <windowsize> to the end (i.e. the NFC trigger), and applies a low pass filter to each gesture
#then does likewise for <userID>-nongestures.csv
#outputs <userID>-<param>-gestures.csv and <userID>-<windowsize>-nongestures.csv

import csv, math, os, re, shutil, sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter, freqz

userIDs = re.split(',', (sys.argv[1]).lower())
params = re.split(',', (sys.argv[2]).lower())
activity = (sys.argv[3]).lower()
is_overwritenongestures = True if 'TRUE' == (sys.argv[4]).upper() or '1' == sys.argv[4] else False
is_displayfiltergraphs = True if 'TRUE' == (sys.argv[5]).upper() or '1' == sys.argv[5] else False

#configs
sensors = ['Acc', 'Gyr', 'GRV', 'LAc']
maxprewindowsize = 4
minsamplespergesture = 95 #at 50Hz sampling rate over 4 sensors, 100 samples are gathered in 0.5 seconds; we set 95 to tolerate slight temporal fluctuations

#consts
g_index = 0 #gesture name and index
s_index = 1 #sensor name
t_index = 2 #gesture timestamp (in seconds, range from prewindowsize to 0)
x_index = 3 #x-value
y_index = 4 #y-value
z_index = 5 #z-value
w_index = 6 #GRV w-value
e_index = 7 #Euclidean norms of unfiltered values

def tidy_userIDs():
	global userIDs
	t_userIDs = []
	for userID in userIDs:	
		if 'user' in userID:
			t_userIDs.append('user' + f'{int(userID[4:]):03}')
		else:
			t_userIDs.append('user' + f'{int(userID):03}')
	t_userIDs.sort(reverse = False)
	userIDs = t_userIDs

def tidy_params():
	global params
	t_params = []
	for param in params:
		windowsize = 0
		offset = 0
		if 'o' in param:
			if 'om' in param:
				param = re.sub('om', 'o-', param)
			windowsize = float(param[1:param.index('o')])
			offset = float(param[param.index('o') + 1:])
		else:
			windowsize = float(param[1:])
		if windowsize + offset > maxprewindowsize:
			windowsize = maxprewindowsize - offset
		windowsize = str('%.1f' % windowsize)
		offset = str('%.1f' % offset)
		if 'f' == param[0]:
			t_params.append('f' + windowsize + 'o' + offset)
		elif 'u' == param[0]:
			t_params.append('u' + windowsize + 'o' + offset)
		else:
			sys.exit('ERROR: param not valid: ' + param)
	params = t_params

def rewrite_param(param):
	t_param = param
	if 'o-' in t_param:
		t_param = re.sub('o-', 'om', t_param)
	return t_param

def butter_lowpass(cutoff, r, order):
    nyq = 0.5 * r
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype = 'low', analog = False)
    return (b, a)

def butter_lowpass_filter(d, cutoff, r, order):
    b, a = butter_lowpass(cutoff, r, order)
    y = lfilter(b, a, d)
    return y

def filterGesture(data, name):
	output = []
	for s in sensors:
		times = []
		x = []
		y = []
		z = []
		w = []
		e = []
		for datum in data:
			if s == datum[s_index]:
				times.append(float(datum[t_index]))
				x.append(float(datum[x_index]))
				y.append(float(datum[y_index]))
				z.append(float(datum[z_index]))
				if 'GRV' == s:
					w.append(float(datum[w_index]))
				else:
					e.append(datum[e_index])
		
		#filtering variables
		order = 6
		cutoff = 3.667 #desired cutoff frequency of the filter, Hz
		n = len(times) #total number of samples
		t = np.linspace(times[0], times[len(times) - 1], n, endpoint = False) #evenly spaced time intervals
		r = int(n / (times[len(times) - 1] - times[0])) #sample rate, Hz
		
		b, a = butter_lowpass(cutoff, r, order) #gets the filter coefficients so we can check its frequency response
		filtered_x = butter_lowpass_filter(x, cutoff, r, order)
		filtered_y = butter_lowpass_filter(y, cutoff, r, order)
		filtered_z = butter_lowpass_filter(z, cutoff, r, order)
		filtered_w = []
		if 'GRV' == s:
			filtered_w = butter_lowpass_filter(w, cutoff, r, order)
		
		if is_displayfiltergraphs:
			#plot the frequency response
			w, h = freqz(b, a, worN = 8000)
			plt.subplot(2, 2, 1)
			plt.plot(0.5 * r * w / np.pi, np.abs(h), 'b')
			plt.plot(cutoff, 0.5 * np.sqrt(2), 'ko')
			plt.axvline(cutoff, color = 'k')
			plt.xlim(0, 0.5 * r)
			plt.title("Lowpass Filter Frequency Response")
			plt.xlabel('frequency (Hz)')
			plt.grid()
			
			#plot both the original and filtered x signals
			plt.subplot(2, 2, 2)
			plt.plot(times, x, 'b-', label = 'original x output')
			plt.plot(t, filtered_x, 'g-', linewidth = 2, label = 'filtered x output')
			plt.title("Sensor: " + s)
			plt.xlabel('time (s)')
			plt.grid()
			plt.legend()
			
			#plot both the original and filtered y signals
			plt.subplot(2, 2, 3)
			plt.plot(times, y, 'b-', label = 'original y output')
			plt.plot(t, filtered_y, 'g-', linewidth = 2, label = 'filtered y output')
			plt.xlabel('time (s)')
			plt.grid()
			plt.legend()
			
			#plot both the original and filtered z signals
			plt.subplot(2, 2, 4)
			plt.plot(times, z, 'b-', label = 'original z output')
			plt.plot(t, filtered_z, 'g-', linewidth = 2, label = 'filtered z output')
			plt.xlabel('time (s)')
			plt.grid()
			plt.legend()
			
			plt.subplots_adjust(hspace = 0.35)
			plt.show()
		
		#restructure data for output
		for i in range(len(t)):
			d_x = filtered_x[i]
			d_y = filtered_y[i]
			d_z = filtered_z[i]
			datum = []
			datum.append(name) #adds gesture name and index
			datum.append(s) #adds sensor name
			datum.append(str('%.6f' % t[i])) #adds new timestamp
			datum.append(str('%.6f' % d_x)) #adds filtered x-value
			datum.append(str('%.6f' % d_y)) #adds filtered y-value
			datum.append(str('%.6f' % d_z)) #adds filtered z-value
			if 'GRV' == s:
				datum.append(str('%.6f' % filtered_w[i])) #adds filtered w-value
				datum.append('') #adds empty field instead of Euclidean norm of unfiltered values
				datum.append('') #adds empty field instead of Euclidean norm of filtered values
			else:
				datum.append('') #adds empty field instead of filtered w-value
				datum.append(e[i]) #adds Euclidean norm of unfiltered values
				datum.append(str('%.6f' % math.sqrt(d_x * d_x + d_y * d_y + d_z * d_z))) #adds Euclidean norm of filtered values
			output.append(datum)
	return output

if __name__ == '__main__':
	tidy_userIDs()
	tidy_params()
	
	for userID in userIDs:
		for param in params:
			windowsize = str('%.1f' % float(param[1:param.index('o')]))
			offset = str('%.1f' % float(param[param.index('o') + 1:]))
			endtime = -float(offset)
			starttime = endtime - float(windowsize)
			is_filter = True if 'f' == param[0] else False
			
			activities = [] #'True' for gestures, 'False' for non-gestures, one of each for all
			if 'a' == activity:
				activities.extend([True, False])
			elif 'g' == activity:
				activities.append(True)
			elif 'n' == activity:
				activities.append(False)
			else:
				sys.exit('ERROR: activity not valid: ' + activity)
			
			#get and process gesture and non-gesture data
			for is_gesture in activities:
				f_outfilename = userID + '-' + rewrite_param(param) + '-gestures.csv' if is_gesture else userID + '-' + param[0] + windowsize + '-nongestures.csv'
				if os.path.exists(f_outfilename) and not is_gesture and not is_overwritenongestures:
					break
				
				file = userID + '-gestures.csv' if is_gesture else userID + '-nongestures.csv'
				if os.path.exists(file):
					with open(file, 'r') as f:
						data = list(csv.reader(f)) #returns a list of lists (each line is a list)
						data.pop(0) #removes the column headers
						
						gesture_string = 'GESTURE' if is_gesture else 'NON-GESTURE'
						filter_string = ',FILTERED_EUCLIDEAN_NORM' if is_filter else ''
						
						#build list of gesture indices
						gestureindices = []
						for datum in data:
							if datum[g_index] not in gestureindices:
								gestureindices.append(datum[g_index])
						
						if len(gestureindices) > 0:
							f_outfile = open(f_outfilename, 'w')
							f_outfile.write(gesture_string + ',SENSOR,GESTURE_TIMESTAMP,FILTERED_X-VALUE,FILTERED_Y-VALUE,FILTERED_Z-VALUE,FILTERED_W-VALUE,UNFILTERED_EUCLIDEAN_NORM' + filter_string)
							f_outfile.close()
							
							#for each gesture index: project a time window (backwards from the end), grab the data inside that window, apply a low pass filter to it, and write it
							for gestureindex in gestureindices:
								#get gesture data
								g_output = []
								for datum in data:
									if datum[g_index] == gestureindex:
										t = float(datum[t_index])
										if t >= starttime and t <= endtime:
											g_output.append(datum)
								
								#only process gesture if it has sufficient data (relevant for small time windows)
								if len(g_output) >= minsamplespergesture:
									if is_filter:
										#filter and restructure data
										g_output = filterGesture(g_output, gestureindex)
									else:
										#only restructure data
										SORT_ORDER = {'Acc': 0, 'Gyr': 1, 'GRV': 2, 'LAc': 3}
										g_output.sort(key = lambda g: SORT_ORDER[g[s_index]])
									
									#output filtered gesture
									f_outfile = open(f_outfilename, 'a')
									f_outfile.write('\n' + '\n'.join([','.join(g) for g in g_output]))
									f_outfile.close()
								else:
									print('REJECTED: ' + gestureindex + ' (' + str(len(g_output)) + ' entries)')
							print('OUTPUT: ' + f_outfilename)