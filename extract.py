#python extract.py [userID(s)] [param(s)] [activity] [is_overwritenongestures]
# - userID(s) (may be a list)
# - param(s): f<windowsize>o<offset> or u<windowsize>o<offset> (may be a list)
# - activity: g (gestures only), n (non-gestures only), or a (all)
# - is_overwritenongestures: true of false (for whether or not to re-process an existing nongestures file)
#
#opens the file <userID>-<param>-gestures.csv and reduces each gesture to a set of values (i.e. extracts features)
#then does likewise for <userID>-<windowsize>-nongestures.csv
#outputs <userID>-<param>-features.csv or <userID>-<windowsize>-nonfeatures.csv

import csv, math, os, re, statistics, sys
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew

userIDs = re.split(',', (sys.argv[1]).lower())
params = re.split(',', (sys.argv[2]).lower())
activity = (sys.argv[3]).lower()
is_overwritenongestures = True if 'TRUE' == (sys.argv[4]).upper() or '1' == sys.argv[4] else False

#configs
maxprewindowsize = 4

#consts
g_index = 0 #gesture name and index
s_index = 1 #sensor name
t_index = 2 #gesture timestamp (in seconds, range from prewindowsize to 0)
x_index = 3 #x-value
y_index = 4 #y-value
z_index = 5 #z-value
w_index = 6 #GRV w-value
e_unf_index = 7 #Euclidean norms of unfiltered values
e_fil_index = 8 #Euclidean norms of filtered values

#values
f_names = [] #container to hold the names of the features
data_acc = []
data_gyr = []
data_grv = []
data_lac = []

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

def write_f_name(s):
	for dimension in ['x-', 'y-', 'z-', 'e_unf-', 'e_fil-']:
		f_names.append('Acc-' + dimension + s)
	for dimension in ['x-', 'y-', 'z-', 'e_unf-', 'e_fil-']:
		f_names.append('Gyr-' + dimension + s)		
	for dimension in ['x-', 'y-', 'z-', 'w-']:
		f_names.append('GRV-' + dimension + s)
	for dimension in ['x-', 'y-', 'z-', 'e_unf-', 'e_fil-']:
		f_names.append('LAc-' + dimension + s)

def feature_min(is_firstparse, g_data):
	if is_firstparse:
		write_f_name('min')
	
	f = []
	for datum in g_data:
		f.append(str('%.6f' % min(datum)))
	return ','.join(f)

def feature_max(is_firstparse, g_data):
	if is_firstparse:
		write_f_name('max')
	
	f = []
	for datum in g_data:
		f.append(str('%.6f' % max(datum)))
	return ','.join(f)

def feature_mean(is_firstparse, g_data):
	if is_firstparse:
		write_f_name('mean')
	
	f = []
	for datum in g_data:
		f.append(str('%.6f' % statistics.mean(datum)))
	return ','.join(f)

def feature_med(is_firstparse, g_data):
	if is_firstparse:
		write_f_name('med')
	
	f = []
	for datum in g_data:
		f.append(str('%.6f' % statistics.median(datum)))
	return ','.join(f)
	
def feature_stdev(is_firstparse, g_data):
	if is_firstparse:
		write_f_name('stdev')
	
	f = []
	for datum in g_data:
		f.append(str('%.6f' % np.std(datum, ddof = 1)))
	return ','.join(f)

def feature_var(is_firstparse, g_data):
	if is_firstparse:
		write_f_name('var')
	
	f = []
	for datum in g_data:
		f.append(str('%.6f' % np.var(datum, ddof = 1)))
	return ','.join(f)

def feature_iqr(is_firstparse, g_data):
	if is_firstparse:
		write_f_name('iqr')
	
	f = []
	for datum in g_data:
		q75, q25 = np.percentile(datum, [75, 25])
		f.append(str('%.6f' % (q75 - q25)))
	return ','.join(f)

def feature_kurt(is_firstparse, g_data):
	if is_firstparse:
		write_f_name('kurt')
	
	f = []
	for datum in g_data:
		f.append(str('%.6f' % kurtosis(datum)))
	return ','.join(f)

def feature_skew(is_firstparse, g_data):
	if is_firstparse:
		write_f_name('skew')
	f = []
	for datum in g_data:
		f.append(str('%.6f' % skew(datum)))
	return ','.join(f)

def feature_pkcount(is_firstparse, g_data, threshold):
	if is_firstparse:
		write_f_name('pkcount' + str(threshold))
	f = []
	for datum in g_data:
		f.append(str('%.0f' % len(find_peaks(datum, prominence = threshold)[0])))
	return ','.join(f)

def feature_velo_disp(is_firstparse):
	f = []
	for sensor in ['Acc', 'Gyr', 'LAc']:
		#prepare data
		data = []
		if 'Acc' == sensor:
			data = list(data_acc)
		elif 'Gyr' == sensor:
			data = list(data_gyr)
		elif 'LAc' == sensor:
			data = list(data_lac)
		
		vx = [0]
		dx = [0]
		vy = [0]
		dy = [0]
		vz = [0]
		dz = [0]
		d = [0]
		n = len(data) - 1 #number of samples
		dt = float((data[n][0] - data[0][0]) / n) #sample interval
		for j in range(n):
			vx.append(vx[j] + (data[j][1] + data[j + 1][1]) / 2 * dt / 10)
			dx.append(dx[j] + vx[j + 1] * dt / 10)
			vy.append(vy[j] + (data[j][2] + data[j + 1][2]) / 2 * dt / 10)
			dy.append(dy[j] + vy[j + 1] * dt / 10)
			vz.append(vz[j] + (data[j][3] + data[j + 1][3]) / 2 * dt / 10)
			dz.append(dz[j] + vz[j + 1] * dt / 10)
			d.append(math.sqrt(dx[j] * dx[j] + dy[j] * dy[j] + dz[j] * dz[j]))
		vx.pop(0)
		vy.pop(0)
		vz.pop(0)
		
		if is_firstparse:
			f_names.append(sensor + '-x-velomean')
			f_names.append(sensor + '-y-velomean')
			f_names.append(sensor + '-z-velomean')
			f_names.append(sensor + '-x-velomax')
			f_names.append(sensor + '-y-velomax')
			f_names.append(sensor + '-z-velomax')
			f_names.append(sensor + '-x-disp')
			f_names.append(sensor + '-y-disp')
			f_names.append(sensor + '-z-disp')
			f_names.append(sensor + '-disptotal')
		
		f.append(str('%.6f' % (sum(vx) / len(vx))))
		f.append(str('%.6f' % (sum(vy) / len(vy))))
		f.append(str('%.6f' % (sum(vz) / len(vz))))
		f.append(str('%.6f' % max(vx, key = abs)))
		f.append(str('%.6f' % max(vy, key = abs)))
		f.append(str('%.6f' % max(vz, key = abs)))
		f.append(str('%.6f' % dx[len(dx) - 1]))
		f.append(str('%.6f' % dy[len(dy) - 1]))
		f.append(str('%.6f' % dz[len(dz) - 1]))
		f.append(str('%.6f' % d[len(d) - 1]))
	return ','.join(f)

def extractFeatures(is_firstparse, is_filter):
	#prepare data
	g_data = []
	if is_filter:
		g_data = [[row[1] for row in data_acc], [row[2] for row in data_acc], [row[3] for row in data_acc], [row[4] for row in data_acc], [row[5] for row in data_acc], 
		 [row[1] for row in data_gyr], [row[2] for row in data_gyr], [row[3] for row in data_gyr], [row[4] for row in data_gyr], [row[5] for row in data_gyr],
		 [row[1] for row in data_grv], [row[2] for row in data_grv], [row[3] for row in data_grv], [row[4] for row in data_grv],
		 [row[1] for row in data_lac], [row[2] for row in data_lac], [row[3] for row in data_lac], [row[4] for row in data_lac], [row[5] for row in data_lac]]
	else:
		g_data = [[row[1] for row in data_acc], [row[2] for row in data_acc], [row[3] for row in data_acc], [row[4] for row in data_acc], 
		 [row[1] for row in data_gyr], [row[2] for row in data_gyr], [row[3] for row in data_gyr], [row[4] for row in data_gyr],
		 [row[1] for row in data_grv], [row[2] for row in data_grv], [row[3] for row in data_grv], [row[4] for row in data_grv],
		 [row[1] for row in data_lac], [row[2] for row in data_lac], [row[3] for row in data_lac], [row[4] for row in data_lac]]
	
	#call features for this gesture
	f_data = []
	f_data.append(feature_min(is_firstparse, g_data))
	f_data.append(feature_max(is_firstparse, g_data))
	f_data.append(feature_mean(is_firstparse, g_data))
	f_data.append(feature_med(is_firstparse, g_data))
	f_data.append(feature_stdev(is_firstparse, g_data))
	f_data.append(feature_var(is_firstparse, g_data))
	f_data.append(feature_iqr(is_firstparse, g_data))
	f_data.append(feature_kurt(is_firstparse, g_data))
	f_data.append(feature_skew(is_firstparse, g_data))
	f_data.append(feature_pkcount(is_firstparse, g_data, 0.5))
	f_data.append(feature_velo_disp(is_firstparse))
	return ','.join(f_data)

if __name__ == '__main__':
	tidy_userIDs()
	tidy_params()
	
	for userID in userIDs:
		for param in params:
			windowsize = str('%.1f' % float(param[1:param.index('o')]))
			offset = str('%.1f' % float(param[param.index('o') + 1:]))
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
				f_outfilename = userID + '-' + rewrite_param(param) + '-features.csv' if is_gesture else userID + '-' + param[0] + windowsize + '-nonfeatures.csv'
				if os.path.exists(f_outfilename):
					if not is_gesture and not is_overwritenongestures:
						break
					else:
						os.remove(f_outfilename)
				
				file = userID + '-' + rewrite_param(param) + '-gestures.csv' if is_gesture else userID + '-' + param[0] + windowsize + '-nongestures.csv'
				if os.path.exists(file):
					with open(file, 'r') as f:
						data = list(csv.reader(f)) #returns a list of lists (each line is a list)
						data.pop(0) #removes the column headers
						
						gesture_string = 'GESTURE' if is_gesture else 'NON-GESTURE'
						
						#build list of gesture indices
						gestureindices = []
						for datum in data:
							if datum[g_index] not in gestureindices:
								gestureindices.append(datum[g_index])
						
						#for each gesture index: grab its data, extract its features, and write them
						is_firstparse = True
						f_names.clear()
						for gestureindex in gestureindices:
							data_acc.clear()
							data_gyr.clear()
							data_grv.clear()
							data_lac.clear()
							
							#get gesture data
							for datum in data:
								if datum[g_index] == gestureindex:
									s = datum[s_index]
									d = []
									d.append(float(datum[t_index]))
									d.append(float(datum[x_index]))
									d.append(float(datum[y_index]))
									d.append(float(datum[z_index]))
									if 'GRV' == s:
										d.append(float(datum[w_index]))
									else:
										d.append(float(datum[e_unf_index]))
										if is_filter:
											d.append(float(datum[e_fil_index]))
									
									if 'Acc' == s:
										data_acc.append(d)	
									elif 'Gyr' == s:
										data_gyr.append(d)
									elif 'GRV' == s:
										data_grv.append(d)
									elif 'LAc' == s:
										data_lac.append(d)
							
							#extract features
							f_output = extractFeatures(is_firstparse, is_filter)
							
							#output features to the combined file
							f_outfile = open(f_outfilename, 'a')
							if is_firstparse:
								if not is_filter:
									f_names = list(filter(lambda a: not 'e_fil-' in a, f_names)) #removes every occurrence containing 'e_fil-'
								f_outfile.write(gesture_string + ',' + ','.join(f_names))
								is_firstparse = False
							f_outfile.write('\n' + gestureindex + ',' + f_output)
							f_outfile.close()
						print('OUTPUT: ' + f_outfilename)