#python classify.py [userID(s)] [param(s)] [mode(s)] [sensors] [gesture(s)]
# - userID(s) (may be a list)
# - param(s): f<windowsize>o<offset> or u<windowsize>o<offset> (may be a list)
# - mode(s): authblind, recog, or recogblind (may be a list)
# - sensors: a (all) or list of any of {Acc, Gyr, GRV, LAc}
# - gesture(s): a (all) or specific gesture (e.g. TAP1) (may be a list)
#
#opens all <userIDs>-features.csv files, grabs all the feature data, classifies, and then tests the model
#results are output in the format: <userID> or 'average', precision, std dev. of precision, recall, std dev. of recall, F1, std dev. of F1
#outputs <datetime>-<userIDs>-<mode>-<classifier>-<param>.csv

import datetime, csv, os, re, sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, f1_score, precision_score, precision_recall_curve, recall_score, roc_curve
from sklearn.model_selection import StratifiedKFold

userIDs = re.split(',', (sys.argv[1]).lower())
params = re.split(',', (sys.argv[2]).lower())
modes = re.split(',', (sys.argv[3]).lower())
sensors = ['Acc', 'Gyr', 'GRV', 'LAc'] if 'a' == (sys.argv[4]).lower() else re.split(',', sys.argv[4])
gestures = re.split(',', (sys.argv[5]).lower())
for i in range(len(gestures)):
	if gestures[i] != 'a':
		gestures[i] = gestures[i].upper()

#configs
maxprewindowsize = 4
classifier = 'rfc'
repetitions = 10
folds = 10
fontsize_legends = 20

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

def get_features(data):
	featurecolumns = []
	f_names = data[0][1:]
	f_column = 1
	for f_name in f_names:
		s = re.split('-', f_name)[0]
		for sensor in sensors:
			if s == sensor:
				featurecolumns.append(f_column)
				break;
		f_column = f_column + 1
	featurecolumns.sort(reverse = False)
	featurenames = [f_names[c - 1] for c in featurecolumns]
	return featurenames, featurecolumns

def get_average(l):
	return 0 if 0 == len(l) else sum(l) / len(l)

def get_eer(scores_legit, scores_adv):
	scores_legit = sorted(scores_legit)
	scores_adv = sorted(scores_adv)
	
	#treat each legitimate sample distance as a possible threshold, determine the point where FRR crosses FAR
	for c, threshold in enumerate(scores_legit):
		frr = c * 1.0 / len(scores_legit)
		adv_index = next((x[0] for x in enumerate(scores_adv) if x[1] > threshold), len(scores_adv))
		far = 1 - (adv_index * 1.0 / len(scores_adv))
		if frr >= far:
			return threshold, far

def get_eer_recogblind(scores_legit, scores_adv_typed, total_w, total_b, total_i):
	scores_legit = sorted(scores_legit)
	scores_adv_typed = sorted(scores_adv_typed, key = lambda x:x[0])
	
	#treat each legitimate sample distance as a possible threshold, determine the point where FRR crosses FAR
	for c, threshold in enumerate(scores_legit):
		frr = c * 1.0 / len(scores_legit)
		adv_index = next((x[0] for x in enumerate(scores_adv_typed) if x[1][0] > threshold), len(scores_adv_typed))
		far = 1 - (adv_index * 1.0 / len(scores_adv))
		if frr >= far:
			rejectrate_w = 0 if 0 == total_w else len([i for i in scores_adv_typed if 'W' == i[1] and i[0] >= threshold]) / total_w
			rejectrate_b = 0 if 0 == total_b else len([i for i in scores_adv_typed if 'B' == i[1] and i[0] >= threshold]) / total_b
			rejectrate_i = 0 if 0 == total_i else len([i for i in scores_adv_typed if 'I' == i[1] and i[0] >= threshold]) / total_i
			return threshold, far, rejectrate_w, rejectrate_b, rejectrate_i

def get_far_when_zero_frr(scores_legit, scores_adv):
	scores_legit = sorted(scores_legit)
	scores_adv = sorted(scores_adv)
	
	#treat each legitimate sample distance as a possible threshold, determine the point with the lowest FAR that satisfies the condition that FRR = 0
	for c, threshold in enumerate(scores_legit):
		frr = c * 1.0 / len(scores_legit)
		adv_index = next((x[0] for x in enumerate(scores_adv) if x[1] > threshold), len(scores_adv))
		far = 1 - (adv_index * 1.0 / len(scores_adv))
		if frr > 0.001:
			return threshold, far

def plot_threshold_by_far_frr(scores_legit, scores_adv, far_theta):
	scores_legit = sorted(scores_legit)
	scores_adv = sorted(scores_adv)
	
	frr = []
	far = []
	thresholds = []
	for c, threshold in enumerate(scores_legit):
		frr.append((c * 1.0 / len(scores_legit)) * 100)
		adv_index = next((x[0] for x in enumerate(scores_adv) if x[1] > threshold), len(scores_adv))
		far.append((1 - (adv_index * 1.0 / len(scores_adv))) * 100)
		thresholds.append(threshold)
	plt.figure(figsize = (6, 6))
	plt.rcParams.update({'font.size': fontsize_legends})
	plt.plot(thresholds, far, 'tab:blue', label = 'FAR')
	plt.plot(thresholds, frr, 'tab:orange', label = 'FRR')
	plt.ylabel('error rate (%)')
	plt.xlabel(r'decision threshold, $\theta$')
	plt.axvline(x = far_theta, c = 'red')
	plt.legend(loc = 'best')
	plt.tight_layout(pad = 0.05)
	plt.show()

def plot_threshold_by_precision_recall(labels_test, labels_scores):
	p, r, thresholds = precision_recall_curve(labels_test, labels_scores)
	plt.figure(figsize = (6, 6))
	plt.rcParams.update({'font.size': fontsize_legends})
	plt.title('Precision and Recall Scores as a Function of the Decision Threshold', fontsize = 12)
	plt.plot(thresholds, p[:-1], 'tab:blue', label = 'precision')
	plt.plot(thresholds, r[:-1], 'tab:orange', label = 'recall')
	plt.ylabel('score')
	plt.xlabel(r'decision threshold, $\theta$')
	plt.legend(loc = 'best')
	plt.tight_layout(pad = 0.05)
	plt.show()

def plot_roc_curve(labels_test, labels_scores):
	fpr, tpr, auc_thresholds = roc_curve(labels_test, labels_scores)
	print('AUC of ROC = ' + str(auc(fpr, tpr)))
	plt.figure(figsize = (6, 6))
	plt.rcParams.update({'font.size': fontsize_legends})
	plt.title('ROC Curve', fontsize = 12)
	plt.plot(fpr, tpr, 'tab:orange', label = 'recall optimized')
	plt.plot([0, 1], [0, 1], 'k--')
	plt.axis([-0.005, 1, 0, 1.005])
	plt.xticks(np.arange(0, 1, 0.05), rotation = 90)
	plt.xlabel('false positive rate')
	plt.ylabel('true positive rate (recall)')
	plt.legend(loc = 'best')
	plt.tight_layout(pad = 0.05)
	plt.show()

def get_ascending_userID_list_string():
	for u in userIDs:
		if not 'user' in u and len(u) != 7:
			sys.exit('ERROR: userID not valid: ' + str(u))
	IDs = [int(u[4:]) for u in userIDs]
	IDs.sort(reverse = False)
	return ','.join([f'{i:03}' for i in IDs])

def get_descending_feature_list_string(weights, labels, truncate = 0):
	indicies = [i for i in range(len(weights))]
	for i in range(len(indicies)):
		for j in range(len(indicies)):
			if i != j and weights[indicies[i]] > weights[indicies[j]]:
				temp = indicies[i]
				indicies[i] = indicies[j]
				indicies[j] = temp
	if truncate != 0:
		del indicies[truncate:]
	return '\n'.join([str('%.6f' % weights[i]) + ' (' + labels[i] + ')' for i in indicies])

def write_verbose(f, s):
	outfilename = f + '-verbose.txt'
	outfile = open(outfilename, 'a')
	outfile.write(s + '\n')
	outfile.close()

if __name__ == '__main__':
	tidy_userIDs()
	tidy_params()
	
	for gesture in gestures:
		for mode in modes:
			for param in params:
				windowsize = str('%.1f' % float(param[1:param.index('o')]))
				offset = str('%.1f' % float(param[param.index('o') + 1:]))
				mode_string = mode +'(n=' + str(trainingsizemultiplier) +')' if 'authblind6n' == mode else mode
				filename_string = datetime.datetime.now().strftime("%Y%m%d") + '-' + get_ascending_userID_list_string() + '-' + mode_string + '-' + classifier + '-' + rewrite_param(param) + '-' + ','.join(sensors) + '-' + gesture
				
				output = []
				print('----\nINFO: param: ' + param + ', classifier: ' + classifier + ', mode: ' + mode_string + ', sensor(s): ' + ','.join(sensors) + ', gesture: ' + gesture + '\n----')
				
				a_data = [] #container to hold the feature data for all users
				a_labels = [] #container to hold the corresponding labels
				a_precisions = []
				a_recalls = []
				a_fmeasures = []
				a_pr_stdev = []
				a_re_stdev = []
				a_fm_stdev = []
				a_eers = []
				a_eer_thetas = []
				a_fars = []
				a_far_thetas = []
				a_ee_stdev = []
				a_ee_th_stdev = []
				a_fa_stdev = []
				a_fa_th_stdev = []
				featurenames = [] #container to hold the names of the features
				featurecolumns = [] #container to hold the column indices of the features to be used (determined by sensors)
				is_firstparse = True
				
				if 'authblind' == mode and len(userIDs) > 1 and 'a' == gesture:
					output.append('userID,prec_avg,prec_stdev,rec_avg,rec_stdev,fm_avg,fm_stdev,eer_avg,eer_stdev,eer_theta_avg,eer_theta_stdev,far_avg,far_stdev,far_theta_avg,far_theta_stdev')
					
					#get feature data and labels for all users
					for userID in userIDs:
						file = userID + '-' + rewrite_param(param) + '-features.csv'
						if os.path.exists(file):
							with open(file, 'r') as f:
								data = list(csv.reader(f)) #returns a list of lists (each line is a list)
								if is_firstparse:
									featurenames, featurecolumns = get_features(data)
									is_firstparse = False
								data.pop(0) #removes the column headers
								
								for datum in data:
									d = [datum[0]]
									d.extend([float(datum[n]) for n in featurecolumns])
									a_data.append(d)
									a_labels.append(userID)
					
					#run tests
					for test_terminal in ['TAP1', 'TAP2', 'TAP3', 'TAP4', 'TAP5', 'TAP6']:
						t_precisions = []
						t_recalls = []
						t_fmeasures = []
						t_eers = []
						t_eer_thetas = []
						t_fars = []
						t_far_thetas = []
						
						for userID in userIDs:
							data_train = []
							data_test = []
							labels_train = []
							labels_test = []
							counter = [0, 0, 0, 0, 0, 0, 0]
							for i in range(len(a_data)):
								if re.split('-', a_data[i][0])[0] == test_terminal:
									data_test.append(a_data[i][1:])
									labels_test.append(1 if userID == a_labels[i] else 0)
								else:
									data_train.append(a_data[i][1:])
									labels_train.append(1 if userID == a_labels[i] else 0)
							
							#use the user's first and second sessions' tap gestures for training, rejecting the third session's							
							removetrainindicies = []
							usertrainsplit = int(labels_train.count(1) * 2 / 3) + 1
							usertraincounter = 0
							for i in range(len(labels_train)):
								if 1 == labels_train[i]:
									if usertraincounter < usertrainsplit:
										usertraincounter = usertraincounter + 1
									else:
										removetrainindicies.append(i)
							removetrainindicies.sort(reverse = True)
							for r in removetrainindicies:
								del data_train[r]
								del labels_train[r]
							
							#use the user's third session's tap gestures for testing, rejecting the first and second sessions'
							removetestindicies = []
							usertestsplit = int(labels_test.count(1) * 2 / 3) + 1
							usertestcounter = 0
							for i in range(len(labels_test)):
								if 1 == labels_test[i]:
									if usertestcounter < usertestsplit:
										removetestindicies.append(i)
									usertestcounter = usertestcounter + 1
							removetestindicies.sort(reverse = True)
							for r in removetestindicies:
								del data_test[r]
								del labels_test[r]
							
							for repetition in range(repetitions):
								model = RandomForestClassifier(n_estimators = 100, random_state = repetition).fit(data_train, labels_train)
								labels_pred = model.predict(data_test)
								
								#get precision, recall, and F-measure scores
								precision = precision_score(labels_test, labels_pred, average = 'macro', labels = np.unique(labels_pred))
								recall = recall_score(labels_test, labels_pred, average = 'macro', labels = np.unique(labels_pred))
								fmeasure = f1_score(labels_test, labels_pred, average = 'macro', labels = np.unique(labels_pred))
								t_precisions.append(precision)
								t_recalls.append(recall)
								t_fmeasures.append(fmeasure)
								
								#get EER and find the decision threshold and FAR when optimised for FRR
								labels_scores = model.predict_proba(data_test)[:, 1]
								scores_legit = [labels_scores[i] for i in range(len(labels_test)) if 1 == labels_test[i]]
								scores_adv = [labels_scores[i] for i in range(len(labels_test)) if 0 == labels_test[i]]
								eer_theta, eer = get_eer(scores_legit, scores_adv)
								t_eers.append(eer)
								t_eer_thetas.append(eer_theta)
								far_theta, far = get_far_when_zero_frr(scores_legit, scores_adv)
								t_fars.append(far)
								t_far_thetas.append(far_theta)
								
								write_verbose(filename_string, '----\n----EXCLUDED TERMINAL ' + test_terminal + ', USERID ' + userID + ', REPETITION ' + str(repetition) +
								 '\n----\nVALUES: precision=' + str('%.6f' % precision) + ', recall=' + str('%.6f' % recall) + ', fmeasure=' + str('%.6f' % fmeasure) +
								 ', eer=' + str('%.6f' % eer) + ', eer_theta=' + str('%.6f' % eer_theta) + ', far=' + str('%.6f' % far) + ', far_theta=' + str('%.6f' % far_theta) +
								 '\n----\nFEATURE COUNT: ' + str(len(featurenames)) +
								 '\n----\nORDERED FEATURE LIST:\n' + get_descending_feature_list_string(model.feature_importances_, featurenames))
						t_pr_stdev = np.std(t_precisions, ddof = 1)
						t_re_stdev = np.std(t_recalls, ddof = 1)
						t_fm_stdev = np.std(t_fmeasures, ddof = 1)
						t_ee_stdev = np.std(t_eers, ddof = 1)
						t_ee_th_stdev = np.std(t_eer_thetas, ddof = 1)
						t_fa_stdev = np.std(t_fars, ddof = 1)
						t_fa_th_stdev = np.std(t_far_thetas, ddof = 1)
						
						result_string = (test_terminal + ',' + str('%.6f' % get_average(t_precisions)) + ',' + str('%.6f' % t_pr_stdev) + ','
						 + str('%.6f' % get_average(t_recalls)) + ',' + str('%.6f' % t_re_stdev) + ','
						 + str('%.6f' % get_average(t_fmeasures)) + ',' + str('%.6f' % t_fm_stdev) + ','
						 + str('%.6f' % get_average(t_eers)) + ',' + str('%.6f' % t_ee_stdev) + ','
						 + str('%.6f' % get_average(t_eer_thetas)) + ',' + str('%.6f' % t_ee_th_stdev) + ','
						 + str('%.6f' % get_average(t_fars)) + ',' + str('%.6f' % t_fa_stdev) + ','
						 + str('%.6f' % get_average(t_far_thetas)) + ',' + str('%.6f' % t_fa_th_stdev)
						 )
						output.append(result_string)
						print(result_string)
						
						a_precisions.extend(t_precisions)
						a_recalls.extend(t_recalls)
						a_fmeasures.extend(t_fmeasures)
						a_pr_stdev.append(t_pr_stdev)
						a_re_stdev.append(t_re_stdev)
						a_fm_stdev.append(t_fm_stdev)
						a_eers.extend(t_eers)
						a_eer_thetas.extend(t_eer_thetas)
						a_fars.extend(t_fars)
						a_far_thetas.extend(t_far_thetas)
						a_ee_stdev.append(t_ee_stdev)
						a_ee_th_stdev.append(t_ee_th_stdev)
						a_fa_stdev.append(t_fa_stdev)
						a_fa_th_stdev.append(t_fa_th_stdev)
					result_string = ('average,' + str('%.6f' % get_average(a_precisions)) + ',' + str('%.6f' % get_average(a_pr_stdev)) + ','
					 + str('%.6f' % get_average(a_recalls)) + ',' + str('%.6f' % get_average(a_re_stdev)) + ','
					 + str('%.6f' % get_average(a_fmeasures)) + ',' + str('%.6f' % get_average(a_fm_stdev)) + ','
					 + str('%.6f' % get_average(a_eers)) + ',' + str('%.6f' % get_average(a_ee_stdev)) + ','
					 + str('%.6f' % get_average(a_eer_thetas)) + ',' + str('%.6f' % get_average(a_ee_th_stdev)) + ','
					 + str('%.6f' % get_average(a_fars)) + ',' + str('%.6f' % get_average(a_fa_stdev)) + ','
					 + str('%.6f' % get_average(a_far_thetas)) + ',' + str('%.6f' % get_average(a_fa_th_stdev))
					 )
					output.append(result_string)
					print(result_string + '\n----')
				elif 'recog' == mode:
					output.append('userID,prec_avg,prec_stdev,rec_avg,rec_stdev,fm_avg,fm_stdev,eer_avg,eer_stdev,eer_theta_avg,eer_theta_stdev,far_avg,far_stdev,far_theta_avg,far_theta_stdev')
					
					#get feature data and labels for all users
					for userID in userIDs:
						for is_gesture in [True, False]:
							file = userID + '-' + rewrite_param(param) + '-features.csv' if is_gesture else userID + '/' + userID + '-' + param[0] + windowsize + '-nonfeatures.csv'
							if os.path.exists(file):
								with open(file, 'r') as f:
									data = list(csv.reader(f)) #returns a list of lists (each line is a list)
									if is_firstparse:
										featurenames, featurecolumns = get_features(data)
										is_firstparse = False
									data.pop(0) #removes the column headers
									
									for datum in data:
										check = not is_gesture
										if is_gesture:
											if 'a' == gesture or re.split('-', datum[0])[0] == gesture:
												check = True
										if check:
											a_data.append([float(datum[n]) for n in featurecolumns])
											a_labels.append(1 if is_gesture else 0)
					
					#run tests
					for repetition in range(repetitions):
						clf = RandomForestClassifier(n_estimators = 100, random_state = repetition)
						
						#apply stratified k-fold cross-validation and fit the model on each fold
						skf = StratifiedKFold(n_splits = folds, shuffle = True, random_state = 0)
						fold = 0
						for train, test in skf.split(a_data, a_labels):
							data_train = [a_data[i] for i in train]
							data_test = [a_data[i] for i in test]
							labels_train = [a_labels[i] for i in train]
							labels_test = [a_labels[i] for i in test]
							model = clf.fit(data_train, labels_train)
							labels_pred = model.predict(data_test)
							
							#get precision, recall, and F-measure scores
							precision = precision_score(labels_test, labels_pred, average = 'macro', labels = np.unique(labels_pred))
							recall = recall_score(labels_test, labels_pred, average = 'macro', labels = np.unique(labels_pred))
							fmeasure = f1_score(labels_test, labels_pred, average = 'macro', labels = np.unique(labels_pred))
							a_precisions.append(precision)
							a_recalls.append(recall)
							a_fmeasures.append(fmeasure)
							
							#get EER and find the decision threshold and FAR when optimised for FRR
							labels_scores = model.predict_proba(data_test)[:, 1]
							scores_legit = [labels_scores[i] for i in range(len(labels_test)) if 1 == labels_test[i]]
							scores_adv = [labels_scores[i] for i in range(len(labels_test)) if 0 == labels_test[i]]
							eer_theta, eer = get_eer(scores_legit, scores_adv)
							a_eers.append(eer)
							a_eer_thetas.append(eer_theta)
							far_theta, far = get_far_when_zero_frr(scores_legit, scores_adv)
							a_fars.append(far)
							a_far_thetas.append(far_theta)
							
							write_verbose(filename_string, '----\n----REPETITION ' + str(repetition) + ', FOLD ' + str(fold) +
							 '\n----\nVALUES: precision=' + str('%.6f' % precision) + ', recall=' + str('%.6f' % recall) + ', fmeasure=' + str('%.6f' % fmeasure) +
							 ', eer=' + str('%.6f' % eer) + ', eer_theta=' + str('%.6f' % eer_theta) + ', far=' + str('%.6f' % far) + ', far_theta=' + str('%.6f' % far_theta) +
							 '\n----\nFEATURE COUNT: ' + str(len(featurenames)) +
							 '\n----\nORDERED FEATURE LIST:\n' + get_descending_feature_list_string(model.feature_importances_, featurenames))
							
							fold = fold + 1
					result_string = ('average,' + str('%.6f' % (sum(a_precisions) / len(a_precisions))) + ',' + str('%.6f' % np.std(a_precisions, ddof = 1)) + ','
					 + str('%.6f' % (sum(a_recalls) / len(a_recalls))) + ',' + str('%.6f' % np.std(a_recalls, ddof = 1)) + ','
					 + str('%.6f' % (sum(a_fmeasures) / len(a_fmeasures))) + ',' + str('%.6f' % np.std(a_fmeasures, ddof = 1)) + ','
					 + str('%.6f' % (sum(a_eers) / len(a_eers))) + ',' + str('%.6f' % np.std(a_eers, ddof = 1)) + ','
					 + str('%.6f' % (sum(a_eer_thetas) / len(a_eer_thetas))) + ',' + str('%.6f' % np.std(a_eer_thetas, ddof = 1)) + ','
					 + str('%.6f' % (sum(a_fars) / len(a_fars))) + ',' + str('%.6f' % np.std(a_fars, ddof = 1)) + ','
					 + str('%.6f' % (sum(a_far_thetas) / len(a_far_thetas))) + ',' + str('%.6f' % np.std(a_far_thetas, ddof = 1)))
					output.append(result_string)
					print(result_string + '\n----')
				elif 'recogblind' == mode:
					output.append('userID,prec_avg,prec_stdev,rec_avg,rec_stdev,fm_avg,fm_stdev,eer_avg,eer_stdev,eer_theta_avg,eer_theta_stdev,far_avg,far_stdev,far_theta_avg,far_theta_stdev,rrw,rrb,rri')
					
					a_types = [] #container to hold the corresponding gesture types					
					a_rejectrate_w = []
					a_rejectrate_b = []
					a_rejectrate_i = []
					
					#get feature data and labels for all users
					for userID in userIDs:
						for is_gesture in [True, False]:
							file = userID + '-' + rewrite_param(param) + '-features.csv' if is_gesture else userID + '/' + userID + '-' + param[0] + windowsize + '-nonfeatures.csv'
							if os.path.exists(file):
								with open(file, 'r') as f:
									data = list(csv.reader(f)) #returns a list of lists (each line is a list)
									if is_firstparse:
										featurenames, featurecolumns = get_features(data)
										is_firstparse = False
									data.pop(0) #removes the column headers
									
									for datum in data:
										check = not is_gesture
										if is_gesture:
											if 'a' == gesture or re.split('-', datum[0])[0] == gesture:
												check = True
										if check:
											d = [userID]
											d.extend([float(datum[n]) for n in featurecolumns])
											a_data.append(d)
											a_labels.append(1 if is_gesture else 0)
											a_types.append(re.split('-', datum[0])[0])
					
					#run tests
					for userID in userIDs:
						u_precisions = []
						u_recalls = []
						u_fmeasures = []
						u_eers = []
						u_eer_thetas = []
						u_fars = []
						u_far_thetas = []
						u_rejectrate_w = []
						u_rejectrate_b = []
						u_rejectrate_i = []
						
						data_train = []
						data_test = []
						labels_train = []
						labels_test = []
						types_test = []
						for i in range(len(a_data)):
							if a_data[i][0] == userID:
								data_test.append(a_data[i][1:])
								labels_test.append(a_labels[i])
								types_test.append(a_types[i])
							else:
								data_train.append(a_data[i][1:])
								labels_train.append(a_labels[i])
						total_w = len([i for i in types_test if 'W' == i])
						total_b = len([i for i in types_test if 'B' == i])
						total_i = len([i for i in types_test if 'I' == i])
						
						for repetition in range(repetitions):
							model = RandomForestClassifier(n_estimators = 100, random_state = repetition).fit(data_train, labels_train)
							labels_pred = model.predict(data_test)
							
							#get precision, recall, and F-measure scores
							precision = precision_score(labels_test, labels_pred, average = 'macro', labels = np.unique(labels_pred))
							recall = recall_score(labels_test, labels_pred, average = 'macro', labels = np.unique(labels_pred))
							fmeasure = f1_score(labels_test, labels_pred, average = 'macro', labels = np.unique(labels_pred))
							u_precisions.append(precision)
							u_recalls.append(recall)
							u_fmeasures.append(fmeasure)
							
							#get EER and find the decision threshold and FAR when optimised for FRR, also get the rejection rate of each type of non-tap gesture
							labels_scores = model.predict_proba(data_test)[:, 1]
							scores_legit = [labels_scores[i] for i in range(len(labels_test)) if 1 == labels_test[i]]
							scores_adv = [labels_scores[i] for i in range(len(labels_test)) if 0 == labels_test[i]]
							scores_adv_typed = [(labels_scores[i], types_test[i]) for i in range(len(labels_test)) if 0 == labels_test[i]]
							eer_theta, eer, rejectrate_w, rejectrate_b, rejectrate_i = get_eer_recogblind(scores_legit, scores_adv_typed, total_w, total_b, total_i)
							u_eers.append(eer)
							u_eer_thetas.append(eer_theta)
							if 0 != total_w:
								u_rejectrate_w.append(rejectrate_w)
							if 0 != total_b:
								u_rejectrate_b.append(rejectrate_b)
							if 0 != total_i:
								u_rejectrate_i.append(rejectrate_i)
							far_theta, far = get_far_when_zero_frr(scores_legit, scores_adv)
							u_fars.append(far)
							u_far_thetas.append(far_theta)
							
							write_verbose(filename_string, '----\n----USERID ' + userID + ', REPETITION ' + str(repetition) +
							 '\n----\nVALUES: precision=' + str('%.6f' % precision) + ', recall=' + str('%.6f' % recall) + ', fmeasure=' + str('%.6f' % fmeasure) +
							 ', eer=' + str('%.6f' % eer) + ', eer_theta=' + str('%.6f' % eer_theta) + ', far=' + str('%.6f' % far) + ', far_theta=' + str('%.6f' % far_theta) +
							 ', rrw=' + str('%.6f' % rejectrate_w) + ', rrb=' + str('%.6f' % rejectrate_b) + ', rri=' + str('%.6f' % rejectrate_i) +
							 '\n----\nFEATURE COUNT: ' + str(len(featurenames)) +
							 '\n----\nORDERED FEATURE LIST:\n' + get_descending_feature_list_string(model.feature_importances_, featurenames))
						u_pr_stdev = np.std(u_precisions, ddof = 1)
						u_re_stdev = np.std(u_recalls, ddof = 1)
						u_fm_stdev = np.std(u_fmeasures, ddof = 1)
						u_ee_stdev = np.std(u_eers, ddof = 1)
						u_ee_th_stdev = np.std(u_eer_thetas, ddof = 1)
						u_fa_stdev = np.std(u_fars, ddof = 1)
						u_fa_th_stdev = np.std(u_far_thetas, ddof = 1)
						
						result_string = (userID + ',' + str('%.6f' % get_average(u_precisions)) + ',' + str('%.6f' % u_pr_stdev) + ','
						 + str('%.6f' % get_average(u_recalls)) + ',' + str('%.6f' % u_re_stdev) + ','
						 + str('%.6f' % get_average(u_fmeasures)) + ',' + str('%.6f' % u_fm_stdev) + ','
						 + str('%.6f' % get_average(u_eers)) + ',' + str('%.6f' % u_ee_stdev) + ','
						 + str('%.6f' % get_average(u_eer_thetas)) + ',' + str('%.6f' % u_ee_th_stdev) + ','
						 + str('%.6f' % get_average(u_fars)) + ',' + str('%.6f' % u_fa_stdev) + ','
						 + str('%.6f' % get_average(u_far_thetas)) + ',' + str('%.6f' % u_fa_th_stdev) + ','
						 + str('%.6f' % get_average(u_rejectrate_w)) + ',' + str('%.6f' % get_average(u_rejectrate_b)) + ',' + str('%.6f' % get_average(u_rejectrate_i))
						 )
						output.append(result_string)
						print(result_string)
						
						a_precisions.extend(u_precisions)
						a_recalls.extend(u_recalls)
						a_fmeasures.extend(u_fmeasures)
						a_pr_stdev.append(u_pr_stdev)
						a_re_stdev.append(u_re_stdev)
						a_fm_stdev.append(u_fm_stdev)
						a_eers.extend(u_eers)
						a_eer_thetas.extend(u_eer_thetas)
						a_fars.extend(u_fars)
						a_far_thetas.extend(u_far_thetas)
						a_ee_stdev.append(u_ee_stdev)
						a_ee_th_stdev.append(u_ee_th_stdev)
						a_fa_stdev.append(u_fa_stdev)
						a_fa_th_stdev.append(u_fa_th_stdev)
						a_rejectrate_w.extend(u_rejectrate_w)
						a_rejectrate_b.extend(u_rejectrate_b)
						a_rejectrate_i.extend(u_rejectrate_i)
					result_string = ('average,' + str('%.6f' % get_average(a_precisions)) + ',' + str('%.6f' % get_average(a_pr_stdev)) + ','
					 + str('%.6f' % get_average(a_recalls)) + ',' + str('%.6f' % get_average(a_re_stdev)) + ','
					 + str('%.6f' % get_average(a_fmeasures)) + ',' + str('%.6f' % get_average(a_fm_stdev)) + ','
					 + str('%.6f' % get_average(a_eers)) + ',' + str('%.6f' % get_average(a_ee_stdev)) + ','
					 + str('%.6f' % get_average(a_eer_thetas)) + ',' + str('%.6f' % get_average(a_ee_th_stdev)) + ','
					 + str('%.6f' % get_average(a_fars)) + ',' + str('%.6f' % get_average(a_fa_stdev)) + ','
					 + str('%.6f' % get_average(a_far_thetas)) + ',' + str('%.6f' % get_average(a_fa_th_stdev)) + ','
					 + str('%.6f' % get_average(a_rejectrate_w)) + ',' + str('%.6f' % get_average(a_rejectrate_b)) + ',' + str('%.6f' % get_average(a_rejectrate_i))
					 )
					output.append(result_string)
					print(result_string + '\n----')
				else:
					sys.exit('ERROR: mode not valid: ' + mode)
				
				outfilename = filename_string + '.csv'
				outfile = open(outfilename, 'w')
				outfile.write('\n'.join(output))
				outfile.close()
				print('OUTPUT: ' + outfilename)