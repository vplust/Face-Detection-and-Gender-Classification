import os
import sys
from svm import *
from svmutil import *


text_file = 'Outputdlibnew.txt'
def svm_read_problem(data_file_name):
	
	prob_y = []
	prob_x = []
	for line in open(data_file_name):
		line = line.split(None, 1)
		print line 
		label, features = line
		xi = {}
		for e in features.split():

			ind, val = e.split(":")
			xi[int(ind)] = float(val)
		prob_y += [float(label)]
		prob_x += [xi]
	return (prob_y, prob_x)


y,x = svm_read_problem(text_file)

text_file_combined = 'combinedoutput.txt'
y_c, x_c = svm_read_problem(text_file_combined)

m = svm_train(y, x, '-c 1')

svm_save_model('dlibnewtraining.model', m)

p_label, p_acc, p_val = svm_predict(y_c, x_c, m)





