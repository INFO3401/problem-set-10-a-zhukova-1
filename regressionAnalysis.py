# Anastasiya Zhukova
# INFO 3401 - Monday & Wednesday Problems
# Worked with Kexin Zhai, Jacob Paul
# Text asnwers to the questions can be found at the bottom on the file

import csv
import pprint
import numpy as np
import pandas as pd 
from sklearn import linear_model
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score


class AnalysisData:
	
	def __init__(self):
		self.dataset = []
		self.variables = []
		
	def parser(self, filename):
		self.dataset = pd.read_csv(filename, encoding = 'latin1')
		self.variables = self.dataset.columns.values


try2 = AnalysisData()
try2.parser('candy-data.csv')

#https://stackoverflow.com/questions/46092914/sklearn-linearregression-could-not-convert-string-to-float
#https://towardsdatascience.com/simple-and-multiple-linear-regression-in-python-c928425168f9
#http://www.datasciencemadesimple.com/get-list-column-headers-column-name-python-pandas/
 

class LinearAnalysis:

	def __init__(self, targetY):
		self.targetY = targetY
		self.bestX = []
		self.fit = 0

# run a regression for every column first = that's your x, separate out each column and run an regression on it in order
# get the R^2 ofeach column, record the best one
# THEN run a new regression on the your target variable and compare to the output of your target variable 
	def runSimpleAnalysis(self, data):
		candy_df = data.dataset
		target = pd.DataFrame(candy_df, columns = [self.targetY])

		for variable in candy_df:
			if variable != 'competitorname' and variable != self.targetY:
				#print(variable)

				X = pd.DataFrame(candy_df, columns =[variable])
				lm = linear_model.LinearRegression()
				lm.fit(X, target)
				predict_val = lm.predict(X)
				r2val = r2_score(target, predict_val)
				if r2val > self.fit:
					self.fit = r2val
					self.bestX = variable
					print(self.fit, self.bestX)
					print('Linear Simp Analysis coefs: ', lm.coef_)
					print('Linear Simp Analysis intercept: ', lm.intercept_)


data = AnalysisData()
data.parser('candy-data.csv')

sugarLinAl = LinearAnalysis('sugarpercent')
sugarLinAl.runSimpleAnalysis(data)
chocLinAl = LinearAnalysis('chocolate')
chocLinAl.runSimpleAnalysis(data)



class LogisticAnalysis:

	def __init__(self, targetY):
		self.targetY = targetY
		self.bestX = []
		self.fit = 0

	def runSimpleAnalysis(self, data):
		candy_df = data.dataset
		target = pd.DataFrame(candy_df, columns = [self.targetY])

		for variable in candy_df:
			if variable != 'competitorname' and variable != self.targetY:
				#print(variable)

				X = pd.DataFrame(candy_df, columns =[variable])
				lm = linear_model.LogisticRegression()
				lm.fit(X, target)
				predict_val = lm.predict(X)
				r2val = r2_score(target, predict_val)
				if r2val > self.fit:
					self.fit = r2val
					self.bestX = variable
					print(self.fit, self.bestX)
					print('Logistic Simp Analysis coefs: ', lm.coef_)
					print('Logistic Simp Analysis intercept: ', lm.intercept_)



	def runMultipleRegression(self, data):
		candy_df = data.dataset
		target = pd.DataFrame(candy_df, columns = [self.targetY])
		X = []

		for val in data.dataset.columns.values:
			if val != 'competitorname' and val != self.targetY:
				X.append(val)

		lm = linear_model.LogisticRegression()
		#data.dataset[X] says to pull the data from data.dataset that matches columns found in X
		lm.fit(data.dataset[X], target)
		predict_val = lm.predict(data.dataset[X])
		r2val = r2_score(target, predict_val)
		if r2val > self.fit:
			self.fit = r2val
			print(self.fit, self.bestX)
			print('Mult Logistic Regression coefs: ',lm.coef_)
			print('Mult Logistic Regression intercept: ',lm.intercept_)




LogAl = LogisticAnalysis('chocolate')
LogAl.runSimpleAnalysis(data)
LogAl.runMultipleRegression(data)


###########################
#####Text Answers##########
###########################

#1. Do the two functions find the same optimal variable? Which method best fits this data?

# When comparing the performance of the runSimpleAnalysis function in the LinearAnalysis class vs the LogisticAnalysis class
# on the variable chocolate, both functions find the same optimal variable (fruity) that predicts whether or not a candy is
# chocolate, but the Linear Analysis function fits the data better with a resulting fit of 0.5501 while the Logistic Analysis 
# results in a fit of 0.4257.


# 2. Compare the outcomes of multiple logistic analysis and the simple logistic analysis. Which model best fits the data? Why? 
# The multiple logistic analysis fits the data better (fit: 0.7607) probably because you are using more points from the whole dataset
# to run the analysis rather than running an analsysis one variable/column at a time. This give the function more data points to
# work with as it tries to find a good fit for the logistic regression line.


# 3. Write the equations for your linear, logistic, and multiple logistic regressions.

# Linear simple analysis function, dependent variable chocolate
# y = -0.73964166x + 0.76595745

# Logistic simple analysis, dependent variable chocolate
# p = 1/(1 + e^-(-2.94124604x + 0.78465691))

# Logistic Multiple regression, dependent variable chocolate
# p = 1/(1 + e^-(-2.52858047x + -0.19697876x + 0.03940308x + -0.16539952x + 0.49783674x + -0.47591613x + 
# 0.81511886x + -0.59971553x + -0.2581028x + 0.3224988x + 0.05387906x + -1.68260553))




		