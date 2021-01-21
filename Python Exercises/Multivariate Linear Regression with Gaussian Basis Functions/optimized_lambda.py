import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

def gaussianBasis(data,mean,num1,num2):
	#intializing variables
	features = data.shape[1]
	samples = num2
	#initializing final basis matrix (+1 for the bias terms)
	#and covariance matrix
	basis = np.ones((samples, num1+1))
	covariance = 2*np.identity(features)
	for j in range(0, num1+1):
		for i in range(0,samples):
			if j == 0:
				basis[i,j] = 1
			else:
				X = data.iloc[i].to_numpy()
				MU = mean.iloc[j-1].to_numpy()
				A = X - MU
				covariance_inv = np.linalg.inv(covariance)
				A_T = A.transpose()
				temp = A.dot(covariance_inv)
				temp = temp.dot(A_T)
				temp = (-0.5) * temp
				temp = np.exp(temp)
				basis[i,j] = temp
	return basis

def weightsCalc_l2reg(basis,target,lamda):
	lam_identity = lamda*np.identity(basis.shape[1])
	basis_T = basis.transpose()
	w = np.linalg.inv(lam_identity + basis_T.dot(basis))
	w = w.dot(basis_T)
	w = w.dot(target)

	return w

def weightsCalc(basis,target):
	basis_T = basis.transpose()
	w = np.linalg.inv(basis_T.dot(basis))
	w = w.dot(basis_T)
	w = w.dot(target)
	return w

def RMS_error(predictions,target,num1):
	err = predictions - target
	sse = (np.sum(np.square(err))) / num1
	RMS_err = np.sqrt(sse)
	return RMS_err



#Read the dataset
data = pd.read_csv('auto-mpg.csv')

#Remove the car name column
data = data.drop(labels='car name', axis=1)


#Assigning and normalizing the target values in the mpg column of the dataset
target_vals = data['mpg']
target_vals_norm = stats.zscore(target_vals)

target_vals_norm = pd.DataFrame(target_vals_norm)

#getting the number of samples, training and testing values
samples = len(target_vals)		
train_num = 100
test_num = samples - train_num


#Assigning the training target values
target_train = target_vals_norm.head(train_num)	

#Assigning the test target values
target_test = target_vals_norm.tail(test_num)

#Remove the car name column
data = data.drop(labels='mpg', axis=1)

#Zscore normalization of the dataset (mean = 0, std = 1) for all the features
data_norm = stats.zscore(data)
data_norm = pd.DataFrame(data_norm)

#Assigning the training data values
data_train = data_norm.head(train_num)

#Assigning the testing data values
data_test = data_norm.tail(samples-train_num)

#initializing lists for storing the RMS training and testing errors 
#for the number of basis functions used
RMS_train = np.zeros((10,1))
RMS_test = np.zeros((10,1))
lamda = 0.055
basis_num = 90


#randomly sampling training data points for the
# centers of the gaussian basis functions
means = data_train.sample(n=basis_num, random_state=1)

#Creating the gaussian basis functions for the training data
basis_train = gaussianBasis(data_train,means,basis_num,train_num)

#Creating the weight/coefficients vector
weights = weightsCalc_l2reg(basis_train,target_train,lamda)

#Creating the gaussian basis functions for the test data
basis_test = gaussianBasis(data_test,means,basis_num,test_num)

#Creating the test predictions based on the weights calcualted from the training
test_predictions_regularized = basis_test.dot(weights)

#Calculating the RMS test error using the known target values and the predicted values from the data 
error_regularized = RMS_error(test_predictions_regularized, target_test,test_num)

#Comparing performance with linear regression without regularization or gaussian basis functions 
#Fitting the model with training data
reg = LinearRegression().fit(data_train, target_train)

#Calculating predictions for the test data using the fitted model
test_predictions = reg.predict(data_test)

#Calculating the RMS error for the linear regression model
error = RMS_error(test_predictions, target_test,test_num)

#printing the error for the two models
print('RMS_err_regularized: ', error_regularized[0])
print('RMS_err: ', error[0])