import numpy as np
from numpy import loadtxt, zeros, ones, array, linspace, logspace
from pylab import scatter, show, title, xlabel, ylabel, plot, contour

def load_data():
	#Load the dataset
	data = loadtxt('./dataset/housing.data',dtype=np.float)

	#Plot the data
	scatter(data[:, 5], data[:, 13], marker='o', c='b')
	title('Relationship between RM and price')
	xlabel('Avg number of rooms')
	ylabel('Housing Price')
	show()

	X = data[:, 5]
	y = data[:, 13]

	#number of training samples
	m = y.size

	print("m = ", m)
	print(X)
	print(y)

def main():
	print("Main func")
	
	
load_data()
main()