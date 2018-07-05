import csv
import math
import numpy
import matplotlib.pyplot as plt

#define global array to take data and thetas
tSetX = numpy.empty(shape=(2,100))		 
tSetY = numpy.empty(shape=(1,100))		
theta = numpy.empty(shape=(2,1))
wts = numpy.zeros(shape = (100,100))

#rake in data
def readData():
	fileX = open('weightedX.csv', 'r')
	fileY = open('weightedY.csv', 'r')
	xreader = csv.reader(fileX, delimiter=',')
	yreader = csv.reader(fileY, delimiter=',')
	i=0
	for row in xreader:
		tSetX[0,i] = 1.0
		tSetX[1,i] = float(row[0])
		i += 1
	i=0
	for row in yreader:
		tSetY[0,i] = float(row[0])
		i += 1
	fileX.close()
	fileY.close()

#normalisation
def normalisation():
	tSetX[1,:] = (tSetX[1,:] - numpy.mean(tSetX[1,:]))/numpy.std(tSetX[1,:])

#unweighted analytical solution
def analyticalSolutionUw():
	theta = numpy.linalg.inv(tSetX.dot(tSetX.transpose())).dot(tSetX).dot(tSetY.transpose())
	return theta

#weighted analytical solution
def analyticalSolutionW(x,tau):
	for i in range(0,100):
		wts[i,i] = math.exp(-((x-tSetX[1,i])**2)/2/tau/tau)
	theta = numpy.linalg.inv(tSetX.dot(wts).dot(tSetX.transpose())).dot(tSetX).dot(wts).dot(tSetY.transpose())
	return theta[0,0] + theta[1,0] * x

#print graph for unweighted answer
def printUnweightedPlot():
	plt.xlabel(r"$X$", fontsize = 15)
	plt.ylabel(r"$Y$", fontsize = 15)
	plt.title("Locally Weighted Linear Regression : Decision Boundary")
	plt.scatter(tSetX[1,:].reshape(100,1),tSetY.transpose(),marker="+")
	X = numpy.linspace(-2,2.5,283)
	Y = numpy.empty(shape = (283,))
	i =0
	theta = analyticalSolutionUw()
	for it in X:
		Y[i] = theta[0] + theta[1]*X[i]
		i += 1
	plt.plot(X,Y,label='Linear')	

#print graph for weighted answer
def printWeightedPlot(tauv):
	X = numpy.linspace(-2,2.5,283,endpoint=True)
	Y = numpy.empty(shape = (283,))
	i =0
	for it in X:
		Y[i] = analyticalSolutionW(X[i], tauv)
		i += 1
	plt.plot(X,Y,label = str(tauv))



#Take inputs from files
readData()

#pre-process data to normalise
normalisation()

#print unweighted answer
theta = analyticalSolutionUw()
print("=> The equation without weights is given by --> Y = " + str(round(theta[1,0],7)) +"*X + " + str(round(theta[0,0],7))) 

#plot unweighted graph
printUnweightedPlot()

#print weighted plots

printWeightedPlot(0.1)
printWeightedPlot(0.3)
#printWeightedPlot(0.8)
printWeightedPlot(2)
printWeightedPlot(10)
plt.legend(loc = 'upper left',title="Bandwidth Parameter",frameon=False)
plt.show()