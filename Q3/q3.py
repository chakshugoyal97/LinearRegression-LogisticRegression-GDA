import csv
import math
import numpy
import matplotlib.pyplot as plt

#define global array to take data and thetas
tSetX = numpy.empty(shape=(3,100))		
tSetY = numpy.empty(shape=(1,100))		
thetas = numpy.empty(shape=(3,1000))
k = 0

#rake in data
def takeData():
	fileX = open('logisticX.csv', 'r')
	fileY = open('logisticY.csv', 'r')
	xreader = csv.reader(fileX, delimiter=',')
	yreader = csv.reader(fileY, delimiter=',')
	i=0
	for row in xreader:
		tSetX[0,i] = 1.0
		tSetX[1,i] = float(row[0])
		tSetX[2,i] = float(row[1])
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
	tSetX[2,:] = (tSetX[2,:] - numpy.mean(tSetX[2,:]))/numpy.std(tSetX[2,:])

#hypothesis function
def hthetax(ptheta,i):
	return 1/(1+math.exp(-ptheta.transpose().dot(tSetX[:,i])))

#Hessian matrix calculation
def hessian(ptheta):
	hessMatrix = numpy.empty(shape=(3,3))
	for j in range(0,3):
		for k in range(0,3):
			hessMatrix[j,k] = 0
			for i in range(0,100):
				hessMatrix[j,k] += (1-hthetax(ptheta,i))*(hthetax(ptheta,i))*tSetX[j,i]*tSetX[k,i]
			hessMatrix[j,k] = -hessMatrix[j,k]
	return hessMatrix

#delta matric calculation
def delta(ptheta):
	delTheta = numpy.empty(shape=(3,1))
	for j in range(0,3):
		delTheta[j,0] = 0
		i=0
		for i in range(0,100):
			delTheta[j,0] += (tSetY[0,i] - hthetax(ptheta,i))*tSetX[j,i]
	return delTheta

#newton's algorithm
def newtonsAlgorithm(stopCriteria):
	global k
	k = 1 #no.of iteration
	thetas[:,0]=startTheta

	while True:

		thetas[:,k] = thetas[:,k-1] - numpy.linalg.inv(hessian(thetas[:,k-1])).dot(delta(thetas[:,k-1])).reshape(3,)

		if (abs((thetas[:,k]-thetas[:,k-1]).all()) < stopCriteria):
			break
		
		if(k==1000-1):
			print "Couldn't converge"
			break
		k += 1

#plot training data
def plotTrainingData():
	X1p,X2p, X1m, X2m = [],[],[],[]
	for i in range(0,100):
		if(tSetY[0,i] == 0):
			X1p.append(tSetX[1,i])
			X2p.append(tSetX[2,i])
		else:
			X1m.append(tSetX[1,i])
			X2m.append(tSetX[2,i])
		i = i + 1

	plt.scatter(X1p,X2p,marker="+",label=r"$Y=0$")
	plt.scatter(X1m,X2m,marker="_",label=r"$Y=1$")

#plot boundary
def plotLine():
	X1 = numpy.linspace(-3,3,256,endpoint=True)
	X2 = numpy.empty(shape = (256,))
	i =0
	for it in X1:
	 	X2[i] = -(thetas[0,k] + thetas[1,k]*X1[i])/thetas[2,k]
	 	i += 1

	plt.plot(X1,X2)
	plt.title("Logistic Regression")
	plt.xlabel(r"$X_1$", fontsize = 15)
	plt.ylabel(r"$X_2$", fontsize = 15)
	plt.legend(loc = 'upper left',title="CLASSES")
	plt.show()

#Read Data
takeData()

#pre-process data
normalisation()

#initialise
e = 1e-7
startTheta = [0.0,0.0,0.0]

#running for solutions and print to console
newtonsAlgorithm(e)
print("=> With initialised parameters --> " + str(startTheta))
print("=> Stopping Criteria --> " + str(e))
print("=> Number of iterations taken --> " + str(k))
print("=> Equation obtained --> " + " X2*" 
	+ str( round( thetas[2,k],7 ) ) + " + X1*" + str(round(thetas[1,k],7)) +" + "+ str(round(thetas[0,k],7)) + " = 0" )

#plot training points
plotTrainingData()

#plot boundary line
plotLine()