import csv
import math
import time
import numpy
from numpy.linalg import inv
from numpy.linalg import det
import matplotlib.pyplot as plt

#define global array to take data and thetas
tSetX = numpy.empty(shape=(2,100))		
tSetY = numpy.empty(shape=(1,100))		
thetas = numpy.empty(shape=(2,1))
k = 0

#rake in data
def takeData():
	fileX = open('q4x.dat', 'r')
	fileY = open('q4y.dat', 'r')
	xreader = csv.reader(fileX, delimiter=' ')
	yreader = csv.reader(fileY, delimiter=' ')
	i=0
	for row in xreader:
		tSetX[0,i] = float(row[0])
		tSetX[1,i] = float(row[2])
		i += 1
	i=0
	for row in yreader:
		tSetY[0,i] = 0 if (row[0] == "Alaska") else 1	#Alaska = 0, Canada = 1
		i += 1

	fileX.close()
	fileY.close()

#normalisation
def normalisation():
	tSetX[0,:] = (tSetX[0,:] - numpy.mean(tSetX[0,:]))/numpy.std(tSetX[0,:])
	tSetX[1,:] = (tSetX[1,:] - numpy.mean(tSetX[1,:]))/numpy.std(tSetX[1,:])

#mean values u0 u1
def phi():
	count,phi=0,0
	frac = 0.0
	for i in range(0,100):
		if(tSetY[0,i]==1):
			phi += 1
		count += 1
	return phi/float(count)

def mean(a):
	count=0
	mean = numpy.zeros(shape=(2,1))
	for i in range(0,100):
		if(tSetY[0,i]==a):
			mean += tSetX[:,i].reshape(2,1)
			count += 1
	return mean/count

def covMatrix():
	count=0
	covMat = numpy.zeros(shape=(2,2))
	for i in range(0,100):
		covMat += (tSetX[:,i].reshape(2,1)-mean(tSetY[0,i])).dot((tSetX[:,i].reshape(2,1)-mean(tSetY[0,i])).transpose())
		#print covMat
	return covMat/100

def covMatrixG(a):
	count=0
	covMat = numpy.zeros(shape=(2,2))
	for i in range(0,100):
		if tSetY[0,i] == a:
			covMat += (tSetX[:,i].reshape(2,1)-mean(a)).dot((tSetX[:,i].reshape(2,1)-mean(a)).transpose())
			count += 1
	#print count 
	return covMat/count

# AX = B , where A = (u0 -u1)T(E-1 + E-1T)  and B = u0T*E-1*u0 - u1T*E-1*u1
def solveBoundaryLinear():
	A = numpy.empty(shape=(1,2))
	B = numpy.empty(shape=(1,1))
	A = ((mean(0)-mean(1)).transpose()).dot( inv(covMatrix()).transpose() + inv(covMatrix()) )
	B = (mean(0).transpose()).dot(inv(covMatrix())).dot(mean(0))-(mean(1).transpose()).dot(inv(covMatrix())).dot(mean(1))
	theta = [-B[0,0],A[0,0],A[0,1]]		
	return theta

# XT*A*X + B*X = G where A = (u0 -u1)T(E-1 + E-1T)  and B = u0T*E-1*u0 - u1T*E-1*u1
def solveBoundaryQuad():
	A = numpy.empty(shape=(2,2))
	B = numpy.empty(shape=(1,2))
	G = numpy.empty(shape=(1,1))
	
	A = inv(covMatrixG(1)) - inv(covMatrixG(0))
	B = mean(0).transpose().dot(inv(covMatrixG(0)) + inv(covMatrixG(0))) - mean(1).transpose().dot(inv(covMatrixG(1)) + inv(covMatrixG(1)))	
	G = -numpy.log(det(covMatrixG(1))/det(covMatrixG(1))) + \
	(mean(0).transpose()).dot(inv(covMatrixG(0))).dot(mean(0))-(mean(1).transpose()).dot(inv(covMatrixG(1))).dot(mean(1))

	theta = [A[0,0] , A[0,1]+A[1,0] , A[1,1] , B[0,0] , B[0,1] , G[0,0] ]		
	return theta

def plotTrainingData():
	i=0
	X1p,X2p, X1m, X2m = [],[],[],[]
	for i in range(0,100):
		if(tSetY[0,i] == 0):
			X1p.append(tSetX[0,i])
			X2p.append(tSetX[1,i])
		else:
			X1m.append(tSetX[0,i])
			X2m.append(tSetX[1,i])
		i = i + 1
	plt.scatter(X1p,X2p,marker="+",label="Alaska, y=0") # + -> Alaska
	plt.scatter(X1m,X2m,marker="_",label="Canada, y=1") # - -> Canada

def plotLinearBoundary():
	X1 = numpy.linspace(-2.5,2.5,256,endpoint=True)
	X2 = numpy.empty(shape = (256,))
	i =0
	for it in X1:
	 	X2[i] = -(thetas[0] + thetas[1]*X1[i])/thetas[2]
	 	i += 1
	plt.plot(X1,X2,label=r"$\Sigma_0 = \Sigma_1$")
	plt.xlabel(r"$X_1$ - Growth ring diameter in fresh water (normalised) ", fontsize = 14)
	plt.ylabel(r"$X_2$ - Growth ring diameter in marine water (normalised) ", fontsize = 14)

def plotQuadraticBoundary():
	x1 = numpy.linspace(-2.5,2.5,256)
	x2 = numpy.linspace(-2.5,2.5,256)
	X1,X2 = numpy.meshgrid(x1,x2)

	Z = thetas[0]*X1**2 + X1*X2*(thetas[1]) + thetas[2]*X2**2 + thetas[3]*X1 + thetas[4]*X2 -thetas[5]
	plt.contour(X1,X2,Z,[0])
	plt.legend(loc = 'upper left',title="Legend")#,frameon=False)
	plt.show()

#read data from dat files
takeData()

#normalisation
normalisation()

#print linear solutions
thetas = solveBoundaryLinear()
print("\n**Solution assuming identical covariance matrices**")
print("=> phi -->\n " + str(phi()) )
print("=> U0 --> \n" + str(mean(0)) )
print("=> U1 --> \n" + str(mean(1)) )
print("=> E0 = E1 --> \n" + str(covMatrix()))
print("=> Equation obtained -->\n " + " X2*" + str( round( thetas[2],7 ) )+ " + X1*" + str(round(thetas[1],7)) +" + "+ str(round(thetas[0],7)) + " = 0" )
#print("=> Equation obtained --> "thetas)

#plot training data
plotTrainingData()

#plot line for linear boundary
plotLinearBoundary()

#print quadratic solution
thetas = solveBoundaryQuad()
print("\n**General Solution**")
print("=> phi -->\n" + str(phi()) )
print("=> U0 --> \n" + str(mean(0)))
print("=> U1 --> \n" + str(mean(1)))
print("=> E0 -->\n" + str(covMatrixG(0)))
print("=> E1 -->\n" + str(covMatrixG(1)))
print("=> Equation obtained -->\n X1^2({}) + X1*X2({}) + X2^2({}) + X1({}) + X2({}) = {}".format(round(thetas[0],7),round(thetas[1],7),round(thetas[2],7),round(thetas[3],7),round(thetas[4],7),round(thetas[5],7) ))

#plot quadratic boundary
plotQuadraticBoundary()
