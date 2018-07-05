import csv
import math
import numpy
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#define global array to take data and thetas
tSetX = numpy.empty(shape=(2,100))		#wine's acidity 
tSetY = numpy.empty(shape=(1,100))		#wine's density
thetas = numpy.empty(shape=(2,100000))
jthetas = []
theta0 = numpy.empty(shape=(2,1))
k = 0
converge = True

#rake in data
def readData():
	fileX = open('linearX.csv', 'r')
	fileY = open('linearY.csv', 'r')
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

#cost function
def costfunc(ptheta):
	jtheta = 0.0
	i = 0
	for i in range(0,100):
		jtheta = jtheta + ( tSetY[0,i] - (ptheta.transpose()).dot(tSetX[:,i]) )**2
	return jtheta/2

#gradient descent algorithm
def gradientDescent(eta,stopCriteria):
	global k
	k = 1 #no.of iteration
	thetas[:,0] = theta0
	global jthetas
	jthetas.append(costfunc(thetas[:,0]))

	while True:
		thetas[:,k] = [0.0,0.0]

		for i in range(0,100):
			thetas[:,k] = thetas[:,k] + ( tSetY[0,i] - (thetas[:,k-1].transpose()).dot(tSetX[:,i]) )*tSetX[:,i]

		thetas[:,k] = thetas[:,k-1] + eta * thetas[:,k]
		jthetas.append(costfunc(thetas[:,k]))
		if (abs(jthetas[k]-jthetas[k-1]) < stopCriteria):
			break
		
		if( k==100000-1 or thetas[0,k]>10000000):
			print "Couldn't converge"
			global converge
			converge = False
			break
		k += 1

#plot line function
def plotLine():
	plt.title("Linear Regression: Best Fit")
	plt.xlabel("X (Wine acidity - normalised)->")
	plt.ylabel("Y (Wine density)->")
	plt.scatter(tSetX[1,:].reshape(100,1),tSetY.transpose())
	X = numpy.linspace(-2,4,256,endpoint=True)
	Y = numpy.empty(shape = (256,))
	i =0
	for it in X:
		Y[i] = thetas[0,k] + thetas[1,k]*X[i]
		i += 1

	if(converge):
		plt.plot(X,Y,color="green")
	plt.show()

def initVarsForPlot():
	global x,y,X,Y,Z
	x = numpy.linspace(-2.5,4,50)
	y = numpy.linspace(-2.5,2.5,50)
	X,Y = numpy.meshgrid(x,y)
	Z = numpy.ndarray(X.shape)
	for i in range(0,50):
		for j in range(0,50):
			theta = numpy.empty(shape=(2,1))
			theta[0,0],theta[1,0] = X[i,j],Y[i,j]
			Z[i,j] = costfunc(theta)	

#plot 3d function
def plot3dGraph():
	plt.ion()
	fig = plt.figure()
	ax = fig.gca(projection = "3d")
	ax.set_title("Working Gradient Descent - 3D Mesh")
	#plt.title("Working Gradient Descent - 3D Mesh")
	surf = ax.plot_surface(X, Y, Z, cmap=cm.Oranges, linewidth=0, antialiased=False, alpha=0.5)
	ax.scatter([thetas[0,k]],[thetas[1,k]],[jthetas[k]],marker = "x", c ="red",s=50)
	fig.colorbar(surf, shrink=0.5, aspect=5)
	ax.set_xlabel(r'$\theta_0$', fontsize = 15)
	ax.set_ylabel(r'$\theta_1$', fontsize = 15)
	ax.set_zlabel("COST")
	for i in range(0,k):
		ax.scatter([thetas[0,i]],[thetas[1,i]],[jthetas[i]],color='k',s=5)
		plt.pause(0.2)
		if(i>70):
			break
	plt.ioff()
	plt.close()


#plot contour function
def plotContour():
	plt.ion()
	plt.xlabel(r'$\theta_0$', fontsize = 15)
	plt.ylabel(r'$\theta_1$', fontsize = 15)
	plt.title("Working Gradient Descent - Contour Plot")
	CS = plt.contour(X,Y,Z)
	plt.clabel(CS, inline=1,fontsize=10)
	plt.scatter(thetas[0,k],thetas[1,k],marker = "x", c ="red")
	for i in range(0,k):
		plt.scatter([thetas[0,i]],[thetas[1,i]],color='k',s=5)
		plt.pause(0.2)
		#if(i>250):
		#	break
	plt.ioff()
	plt.show()

#take input
readData()

#normalisation
normalisation()

#choose parameters and start variables
theta0 = [-2,2] 
n = 0.0005  #learning rate
e = 1e-11   #stopping criteria

#application
gradientDescent(n , e)

#print answers
print("=> With initialised parameters --> " + str(theta0))
print("=> Stopping Criteria --> " + str(e))
print("=> Learning Rate --> " + str(n))
if(converge):
	print("=> Number of iterations taken --> " + str(k))
	print("=> Equation obtained --> Y = " +str(thetas[0,k])+" + "+str(thetas[1,k]) +"*X")

print("**Close the popped up graphs to see further graphs**")

#plot line
plotLine()

#initialise variables for 3d plots
initVarsForPlot()

#plot 3d mesh
plot3dGraph()

#plot contour
plotContour()