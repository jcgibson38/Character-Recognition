import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D

### Configure Plot ###
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.set_xlabel('Ink')
ax.set_ylabel('Aspect')
ax.set_zlabel('Right Heaviness')
ax.set_axis_bgcolor((43./255,48./255,59./255))
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.zaxis.label.set_color('white')
[a.set_color('white') for a in ax.xaxis.get_ticklabels()]
[a.set_color('white') for a in ax.yaxis.get_ticklabels()]
[a.set_color('white') for a in ax.zaxis.get_ticklabels()]

def plotpoints(X,classes,ax):
    scattercolors = {'_m':'m','_r':'c','09':'y','_8':'b','_9':'g'}
    for i,myclass in enumerate(classes):
        ax.scatter(X[i,0],X[i,1],X[i,2],c=scattercolors[myclass],marker='o',s=40)

def plotplane(W,ax,color='r',alpha=0.50):
    # Generate xs and ys. #
    xx,yy = np.meshgrid(np.arange(0,.18,.01),np.arange(0.2,2.,0.1))
    # Calculate zs #
    zz = (-W[0]*xx-W[1]*yy-W[3])*1./W[2]
    # Plot #
    ax.plot_surface(xx,yy,zz,color=color,alpha=alpha)
####################

# Read in the training set #
pngs = glob.glob('pla_training_set/*.png')

# Initialize Variables #
np.random.seed(173)
n = len(pngs) # Number of data points #
d = 3 # Dimensions #
X = np.empty((n,d+1)) # Contains data for all pngs #
Y = [] # Contains corresponding class of the ith png in X #

for k,png in enumerate(pngs):
    # Load the image #
    pngimage = Image.open(png)
    pngarray = np.array(pngimage)
    # Get the alpha channel of the current image #
    myalpha = pngarray[:,:,0]
    myalpha = np.array(255-myalpha,dtype=float)
    h,w = myalpha.shape
    # Dimension 1 -> total ink #
    allblack = 255.*h*w
    myink = myalpha.sum()
    myweightedink = myink/allblack
    # Dimension 2 -> aspect ratio of character #
    x = np.linspace(0,w,w,endpoint=False)
    y = np.linspace(h,0,h,endpoint=False)
    xgrid,ygrid = np.meshgrid(x,y)
    xmin = xgrid[ myalpha>0 ].min()
    xmax = xgrid[ myalpha>0 ].max()
    ymin = ygrid[ myalpha>0 ].min()
    ymax = ygrid[ myalpha>0 ].max()
    width = xmax-xmin
    height = ymax-ymin
    myaspect = height/width
    # Dimension 3 -> right heaviness of character #
    middle = (xmax+xmin)/2.
    rightofmiddle = xgrid>middle
    myrightheaviness = ( myalpha[rightofmiddle] ).sum()/myink
    # Classification of pngs #
    myclass = re.findall('_(.{2})\.png',png)
    Y += myclass
    # Store info #
    X[k,:] = np.array([myweightedink,myaspect,myrightheaviness,1.])
##########################

plotpoints(X,Y,ax)
#plt.show()


### Training ###
CY = list(set(Y))
numclasses = len(CY) # Number of seperable classes #
Ws = np.empty((numclasses-1,d+1))

# We need to loop through (#classes-1) #
# Set classification of each point #
for i in range(numclasses-1):
    YY = np.ones(len(Y))
    for j in range(len(Y)):
        if Y[j] != CY[i]:
            YY[j] *= -1
    # PLA Step #
    # Initial Guess of [a,b,c,d] #
    W = np.array([1.,1.,1.,1.])
    while True:
        # Classify each point based on current W #
        WTX = np.dot(W,X.T)
        # If all points are correctly classified we can break #
        if all(np.sign(WTX)==YY):
            break
        # We need a misclassified point #
        imis = np.argmax( np.sign(WTX) != YY ) # This just gets the first misclassified point #
        # Update W to include this point #
        W += YY[imis]*X[imis,:] # PLA step #
    # Log W #
    Ws[i,:] = W
    #plotplane(W,ax,'mediumseagreen')
#plotpoints(X,Y,ax)
#plt.show()
############################

### Testing results ###
# Test results on full set #
testpngs = glob.glob('pngs/*__m.png')
classY = np.empty( (len(testpngs),Ws.shape[0]) )
for k,png in enumerate(testpngs):
    # Load the image #
    pngimage = Image.open(png)
    pngarray = np.array(pngimage)
    # Get the alpha channel of the current image #
    myalpha = pngarray[:,:,0]
    myalpha = np.array(255-myalpha,dtype=float)
    h,w = myalpha.shape
    # Dimension 1 -> total ink #
    allblack = 255.*h*w
    myink = myalpha.sum()
    myweightedink = myink/allblack
    # Dimension 2 -> aspect ratio of character #
    x = np.linspace(0,w,w,endpoint=False)
    y = np.linspace(h,0,h,endpoint=False)
    xgrid,ygrid = np.meshgrid(x,y)
    xmin = xgrid[ myalpha>0 ].min()
    xmax = xgrid[ myalpha>0 ].max()
    ymin = ygrid[ myalpha>0 ].min()
    ymax = ygrid[ myalpha>0 ].max()
    width = xmax-xmin
    height = ymax-ymin
    myaspect = height/width
    # Dimension 3 -> right heaviness of character #
    middle = (xmax+xmin)/2.
    rightofmiddle = xgrid>middle
    myrightheaviness = ( myalpha[rightofmiddle] ).sum()/myink
    # Log info #
    myX = np.array([myweightedink,myaspect,myrightheaviness,1.])

    # Plot classified points #
    for i in range(len(Ws)):
        classY[k,i] = np.dot(Ws[i,:],myX.T)
    #if classY[k,3] > 0:
        #ax.scatter(myX[0],myX[1],myX[2],c='b',marker='o',s=40)
    #else:
        #ax.scatter(myX[0],myX[1],myX[2],c='r',marker='o',s=40)
###########################
#plotplane(Ws[3,:],ax,'mediumseagreen')
#plt.show()
print len(classY[:,0])
print np.sum(classY[:,0]>0)
