import numpy as np
import matplotlib.pylab as plt

#weights = np.matrix([[0.1, -0.1],[-0.1,0.1]])
weights = np.matrix([[20., -21.],[-0.77,0.77]])
bias    = np.array([-1.5,-1.5])
x       = np.random.uniform(-1,1,size=1000000*2).reshape(1000000,2)
newX    = []


for val in x:
    newX += [np.matrix(val) * weights + bias]

newX = np.array(newX,dtype=np.float32)
classOne = newX.T[0][0]
print classOne

print len(x.T[0]),len(x.T[1]), len(classOne)
plt.hist2d(x.T[0],x.T[1], weights=classOne, bins=20)
plt.show()
