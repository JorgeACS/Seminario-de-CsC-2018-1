import numpy
from pylab import plot, show, grid, xlabel, ylabel,savefig
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from brownian import brownian
# The Wiener process parameter.
delta = 2
# Total time.
T = 30.0
# Number of steps.
N = 500
# Time step size
dt = T/N
# Number of realizations to generate.
m = 5000
# Create an empty array to store the realizations.
x = numpy.empty((m,N+1))
# Initial values of x.
x[:, 0] = 50

brownian(x[:,0], N, dt, delta, out=x[:,1:])



t = numpy.linspace(0.0, N*dt, N+1)

for k in range(m):
    plot(t, x[k])
xlabel('t', fontsize=16)
ylabel('x', fontsize=16)
grid(True)
#show()
    
savefig("lineas.png")
plt.close()
half=N//2

mu = 50
sigma =numpy.std(x[:,-1])
plt.subplot(1, 1, 1)
n, bins, patches =plt.hist(x[:,-1],bins='auto',histtype='stepfilled', facecolor='b',normed=1)
y = mlab.normpdf(bins,mu, sigma)
l = plt.plot(bins,y, 'r--', linewidth=1)
plt.title(f'mu {mu} sigma {sigma}')
plt.ylabel('Frecuencia')


#plt.tight_layout()
plt.savefig("chart.png")
plt.close()