import numpy as np # abbreviate numpy as np
from scipy.integrate import odeint # imports odeint type form scipy
import matplotlib.pyplot as plt # imports plots from plotting library

# function that returns dy/dt
def model(z,t,u):
    x = z[0]
    y = z[1]
    dxdt = (-x+u)/2
    dydt = (-y+x)/5
    dzdt = [dxdt, dydt]
    return dzdt

# initial condition
x0 = 0
y0 = 0
z0 = [x0, y0]

# time points
n = 401
t = np.linspace(0,40,n)
u = np.zeros(n)
# at t = 5 seconds, u is equal to two
u[51:] = 2.0
x = np.zeros(n)
y = np.zeros(n)
# solve ODE
for i in range(1,n):
    tspan = [t[i-1],t[i]]
    z = odeint(model,z0,t,args=(u[i],))
    z0 = z[1]
    x[i] = z0[0]
    y[i] = z0[1]

# plot results
plt.plot(t,u,'g:',label='u(t)')
plt.plot(t,x,'b-',label='x(t)')
plt.plot(t,y,'r--',label='y(t)')
plt.ylabel('values')
plt.xlabel('time')
plt.legend(loc='best')
plt.margins(.05)
# save figure runs plot and saves image, needs to be done before show (plt.show removes prior figures)
plt.savefig("odeint_P4.png")
plt.show()
