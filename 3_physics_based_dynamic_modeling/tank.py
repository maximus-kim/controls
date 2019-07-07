import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# define tank model # apparently "_" and "time" is recognized
def tank(Level,time,valve):
    c = 50.0   # valve coefficient (kg/s / %open)
    rho = 1000.0 # water density (kg/m^3)
    A = 1.0      # tank area (m^2)
    # calculate derivative of the Level
    dLevel_dt = (c/(rho*A)) * valve
    return dLevel_dt

# time span for the simulation for 10 sec, every 0.1 sec
ts = np.linspace(0,10,101)

# empty matrix/list
u_list = []
for value in ts:
    if value < 2:
        u = 0
    elif value < 7:
        u = 100
    else:
        u = 0
    u_list.append(u)


# level initial condition
Level0 = 0

# for storing the results
z = np.zeros(len(ts))

# simulate with ODEINT
for i in range(len(ts)):
    valve = u_list[i]
    y = odeint(tank,Level0,ts,args=(valve,))
    Level0 = y[-1] # take the last point
    z[i] = Level0 # store the level for plotting

# plot results
plt.figure()
plt.subplot(2,1,1)
plt.plot(ts,z,'b-',linewidth=3)
plt.margins(.05)
plt.ylabel('Tank Level')
plt.subplot(2,1,2)
plt.plot(ts,u_list,'r--',linewidth=3)
plt.ylabel('Valve')
plt.xlabel('Time (sec)')
plt.margins(.05)
plt.savefig("tank.png")
plt.show()
