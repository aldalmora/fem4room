"""
Simulate a room with Neumann Boundary conditions.
Compares the frequency response for several points in the room with the theoretical frequencies for the modes.
"""
import fem4room.FEM_3D as fem
from fem4room import Visualization,Other,Simulation,TimeEngine
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import time

c=340.

#Source frequency
f_bands={'50':[50,100]}
simulation_times = {'50':3}
wavelength = c/np.array(f_bands['50'])

#Geometry
Lx=2.1*np.average(wavelength)
Ly=.9*np.average(wavelength)
Lz=.5*np.average(wavelength)

#Modes Neumann
#k**2 = pi^2 ( (nx/Lx)**2 + (ny/Ly)**2 + (nz/Lz)**2 )  
_ev_cube=[]
for i in range(0,10):
    for j in range(0,10):
        for k in range(0,10):
            _ev_cube.append( np.pi**2 * ((i/Lx)**2 + (j/Ly)**2 + (k/Lz)**2))
_ev_cube = np.sqrt(np.sort(_ev_cube))
eig_f = _ev_cube*c/(2*np.pi)

#Simulation Conf.
sName = 'Source'
xS = np.array([Lx/3,Ly/3.1,Lz/1.75])
#Receiver Names
rNames = ['1','2','3','4','5'] 
#Receiver Positions
xR = np.array([[Lx/10,Ly/10,Lz/10],[Lx/1.1,Ly/1.5,Lz/2],[Lx/3,Ly/3,Lz/3],[Lx/4,Ly/2,Lz/1.3],[Lx/2,Ly/4,Lz/3]])

m = fem.Mesh('Mesh')
cTag = m.createCube(Lx,Ly,Lz)
pTag = m.addNamedPoint(xS[0],xS[1],xS[2],sName)
m.model.mesh.embed(pTag[0],[pTag[1]],cTag[0],cTag[1])
for i_xr in range(0,len(xR)):
    pTag = m.addNamedPoint(xR[i_xr,0],xR[i_xr,1],xR[i_xr,2],rNames[i_xr])
    m.model.mesh.embed(pTag[0],[pTag[1]],cTag[0],cTag[1])

simulation = Simulation(simulation_times,sName,rNames)
simulation.setMesh(m)
simulation.setFrequencyBands(f_bands)
# simulation.setParameters(elementsPerWavelength=10)
time_arrays,source_signals,receiver_signals = simulation.run()

plt.figure()
for e in eig_f:
    plt.axvline(x=e)
for i_xr in range(0,len(xR)):
    frequencies,response = simulation.calcFrequencyResponse(i_xr)
    plt.semilogy(frequencies,np.abs(response))
plt.title('Frequency Response')
plt.xlim(f_bands['50'])

plt.show()

# s = s[0:300,:]
# Visualization.showTime(m,['v1'],[s],[m.nodeTags])