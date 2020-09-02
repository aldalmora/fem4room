"""

"""
import fem4room.FEM_3D as fem
from fem4room import Other,Simulation,Boundary
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import time

c=340.
abs_coef=0.14
ref_coef=np.sqrt(1-abs_coef)
Z = (1+ref_coef)/(1-ref_coef)

#Source frequency
band={'100':[50,100]}
impedance_pg=1

#Geometry
Lx=4
Ly=3
Lz=3

# Lx=4
# Ly=4
# Lz=5
# abs_coef=0.9
# S=13 + 8*0.8

#Sabine
S = 2*Lx*Ly + 2*Ly*Lz + 2*Lx*Lz
T60 = 24 * np.log(10) * (Lx*Ly*Lz) / (c*S*abs_coef)
print('T60: ' + str(T60))

#Simulation Conf.
order = 1
cfl = .8
tf = 2*T60
Z = (1+ref_coef)/(1-ref_coef)
xS = np.array([1,2.5,0.91])
xR = np.array([1.88,0.95,0.91])

m = fem.Mesh('Mesh')
bTag = m.createCube(Lx,Ly,Lz)
m.fac.synchronize()
dt_Surfaces = m.model.getBoundary(bTag)
m.model.addPhysicalGroup(2,np.array(dt_Surfaces)[:,1],impedance_pg)
m.model.setPhysicalName(2,impedance_pg,'Wall')
pointSource = m.addNamedPoint(xS[0],xS[1],xS[2],'Source')[1]
pointReceiver = m.addNamedPoint(xR[0],xR[1],xR[2],'Receiver')[1]
m.fac.synchronize()
m.model.mesh.embed(0,[pointReceiver,pointSource],3,bTag[1])
m.fac.synchronize()

simulation = Simulation({'100':tf},'Source',['Receiver'])
simulation.setImpedance('Wall',{'100':Z})
simulation.setMesh(m)
simulation.setFrequencyBands(band)
time_arrays,source_signals,receiver_signals = simulation.run()

arg_max = np.argmax(source_signals['100'])
time_max = time_arrays['100'][arg_max]
s = receiver_signals['100'][0]

plt.figure()
sdB = 20*np.log10(np.abs(s)/np.max(s))
plt.plot(time_arrays['100'],sdB)
plt.axvline(T60+time_max,color='green')
plt.axhline(-60,color='red')
plt.legend(['Pressure','T60 Sabine','-60dB'])
plt.xlabel('t(s)')
plt.ylabel('|Normalized Pressure| (dB)')
plt.ylim([-70,3])

plt.show()