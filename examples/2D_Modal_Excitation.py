"""
Use impulsive sources with characteristic frequencies and positions.
Tries to get the modes of the 2D domain evolving in time.
"""

import fem4room.FEM_2D as fem
from fem4room import Sources,Visualization,Other,TimeEngine,Solver
import numpy as np
import numpy.linalg as la
import scipy.sparse as sparse
import time

c=340.

#Geometry
Lx=13; Ly=7

#Simulation Conf.
order=1; h=.3

#Get the modes for the Square Lx x Ly
m = fem.Mesh.MeshSquare('eig',Lx,Ly,h,order)
t1 = time.time()
engine = fem.Engine(m,order,order)
dofs=engine.dof
M = engine.M_Matrix()/(c**2)
K = engine.K_Matrix()
C = sparse.csc_matrix(M.shape)
slv = Solver(engine)
k2,v = slv.eig(K,M,30,0,'LM')
t2 = time.time()
print('Time EIG - ' + str(t2-t1))

#Get the first modes, unique up to 4 decimal places
idx_positive = np.where(k2>0)[0]
v = v[:,idx_positive]
k2 = k2[idx_positive]
k2,idx = np.unique(np.floor(np.real(k2*1e4)),return_index=True)
k2 = k2[0:5]/1e4
k = np.sqrt(k2)
v = v[:,idx]
eig_f = k/(2*np.pi)

#For each eigenfrequency, simulates a source in its maximum value node
for i in range(0,len(eig_f)):
    fS=eig_f[i]

    #Add the view of the mode
    s = v.T[i].reshape((1,len(v)))
    Visualization.addView(m,'{:.2f}'.format(eig_f[i]),s,m.nodeTags)

    #Get the maximum value node of the mode
    idx_sort = np.argsort(v.T[i])
    load_dof = idx_sort[10]
    dof_eig_source = engine.dof[load_dof]

    #Simulate in time
    cfl=.8
    sigma_t=4/fS 
    wavelength=c/fS
    dt=cfl*h/c #CFL Condition
    tf=10*sigma_t
    dof_source = dofs[Other.nearest_dof(dofs,dof_eig_source)]

    ############# Time Engine
    tengine = TimeEngine.Newmark_iterative(M,C,K,dt,alpha=.25,delta=.5)

    #Initial
    ndofs = len(engine.dof)
    return_dofs = np.unique(m.triangles)
    tspan = np.arange(0,tf,dt)

    ################# Source Term
    dof_select = np.all(dofs==dof_source,axis=1)*1
    time_monopole = Sources.monopole_by_band(tspan,[fS],sigma=sigma_t)
    f = lambda time_index: time_monopole[time_index] * dof_select

    s,s_main = tengine.solve(tspan,f,[],return_dofs,1)

    #Get only the last period of the simulation
    cycle_steps=int(np.round((1/fS)/dt))
    s = s[-cycle_steps:,:]
    s = s.reshape((1,len(s),len(s[0])))
    Visualization.showTime(m,['time ' + '{:.2f}'.format(eig_f[i])],s,[return_dofs],fltkrun=False)


m.FLTKRun()

