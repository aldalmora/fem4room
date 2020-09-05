"""
    Simulate an impedance tube.
    Test the theory
"""
import fem4room.FEM_3D as fem
import fem4room.FEM_2D as fem2d
from fem4room import Sources,Visualization,Other,Boundary,Solver,TimeEngine
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
import time

c=340.
Z = 3 + 5j 
R = np.abs((Z-1)/(Z+1)) * np.cos(np.angle((Z-1)/(Z+1)))

#Source frequency
frequency=np.array([300.])
wavelength=c/frequency

#Geometry
Lx=10*np.max(wavelength)
r=np.max(wavelength)/5
h=np.min(wavelength)/10 #At least a certain amount of elements per wavelength
impedance_pg=1
source_pg=2
xR = [Lx - 2*np.max(wavelength)] #

m = fem.Mesh('Mesh3D')
#bTag = m.fac.addBox(0,0,0,Lx,Ly,Lz)
bTag = m.fac.addCylinder(0,r,r,Lx,0,0,r)

m.fac.synchronize()
dt_Boundary = m.model.getBoundary((3,bTag),recursive=True)
m.model.mesh.setSize(dt_Boundary,h)
dt_Surfaces = m.model.getBoundary((3,bTag))
m.model.addPhysicalGroup(2,np.array(dt_Surfaces)[1:2,1],impedance_pg)
m.model.addPhysicalGroup(2,np.array(dt_Surfaces)[2:3,1],source_pg)
pointReceiver = [m.fac.addPoint(_xr,r,r) for _xr in xR]
m.fac.synchronize()
m.model.mesh.embed(0,pointReceiver,3,bTag)
m.generate()

#Simulation Conf.
order = 1
cfl = .8
tf = 1.8*Lx/c
dt=cfl*(h)/c #CFL Condition

engine = fem.Engine(m,order,order)
dofs=engine.dof
M = engine.M_Matrix()/(c**2)
K = engine.K_Matrix()

idx_xR = [Other.nearest_dof(dofs,[_xr,r,r]) for _xr in xR]

#Initial
ndofs = len(engine.dof)
tspan = np.arange(0,tf,dt)

############# Time Engine
f = lambda time_index: 0*engine.dof[:,0]
g = lambda time_index: 0*engine.dof[:,0] + np.sin(2*np.pi*frequency*tspan[time_index])

M_2d,nodeTags_imp = Boundary.Surface_Mass_Matrix(m,impedance_pg)
nodeTags_idx = np.where(nodeTags_imp[:,None]==m.nodeTags)[1]
idx_row = nodeTags_idx[M_2d.row]
idx_col = nodeTags_idx[M_2d.col]
C = sparse.coo_matrix((M_2d.data, (idx_row, idx_col)), M.shape).tocsc()
C = C/(c*Z)
# C = sparse.csc_matrix(M.shape)

M,C,K,F,G_Boundary,dof_interior_idx,dof_boundary_idx = Boundary.Apply_Dirichlet(engine,source_pg,M,C,K,f,g)

return_dofs = np.arange(0,K.shape[0])

tengine = TimeEngine.Newmark_iterative(M,C,K,dt,alpha=.25,delta=.5)
s,s_main = tengine.solve(tspan,F,[],return_dofs,1)

s_full = np.zeros([len(tspan),len(engine.dof)])
s_full[:,dof_boundary_idx] = [g(i)[dof_boundary_idx] for i in range(0,len(tspan))]
s_full[:,dof_interior_idx] = s
s = s_full

wavelength_samples = int(np.round((2*wavelength/c)/dt))
estimated_incident=np.max(s[-wavelength_samples:,idx_xR])/(1+R)

plt.figure()
plt.plot(tspan, s[:,idx_xR])
plt.axhline(estimated_incident,c='b')
plt.xlabel('t(s)')
plt.ylabel('Pressure')
plt.legend(['Simulated','Theoretical Incident Wave Amplitude'])
plt.show()
# Visualization.showTime(m,['v'],[s],[np.arange(0,ndofs)],fltkrun=True)