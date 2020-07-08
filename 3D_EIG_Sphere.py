import FEM.FEM_3D as fem
from FEM import Solver,Boundary
import numpy as np
import numpy.linalg as la
import gmsh
import scipy.sparse.linalg as sla
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.io import mmwrite,mmread
import matplotlib.pyplot as plt
import time

sla.use_solver(useUmfpack=True)

class MyInv(sla.LinearOperator):
    def __init__(self, M):
        self.solve = sla.factorized(M)
        self.shape = M.shape

    def _matvec(self, x):
        return self.solve(x)

_ev_sphere = [3.14159,4.49341,5.76346,6.28318,6.98793,7.72525,8.18256,9.09501,9.35581]
errors = []
t_matrices = []
t_eigs = []
nddls = []
_h =[0.09,0.08,0.07,0.06]#
showModes=0
plotErrors=1

def f(x,y,z):
    return 0*x

def g(x,y,z):
    return 0*x

for h in _h:
    m = fem.Mesh('Mesh3D')
    bTag = m.fac.addSphere(0,0,0,1)

    m.fac.synchronize()
    dt_Boundary = m.model.getBoundary((3,bTag),recursive=True)
    m.model.occ.setMeshSize(dt_Boundary,h)
    dt_Surfaces = m.model.getBoundary((3,bTag))
    m.model.addPhysicalGroup(2,np.array(dt_Surfaces)[:,1],1)
    m.fac.synchronize()

    m.generate()

    t1m = time.time()
    engine = fem.Engine(m,1,1)
    K = engine.K_Matrix()
    M = engine.M_Matrix()
    t2m = time.time()
    K,M,F,G_Boundary,ddl_interior_idx,ddl_boundary_idx = Boundary.Apply_Dirichlet(engine,1,K,M,f,g)
    t_matrices.append(t2m-t1m)

    nddls.append(K.shape[0])

    t1e = time.time()
    solver = Solver(engine)
    ev,v = solver.eig(K,M,30,sigma=0,which='LM')
    t2e = time.time()
    t_eigs.append(t2e-t1e)

    print(str(len(engine.ddl)) + ' - Matrices(K,M B.C.) - ' + str(t2m-t1m) + ' - EIGS ' + str(t2e-t1e))

    w, w_indexes = np.unique(np.floor(np.real(np.sqrt(ev*1e3))),return_index=True)
    w = np.sqrt(ev[w_indexes])

    v = v.T[w_indexes]

    l = np.min([len(w),len(_ev_sphere)])
    errors.append(la.norm(np.abs(w[0:l]-_ev_sphere[0:l]))/la.norm(w[0:l]))

    if showModes and len(_h) == 1:
        option = gmsh.option
        v_all = np.zeros((len(w),len(engine.ddl)))
        for i in range(0,len(w)):
            v_all[i,ddl_interior_idx] = np.real(v[i])#/np.max(np.abs(v[i]))
            v_all[i,ddl_boundary_idx] = G_Boundary

        for i in range(0,len(w)):
            viewTag = m.pos.add(str(i))
            m.pos.addModelData(viewTag,0,m.name,'NodeData',m.nodeTags,v_all[i,:].reshape(-1,1),numComponents=1)
            alias = "View["+str(i)+"]."
            option.setNumber(alias + "IntervalsType",1)
            option.setNumber(alias + "Boundary",2)
            option.setNumber(alias + "Visible",0)
            option.setNumber(alias + "RangeType",2)
            option.setNumber(alias + "CustomMin",-1)
            option.setNumber(alias + "CustomMax",1)

        option.setNumber("View[0].Visible",1)

if plotErrors:
    plt.figure()
    plt.loglog(_h,errors)
    plt.loglog(_h,3 * np.array(_h)**2,'k-.')
    plt.legend(['|Error EV|','Ref 2'])
    plt.xlabel('h')
    plt.ylabel('|Error|')

    plt.figure()
    plt.loglog(nddls,t_matrices)
    plt.loglog(nddls,t_eigs)
    plt.loglog(nddls,3 * 10**-4  * np.array(nddls,dtype=np.float64),'k-+')
    plt.loglog(nddls,10**-7  * np.array(nddls,dtype=np.float64)**2,'k-.')
    plt.legend(['K & M', 'EIGS', 'Ref 1', 'Ref 2'])
    plt.xlabel('#ddl')
    plt.ylabel('Time')

    plt.show()

if showModes and len(_h) == 1:
    m.FLTKRun()
