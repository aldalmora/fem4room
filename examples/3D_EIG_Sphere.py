"""
    Test the convergence of the eigenproblem for a sphere with Dirichlet b.c. .
"""

import fem4room.FEM_3D as fem
from fem4room import Solver,Boundary
import numpy as np
import numpy.linalg as la
import gmsh
import scipy.sparse as sparse
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt
import time

_h = [0.09,0.08,0.07,0.06]

plotErrors=True
showModes=True
showModes=showModes and len(_h) == 1 #Show modes only if there is only one step-size simulated

#Store values for plotting
errors = []
t_matrices = []
t_eigs = []
ndofs = []

#Eigenvalues for the sphere (zeros of the bessel functions)
_ev_sphere = np.array([3.14159,4.49341,5.76346,6.28318,6.98793,7.72525,8.18256,9.09501,9.35581])

for h in _h:
    #Create a sphere and tag its surfaces
    m = fem.Mesh('Mesh3D')
    bTag = m.fac.addSphere(0,0,0,1)
    m.fac.synchronize()
    dt_Boundary = m.model.getBoundary((3,bTag),recursive=True) #Get the nodes from the boundary/surface
    m.model.mesh.setSize(dt_Boundary,h)
    dt_Surfaces = m.model.getBoundary((3,bTag))
    m.model.addPhysicalGroup(2,np.array(dt_Surfaces)[:,1],1)
    m.fac.synchronize()
    m.generate()

    #Prepare fem4room and generate the FEM matrices
    t1m = time.time()
    engine = fem.Engine(m,1,1)
    K = engine.K_Matrix()
    M = engine.M_Matrix()
    C = sparse.csc_matrix(M.shape)
    t2m = time.time()
    f = lambda time_index: 0*engine.dof[:,0]
    g = lambda time_index: 0*engine.dof[:,0]
    M,C,K,F,G_Boundary,dof_interior_idx,dof_boundary_idx = Boundary.Apply_Dirichlet(engine,1,M,C,K,f,g)
    t_matrices.append(t2m-t1m)

    t1e = time.time()
    solver = Solver(engine)
    ev,v = solver.eig(K,M,30,sigma=0,which='LM')
    t2e = time.time()
    t_eigs.append(t2e-t1e)
    print(str(len(engine.dof)) + ' - Matrices(K,M B.C.) - ' + str(t2m-t1m) + ' - EIGS ' + str(t2e-t1e))

    #Get the first 10 only-positive eigenvalues, unique up to 5 decimal places
    w, w_indexes = np.unique(np.floor(np.real(ev[ev>0]*1e5)), return_index=True)
    w = w[0:10]/1e5
    w = np.sqrt(w)
    
    #Get the error of the first eigen-value
    err = np.abs(w[0]-_ev_sphere[0])/la.norm(w[0])
    ndofs.append(K.shape[0])
    errors.append(err)

    v = v.T[w_indexes]
    if showModes and len(_h) == 1:
        option = gmsh.option
        v_all = np.zeros((len(w),len(engine.dof)))
        for i in range(0,len(w)):
            v_all[i,dof_interior_idx] = np.real(v[i])#/np.max(np.abs(v[i]))
            v_all[i,dof_boundary_idx] = G_Boundary(0)

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
    plt.legend(['|$\epsilon$ eigenvalues|','$h^{-2}$'])
    plt.xlabel('h')
    plt.ylabel('|Error|')

    plt.figure()
    plt.loglog(ndofs,t_matrices)
    plt.loglog(ndofs,t_eigs)
    plt.loglog(ndofs,3 * 10**-4  * np.array(ndofs,dtype=np.float64),'k-+')
    plt.loglog(ndofs,10**-7  * np.array(ndofs,dtype=np.float64)**2,'k-.')
    plt.legend(['K & M', 'EIGS', 'Ref 1', 'Ref 2'])
    plt.xlabel('#dof')
    plt.ylabel('Time')

    plt.show()

if showModes and len(_h) == 1:
    m.FLTKRun()
