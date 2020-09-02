"""
    Check the finite elements convergence for a 2D eigenproblem (circle of radius 1) with Dirichlet b.c. .
"""

import numpy as np
import scipy.sparse.linalg as sla
import scipy.sparse as sparse
import scipy.special as special
import matplotlib.pyplot as plt
import numpy.linalg as la
import time

import fem4room.FEM_2D as fem
from fem4room import Boundary

_h = [0.09,0.08,0.07,0.06]
orders = [1,2]

#Store the values for plotting
errors_p1=[]
errors_p2=[]
t_matrices=[]
t_eigs=[]
dofs=[]

for h in _h:
    for order in orders:
        #Mesh a circle
        m = fem.Mesh('eig_problem')
        (pC,lC,clC,sC) = m.createCircle(0, 0, 0, radius=1, h = h, addInterior=True)
        m.model.addPhysicalGroup(1,lC,1)
        m.generate(order)
        print('Vertices: ' + str(len(m.vertices)))

        t1_m = time.time()
        engine = fem.Engine(m,order,order)
        M = engine.M_Matrix()
        K = engine.K_Matrix()
        C = sparse.csc_matrix(M.shape)
        f = lambda time_index: engine.dof[:,0]*0
        g = lambda time_index: engine.dof[:,0]*0
        M, C,K,F,G_Boundary,dof_interior_idx,dof_boundary_idx = Boundary.Apply_Dirichlet(engine,1,M,C,K,f,g)
        t2_m = time.time()
        print('K & M Matrices ' + str(K.shape) + ' - ' + str(t2_m-t1_m))

        t1_e = time.time()
        w,v = sla.eigs(K,30,M,sigma=0,which='LM') #TODO: Change to Solver
        t2_e = time.time()
        print('EIGS - ' + str(t2_e-t1_e))

        #Get the first 10 only-positive eigenvalues, unique up to 5 decimal places
        w = np.unique(np.floor(np.real(w[w>0]*1e5)))
        w = w[0:10]/1e5
        w = np.sqrt(w)

        #The zeros of the bessel function
        zeros = np.zeros(100)
        k=0
        for i in range(0,10):
            _z = special.jn_zeros(i,10)
            for _lz in _z:
                zeros[k] = _lz
                k+=1
        zeros = np.unique(zeros)[0:10]

        #Get the error for the first eigenvalue
        err = np.abs(w[0]-zeros[0])/la.norm(w[0])

        if order==1:
            errors_p1.append(err)
            dofs.append(len(m.vertices))
            t_matrices.append(t2_m-t1_m)
            t_eigs.append(t2_e-t1_e)
        else:
            errors_p2.append(err)

plt.figure()
plt.loglog(_h,errors_p1)
plt.loglog(_h,errors_p2)
plt.loglog(_h,10**-1 * np.array(_h)**2,'k-+') #Reference line
plt.legend(['Error(1)','Error(2)','$h^{-2}$'])
plt.xlabel('h')
plt.ylabel('|Error|')

plt.figure()
plt.loglog(dofs,t_matrices)
plt.loglog(dofs,t_eigs)
plt.loglog(dofs,5*(10**-6) * np.array(dofs),'k--') #Reference line
plt.loglog(dofs,5*(10**-8) * np.array(dofs,dtype=np.float64)**2,'k-+') #Reference line
plt.legend(['Matrices','EIGS','Ref(1)','Ref(2)'])
plt.xlabel('#dof')
plt.ylabel('t(s)')
 
plt.show()