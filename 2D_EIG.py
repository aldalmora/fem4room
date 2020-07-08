import numpy as np
import scipy.sparse.linalg as sla
from scipy.sparse.csgraph import reverse_cuthill_mckee
import scipy.special as special
import matplotlib.pyplot as plt
import numpy.linalg as la
import time

import FEM.FEM_2D as fem
from FEM import Boundary

def f(x,y):
    return 0*x

def g(x,y,z):
    return 0*x

r = 1
_h =  [0.1,0.08,0.06,0.04,0.02]
order = 2
formule = 2

errors=[]
t_matrices=[]
t_eigs=[]
ddls=[]

for h in _h:
    m = fem.Mesh('eig_problem')
    (pC,lC,clC,sC) = m.createCircle(0, 0, 0, r, h = h, addInterior=True)
    m.model.addPhysicalGroup(1,lC,1)
    m.generate(order)
    ddls.append(len(m.vertices))
    print('Vertices: ' + str(len(m.vertices)))

    t1 = time.time()

    engine = fem.Engine(m,order,formule)

    M = engine.M_Matrix()

    K = engine.K_Matrix()

    K,M,F,G_Boundary,ddl_interior_idx,ddl_boundary_idx = Boundary.Apply_Dirichlet(engine,1,K,M,f,g)

    #Matrices bandwidth optimization
    idx_rcm = reverse_cuthill_mckee(K)
    K = K[idx_rcm,:]
    K = K[:,idx_rcm]
    M = M[idx_rcm,:]
    M = M[:,idx_rcm]

    t2 = time.time()
    t_matrices.append(t2-t1)
    print('K & M Matrices ' + str(K.shape) + ' - ' + str(t2-t1))

    # plt.figure()
    # A = K+M
    # A.data = np.ones(len(A.data))
    # plt.imshow(A.A,cmap=plt.cm.binary)
    # # #plt.figure()
    # # #plt.spy(M)
    # plt.show()

    t1 = time.time()
    w,v = sla.eigs(K,20,M,sigma=0,which='LM') #Change to Solver
    t2 = time.time()
    t_eigs.append(t2-t1)
    print('EIGS - ' + str(t2-t1))

    w = np.unique(np.floor(np.real(w[w>0]*1e4)))
    w = w[0:10]/1e4
    w = np.sqrt(w)

    zeros = np.zeros(100)
    k=0
    for i in range(0,10):
        _z = special.jn_zeros(i,10)
        for _lz in _z:
            zeros[k] = _lz
            k+=1

    zeros = np.unique(zeros)[0:10]
    err = np.abs(w[0]-zeros[0])/la.norm(w[0])
    errors.append(err)

print(w)
print(zeros)

plt.figure()
plt.loglog(_h,errors)
plt.loglog(_h,10**-1 * np.array(_h),'k--')
plt.loglog(_h,10**-1 * np.array(_h)**2,'k-+')
plt.legend(['Error 1º0','Ref(1)','Ref(2)'])
plt.xlabel('h')
plt.ylabel('|Error|')

plt.figure()
plt.loglog(ddls,t_matrices)
plt.loglog(ddls,t_eigs)
plt.loglog(ddls,5*(10**-6) * np.array(ddls),'k--')
plt.loglog(ddls,5*(10**-8) * np.array(ddls,dtype=np.float64)**2,'k-+')
plt.legend(['Matrices','EIGS','Ref(1)','Ref(2)'])
plt.xlabel('#ddl')
plt.ylabel('t(s)')
 
plt.show()