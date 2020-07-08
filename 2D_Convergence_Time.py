import numpy as np
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

import FEM.FEM_2D as fem2d
from FEM.Boundary import Boundary
from scipy.sparse.csgraph import reverse_cuthill_mckee

L = 2*np.pi
_h = [0.1,0.08,0.04,0.03]
orders = [1,2]
formule = 2
plot=0
plot_errors=1
plot_time=0

qtd_vtx = []
t_init = []
t_K = []
t_M = []
t_Dirichlet = []
t_Solver = []

def g(x,y,z):
    return x

def f(x,y): 
    return x + 3*np.sin(x)*np.sin(y) # exact (1/3)cos(x)cos(y)

def f_exact(x,y):
    return x + np.sin(x)*np.sin(y)

def grad_exact(x,y):
    dx = np.cos(x)*np.sin(y) + 1
    dy = np.sin(x)*np.cos(y)
    return [dx,dy]

_L2_o1 = []
_H1_o1 = []
_L2_o2 = []
_H1_o2 = []

for order in orders:
    for h in _h:

        m = fem2d.Mesh('TP6')
        (pS,lS,clS) = m.createSurface([[0,0,0],[L,0,0],[L,L,0],[0,L,0]],h=h)
        sS = m.fac.addPlaneSurface([clS])
        m.model.addPhysicalGroup(1,lS,1)

        #(pS,lS,clS) = m.createSurface([[0,0,0],[L,0,0],[L,L,0],[0,L,0]],h= h)
        #(pC,lC,clC,sC) = m.createCircle(L/2, L/2, 0, L/6, h = 0.3*h, addInterior=True) 
        #sS = m.fac.addPlaneSurface([clS,clC])
        #m.fac.synchronize()
        #m.model.addPhysicalGroup(1,lS,1)

        m.generate(order)

        t1 = time.time()

        qtd_vtx.append(len(m.vertices))
        t1_init = time.time()
        engine = fem2d.Engine(m,order,formule)
        t2_init = time.time()
        t_init.append(t2_init-t1_init)
        
        t1_K = time.time()
        K = engine.K_Matrix()
        t2_K = time.time()
        t_K.append(t2_K-t1_K)

        t1_M = time.time()
        M = engine.M_Matrix()
        t2_M = time.time()
        t_M.append(t2_M-t1_M)

        t1_Dirichlet = time.time()
        K,M,F,G_Boundary,ddl_interior_idx,ddl_boundary_idx = Boundary.Apply_Dirichlet(engine, 1, K, M, f, g)
        t2_Dirichlet = time.time()
        t_Dirichlet.append(t2_Dirichlet-t1_Dirichlet)
        A = K + M

        t1_Solver = time.time()
        idx_rcm = reverse_cuthill_mckee(A)
        A = A[idx_rcm,:]
        A = A[:,idx_rcm]
        F = F[idx_rcm]

        S_ = sla.spsolve(A,F)
        t2_Solver = time.time()
        t_Solver.append(t2_Solver-t1_Solver)

        ddl = engine.ddl
        S = np.zeros(len(ddl))
        S[ddl_boundary_idx] = G_Boundary
        S[ddl_interior_idx] = S_[np.argsort(idx_rcm)]
        t2 = time.time()

        S_exact = np.array([f_exact(ddl[i,0],ddl[i,1]) for i in range(0,len(ddl))])
        g_exact = [grad_exact(ddl[i,0],ddl[i,1]) for i in range(0,len(ddl))]

        print('order ' + str(order) + '| h ' + str(h) + '| ddls ' + str(len(ddl)) + '| time ' + str(t2 - t1))

        u,dxu,dyu,omega = engine.u,engine.dxu,engine.dyu,engine.omega
        if order==1:
            _L2_o1.append(fem2d.calcL2(u,omega,S,S_exact))
            _H1_o1.append(fem2d.calcH1(u,dxu,dyu,omega,S,S_exact,g_exact))
        elif order==2:
            _L2_o2.append(fem2d.calcL2(u,omega,S,S_exact))
            _H1_o2.append(fem2d.calcH1(u,dxu,dyu,omega,S,S_exact,g_exact))


        if plot:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot_trisurf(ddl[:,0],ddl[:,1],S)
            ax.plot_trisurf(ddl[:,0],ddl[:,1],S_exact)


            plt.show()

if plot_errors:
    plt.figure()
    plt.loglog(_h,_L2_o1)
    plt.loglog(_h,_H1_o1)
    plt.loglog(_h,_L2_o2)
    plt.loglog(_h,_H1_o2)
    plt.loglog(_h,2*(10**-1) * np.array(_h),'k-.')
    plt.loglog(_h,5*(10**-2) * np.array(_h)**2,'k--')
    plt.loglog(_h,5*(10**-3) * np.array(_h)**4,'k-+')
    plt.legend(['L2(1)','H1(1)','L2(2)','H1(2)','Ref 1','Ref 2','Ref 3'])
    plt.xlabel('h')
    plt.ylabel('|Error|')
    ax = plt.gca()
    ax.invert_xaxis()
    plt.show()

if plot_time and len(orders)==1:
    plt.figure()
    plt.loglog(qtd_vtx,t_init)
    plt.loglog(qtd_vtx,t_K)
    plt.loglog(qtd_vtx,t_M)
    plt.loglog(qtd_vtx,t_Dirichlet)
    plt.loglog(qtd_vtx,t_Solver)
    plt.loglog(qtd_vtx,5*(10**-6) * np.array(qtd_vtx),'k--')
    plt.loglog(qtd_vtx,5*(10**-8) * np.array(qtd_vtx,dtype=np.float64)**2,'k-+')
    plt.legend(['Init','K','M','Dirichlet','Solver','Ref(1)','Ref(2)'])
    plt.xlabel('#ddl')
    plt.ylabel('t (s)')


    plt.show()