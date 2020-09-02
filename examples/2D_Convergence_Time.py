"""
    Check the finite elements convergence for a partial differential equation in 2D.
"""

import numpy as np
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import time

import fem4room.FEM_2D as fem2d
from fem4room import Boundary,Solver

_h = [0.09,0.08,0.07,0.06]
orders = [1,2]
formule = 2

plot_errors=True
plot_time=False

plot_time=plot_time and len(orders)==1 #Plot time only if there is only one order to be tested

#Store results for plotting
qtd_vtx = []
t_init = []
t_K = []
t_M = []
t_Dirichlet = []
t_Solver = []
L2_o1 = []
H1_o1 = []
L2_o2 = []
H1_o2 = []

#Exact solution
L = 2*np.pi
def f_exact(x,y):
    return x + np.sin(x)*np.sin(y)

#Gradient of the exact solution (for H1 norm)
def grad_exact(x,y):
    dx = np.cos(x)*np.sin(y) + 1
    dy = np.sin(x)*np.cos(y)
    return [dx,dy]

for order in orders:
    for h in _h:
        #Squared Mesh L x L
        m = fem2d.Mesh('Mesh')
        (pS,lS,clS) = m.createSurface([[0,0,0],[L,0,0],[L,L,0],[0,L,0]],h=h)
        sS = m.fac.addPlaneSurface([clS])
        m.model.addPhysicalGroup(1,lS,1)
        m.generate(order)

        t1_total = time.time()
        qtd_vtx.append(len(m.vertices))
        t1_init = time.time()
        engine = fem2d.Engine(m,order,formule)
        dofs=engine.dof
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
        C = sparse.csc_matrix(M.shape)

        t1_Dirichlet = time.time()

        f = lambda time_index: dofs[:,0] + 3*np.sin(dofs[:,0])*np.sin(dofs[:,1])
        g = lambda time_index: dofs[:,0]

        M,C,K,F,G_Boundary,dof_interior_idx,dof_boundary_idx = Boundary.Apply_Dirichlet(engine, 1, M, C, K, f, g)
        t2_Dirichlet = time.time()
        t_Dirichlet.append(t2_Dirichlet-t1_Dirichlet)
        A = K + M

        t1_Solver = time.time()
        S_ = sla.spsolve(A,F(0))
        t2_Solver = time.time()
        t_Solver.append(t2_Solver-t1_Solver)

        #Maps the solution to the whole domain as the Dirichlet b.c. has removed them
        dof = engine.dof
        S = np.zeros(len(dof))
        S[dof_boundary_idx] = G_Boundary(0)
        S[dof_interior_idx] = S_

        t2_total = time.time()

        #The exact solution at the dofs
        S_exact = np.array([f_exact(dof[i,0],dof[i,1]) for i in range(0,len(dof))])
        g_exact = [grad_exact(dof[i,0],dof[i,1]) for i in range(0,len(dof))]

        print('order ' + str(order) + '| h ' + str(h) + '| dofs ' + str(len(dof)) + '| time ' + str(t2_total - t1_total))

        #Calculate the norms
        u,dxu,dyu,omega = engine.u,engine.dxu,engine.dyu,engine.omega
        if order==1:
            L2_o1.append(fem2d.calcL2(u,omega,S,S_exact))
            H1_o1.append(fem2d.calcH1(u,dxu,dyu,omega,S,S_exact,g_exact))
        elif order==2:
            L2_o2.append(fem2d.calcL2(u,omega,S,S_exact))
            H1_o2.append(fem2d.calcH1(u,dxu,dyu,omega,S,S_exact,g_exact))

if plot_errors:
    plt.figure()
    plt.loglog(_h,L2_o1,'-')
    plt.loglog(_h,H1_o1,'-')
    plt.loglog(_h,L2_o2,'-+')
    plt.loglog(_h,H1_o2,'-+')
    #Plot the reference lines
    plt.loglog(_h,8*(10**-1) * np.array(_h),'k:')
    plt.loglog(_h,9*(10**-2) * np.array(_h)**2,'k-.')
    plt.loglog(_h,9*(10**-3) * np.array(_h)**4,color='grey',linestyle=':')
    plt.legend(['L2(1)','H1(1)','L2(2)','H1(2)','$h^{-1}$','$h^{-2}$','$h^{-3}$'],loc='lower left')
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
    #Plot the reference lines
    plt.loglog(qtd_vtx,5*(10**-6) * np.array(qtd_vtx),'k--')
    plt.loglog(qtd_vtx,5*(10**-8) * np.array(qtd_vtx,dtype=np.float64)**2,'k-+')
    plt.legend(['Init','K','M','Dirichlet','Solver','Ref(1)','Ref(2)'])
    plt.xlabel('#dof')
    plt.ylabel('t (s)')
    plt.show()