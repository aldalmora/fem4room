import fem4room.FEM_3D as fem
from fem4room import Boundary,Solver
import numpy as np
import numpy.linalg as la
import gmsh
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import time
import sys

errors = []
t_matrices = []
t_eigs = []
nddls = []
printEigs=0
showModes=0
plotErrors=1

Lx=1
Ly=1.1
Lz=1.2
_h = [0.09,0.08,0.07,0.06]

#Exact eigenvalues for a cube
_ev_cube=[]
for i in range(1,4):
    for j in range(1,4):
        for k in range(1,4):
            _ev_cube.append(np.sqrt( (i*np.pi/Lx)**2 + (j*np.pi/Ly)**2 + (k*np.pi/Lz)**2 ))
_ev_cube = np.sort(_ev_cube)[0:9]

if printEigs:
    print(_ev_cube)

def writeStatus(status):
    sys.stdout.write('Status: ' + status + '                  \r')
    
for h in _h:
    writeStatus('Creating Mesh.')
    m = fem.Mesh('Mesh3D')
    bTag = m.fac.addBox(0,0,0,Lx,Ly,Lz)

    m.fac.synchronize()
    dt_Boundary = m.model.getBoundary((3,bTag),recursive=True)
    m.model.occ.setMeshSize(dt_Boundary,h)
    dt_Surfaces = m.model.getBoundary((3,bTag))
    m.model.addPhysicalGroup(2,np.array(dt_Surfaces)[:,1],1)
    m.fac.synchronize()

    m.generate()
    writeStatus('Mesh generated #ddl: ' + str(len(m.vertices)))

    t1m = time.time()
    writeStatus('Initializing matrices.')
    engine = fem.Engine(m,1,1)
    writeStatus('Calculating K Matrix.')
    K = engine.K_Matrix()
    writeStatus('Calculating M Matrix.')
    M = engine.M_Matrix()
    writeStatus('Applying Dirichlet.')
    C = sparse.csc_matrix(M.shape)

    f = lambda time_index: 0*engine.ddl[:,0]
    g = lambda time_index: 0*engine.ddl[:,0]
    M,C,K,F,G_Boundary,ddl_interior_idx,ddl_boundary_idx = Boundary.Apply_Dirichlet(engine,1,M,C,K,f,g)
    t2m = time.time()
    t_matrices.append(t2m-t1m)

    nddls.append(K.shape[0])

    t1e = time.time()
    writeStatus('EIG.')
    solver = Solver(engine)
    ev,v = solver.eig(K,M,30,sigma=0,which='LM')
    t2e = time.time()
    t_eigs.append(t2e-t1e)

    print(str(len(engine.ddl)) + ' - Matrices(K,M B.C.) - ' + str(t2m-t1m) + ' - EIGS ' + str(t2e-t1e))

    w, w_indexes = np.unique(np.floor(np.real(ev[ev>0]*1e5)), return_index=True)
    w = w[0:10]/1e5
    w = np.sqrt(w)
    err = np.abs(w[0]-_ev_cube[0])/la.norm(w[0])
    errors.append(err)

    v = v.T[w_indexes]
    if printEigs:
        print('----- EIGS ' + str(len(engine.ddl)) + ' -----')
        print(w[0:10])
        print('---------------------------')

    if showModes and len(_h) == 1:
        option = gmsh.option
        v_all = np.zeros((len(w),len(engine.ddl)))
        for i in range(0,len(w)):
            v_all[i,ddl_interior_idx] = np.real(v[i])/np.max(np.abs(v[i]))
            v_all[i,ddl_boundary_idx] = G_Boundary(0)

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
    plt.loglog(_h,7 * np.array(_h)**2,'k-.')
    plt.legend(['|$\epsilon$ eigenvalues|','$h^{-2}$'])
    plt.xlabel('h')
    plt.ylabel('|Error|')
    ax = plt.gca()
    ax.invert_xaxis()

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
