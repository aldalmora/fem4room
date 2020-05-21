import numpy as np
import scipy.sparse as sparse

from .Boundary import Boundary

class Engine:
    def __init__(self,mesh,order,formule):
        self.order = order
        self.formule = formule
        self.mesh = mesh
        self.Boundary = Boundary(self)
        self.ddl,self.numDdl,self.u,self.dxu,self.dyu,self.omega = self.initializeMatrices('Lagrange')
        
    def IntegrationPointsHat(self,formule: int):
        """Return the integration points of the reference triangle"""
        xhat = []
        yhat = []
        omega = []
        if (formule==1):
            xhat = [1/3]
            yhat = [1/3]
            omega = [1/2]
        elif (formule==2):
            xhat = [1/2,1/2,0]
            yhat = [1/2,0,1/2]
            omega = [1/6,1/6,1/6]
        elif (formule==3):
            xhat = [1/6,1/6,2/3]
            yhat = [1/6,2/3,1/6]
            omega = [1/6,1/6,1/6]
        elif (formule==4):
            xhat = [1/3,1/5,3/5,1/5]
            yhat = [1/3,1/5,1/5,3/5]
            omega = [-9/32,25/96,25/96,25/96]
        elif (formule==5):
            xhat = [1/2,1/2,0,1/6,1/6,2/3]
            yhat = [1/2,0,1/2,1/6,2/3,1/6]
            omega = [1/60,1/60,1/60,3/20,3/20,3/20]
        elif (formule==6):
            a = (6 + np.sqrt(15))/21
            b = (6 - np.sqrt(15))/21
            A = (155 + np.sqrt(15))/2400
            B = (155 - np.sqrt(15))/2400
            xhat = [1/3,a,1-2*a,a,b,1-2*b,b]
            yhat = [1/3,a,a,1-2*a,b,b,1-2*b]
            omega = [9/80,A,A,A,B,B,B]
        return (np.array(xhat),np.array(yhat),np.array(omega))

    def initializeMatrices(self, typeEF):
        """Generate the matrices with the values of the base functions, the dx, the dy and the weights for integration."""
        mesh,order,formule = self.mesh, self.order, self.formule
        xhat, yhat, omegaloc = self.IntegrationPointsHat(formule)

        if (typeEF=='Lagrange'):
            phi, dxphi, dyphi = self.LagrangeFunctions(xhat, yhat, order)

            if (order==0):
                for triangle in mesh.triangles:
                    coord_v1 = mesh.vertices[triangle[0]]
                    coord_v2 = mesh.vertices[triangle[1]]
                    coord_v3 = mesh.vertices[triangle[2]]

                    ddl.append([(coord_v1[0]+coord_v2[0]+coord_v3[0])/3,
                                (coord_v1[1]+coord_v2[1]+coord_v3[1])/3])
                    numDdl.append([len(ddl)-1])
                numDdl = np.array(numDdl)
                ddl = np.array(ddl)
            elif (order==1 or order==2): #GMSH generate the nodes
                ddl = mesh.vertices
                numDdl = np.array(mesh.triangles,dtype=np.int32)

            mesh.calcJacobians()
            detJ = mesh.determinants
            J_inv = mesh.inverses
                
            dxhat_dx = J_inv[:,0,0]
            dxhat_dy = J_inv[:,0,1]
            dyhat_dx = J_inv[:,1,0]
            dyhat_dy = J_inv[:,1,1]

            N_ddl = len(ddl)
            N_int = len(mesh.triangles)*len(xhat)
            omega = np.zeros(N_int)
            # u = sparse.lil_matrix((N_int,N_ddl))
            # dxu = sparse.lil_matrix((N_int,N_ddl))
            # dyu = sparse.lil_matrix((N_int,N_ddl))

            #The sparse matrices will have for each line a integration point and each
            #column a ddl. The number of integration points is the number of elements
            #times the number of points per element. So, for each point of integration
            #we have a certain amount of base functions to evaluate, the amount of 
            #ddl per element.
            #Structure for sparse Matrix
            _i = np.zeros(N_int*len(phi[0]))
            _j = np.zeros(N_int*len(phi[0]))
            _v_u = np.zeros(N_int*len(phi[0]))
            _v_dxu = np.zeros(N_int*len(phi[0]))
            _v_dyu = np.zeros(N_int*len(phi[0]))

            for iloc in range(0,len(xhat)): #Points of Integration
                iglob = np.arange(iloc,N_int,len(xhat),dtype=np.int32)
                omega[iglob] = omegaloc[iloc] * (detJ)
                for jloc in range(0,len(phi[0])): #Base functions
                    _idx = jloc*N_int + iloc*len(iglob) + np.arange(0,len(iglob)) #Global stacked index for the sparse matrix
                    jglob = numDdl[:,jloc]
                    _i[_idx] = iglob
                    _j[_idx] = jglob
                    _v_u[_idx] = np.ones(len(iglob))*phi[iloc,jloc]
                    _v_dxu[_idx] = (dxphi[iloc,jloc]*dxhat_dx + dyphi[iloc,jloc]*dyhat_dx)
                    _v_dyu[_idx] = (dxphi[iloc,jloc]*dxhat_dy + dyphi[iloc,jloc]*dyhat_dy)
                    # u[iglob,jglob] = phi[iloc,jloc]
                    # dxu[iglob,jglob] = (dxphi[iloc,jloc]*dxhat_dx + dyphi[iloc,jloc]*dyhat_dx) 
                    # dyu[iglob,jglob] = (dxphi[iloc,jloc]*dxhat_dy + dyphi[iloc,jloc]*dyhat_dy)
                    
            u = sparse.coo_matrix((_v_u, (_i, _j)),(N_int,N_ddl))
            dxu = sparse.coo_matrix((_v_dxu, (_i, _j)),(N_int,N_ddl))
            dyu = sparse.coo_matrix((_v_dyu, (_i, _j)),(N_int,N_ddl))
            u = u.tocsc()
            dxu = dxu.tocsc()
            dyu = dyu.tocsc()

        return (np.array(ddl),np.array(numDdl),u,dxu,dyu,omega)

    def LagrangeFunctions(self, x, y, order):
        """ Get the values and of the base functions and its derivatives at the triangle of reference. """
        phi = []
        dxphi = []
        dyphi = []
        if (order==0):
            for i in range(0,len(x)):
                phi.append([1.])
                dxphi.append([0.])
                dyphi.append([0.])
        elif (order==1): #Ordering (0,0), (1,0) et (0,1)
            for i in range(0,len(x)):
                phi.append([1-x[i]-y[i],x[i],y[i]])
                dxphi.append([-1.,1.,0.])
                dyphi.append([-1.,0.,1.])
        elif (order==2): #Ordering (0,0), (1,0), (0,1), (.5,0), (.5,.5) et (0,.5)
            for i in range(0,len(x)):
                phi.append([(1-x[i]-y[i])*(1-2*x[i]-2*y[i]),
                            x[i]*(2*x[i]-1),
                            y[i]*(2*y[i]-1),
                            4*x[i]*(1-x[i]-y[i]),
                            4*x[i]*y[i],
                            4*y[i]*(1-x[i]-y[i])])
                dxphi.append([4*y[i]+4*x[i]-3,
                                4*x[i]-1,            
                                0,
                                -4*(2*x[i] + y[i] - 1),          
                                4*y[i],
                                -4*y[i]])
                dyphi.append([4*y[i]+4*x[i]-3,    
                                0,            
                                4*y[i]-1,               
                                -4*x[i],            
                                4*x[i],                
                                -4*(x[i] + 2*y[i] - 1)])
        return np.array(phi), np.array(dxphi), np.array(dyphi)
    
    def M_Matrix(self,c=1):
        """ Generate de mass matrix """
        u,omega  = self.u,self.omega
        M = u.transpose().dot(u.transpose().multiply(omega).transpose())
        M = M.tocsc()

        #Garantee int64 to UMFPACK
        M.indices = M.indices.astype(np.int64)
        M.indptr = M.indptr.astype(np.int64)
        return M/(c**2)

    def K_Matrix(self):
        """ Generate de stiffness matrix """
        dxu,dyu,omega  = self.dxu,self.dyu,self.omega
        G1 = dxu.transpose().dot(dxu.transpose().multiply(omega).transpose())
        G2 = dyu.transpose().dot(dyu.transpose().multiply(omega).transpose())
        K = G1+G2
        K = K.tocsc()

        #Garantee int64 to UMFPACK
        K.indices = K.indices.astype(np.int64)
        K.indptr = K.indptr.astype(np.int64)
        return K

    def F_Matrix(self, f):
        """ Generate de forcing matrix """
        ddl,u,omega  = self.ddl,self.u,self.omega
        fh = u.multiply(f(ddl[:,0],ddl[:,1])).transpose()
        return fh.dot(omega)