import numpy as np
import scipy.sparse as sparse
import sys


class Engine:
    def __init__(self,mesh,order,formule,calcForMass3D=False):
        """Initialize the engine and assembles the support matrices 

        :param mesh: Instance of the mesh.
        :type mesh: FEM_2D.Mesh
        :param order: Order of the elements. [1 or 2]
        :type order: int
        :param formule: Formule for the numerical integration. Recommended >= (order)
        :type formule: int
        :param calcForMass3D: True if the mesh is in 3D space. Applies only for assembling the mass matrix. defaults to False
        :type calcForMass3D: bool, optional
        """
        self.order = order
        self.formule = formule
        self.mesh = mesh
        self.calcForMass3D = calcForMass3D
        self.dof,self.numDof,self.u,self.dxu,self.dyu,self.omega = self.initializeMatrices('Lagrange')
        
    def IntegrationPointsHat(self,formule: int):
        """Return the integration points of the reference triangle

        :param formule: Formule for the integration. [1 to 6]
        :type formule: int
        :return: (x,y,omega) ; x and y being the coordinates at the reference domain. Omega the weight of integration.
        :rtype: Tuple of arrays
        """
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
        """Generate the matrices with the values of the base functions, the dx, the dy and the weights for integration.

        :param typeEF: Type of inteporlation. Only 'Lagrange'.
        :type typeEF: string
        :return: (dof,numDof,u,dxu,dyu,omega);
        dof is the position of the degrees of freedom
        numDof is the dofs that composes each element
        u is the value of the interpolation function for each integration point
        dxu is the derivative in x
        dyu is the derivative in y
        omega is the quadrature weight for each integration point
        :rtype: (Array n x 3,Array e x 3,Matrix i x e,Matrix i x e,Array i); 
        n being the number of degrees of freedom; 
        e being the number of elements;
        i being the number of integration points
        """
        mesh,order,formule = self.mesh, self.order, self.formule
        xhat, yhat, omegaloc = self.IntegrationPointsHat(formule)

        if (typeEF=='Lagrange'):
            phi, dxphi, dyphi = self.LagrangeFunctions(xhat, yhat, order)

            if (order==0):
                for triangle in mesh.triangles:
                    coord_v1 = mesh.vertices[triangle[0]]
                    coord_v2 = mesh.vertices[triangle[1]]
                    coord_v3 = mesh.vertices[triangle[2]]

                    dof.append([(coord_v1[0]+coord_v2[0]+coord_v3[0])/3,
                                (coord_v1[1]+coord_v2[1]+coord_v3[1])/3])
                    numDof.append([len(dof)-1])
                numDof = np.array(numDof)
                dof = np.array(dof)
            elif (order==1 or order==2): #GMSH generate the nodes
                dof = mesh.vertices
                numDof = np.array(mesh.triangles,dtype=np.int32)

            if (not self.calcForMass3D):
                mesh.calcJacobians()
                areas_ratio = mesh.determinants
                J_inv = mesh.inverses
                dxhat_dx = J_inv[:,0,0]
                dxhat_dy = J_inv[:,0,1]
                dyhat_dx = J_inv[:,1,0]
                dyhat_dy = J_inv[:,1,1]
            else: #If the coordinates are in 3D uses another formula
                areas_ratio = mesh.calcAreasRatio3D()
                
            N_dof = len(dof)
            N_int = len(mesh.triangles)*len(xhat)
            omega = np.zeros(N_int)

            #The sparse matrices will have for each line a integration point and each
            #column a dof. The number of integration points is the number of elements
            #times the number of points per element. So, for each point of integration
            #we have a certain amount of base functions to evaluate, the amount of 
            #dof per element.
            #Structure for sparse Matrix
            _i = np.zeros(N_int*len(phi[0]))
            _j = np.zeros(N_int*len(phi[0]))
            _v_u = np.zeros(N_int*len(phi[0]))
            _v_dxu = np.zeros(N_int*len(phi[0]))
            _v_dyu = np.zeros(N_int*len(phi[0]))

            for iloc in range(0,len(xhat)): #Points of Integration
                iglob = np.arange(iloc,N_int,len(xhat),dtype=np.int32)
                omega[iglob] = omegaloc[iloc] * areas_ratio
                for jloc in range(0,len(phi[0])): #Base functions
                    _idx = jloc*N_int + iloc*len(iglob) + np.arange(0,len(iglob)) #Global stacked index for the sparse matrix
                    jglob = numDof[:,jloc]
                    _i[_idx] = iglob
                    _j[_idx] = jglob
                    _v_u[_idx] = np.ones(len(iglob))*phi[iloc,jloc]

                    if not self.calcForMass3D: #The derivatives are not needed for the mass matrix
                        _v_dxu[_idx] = (dxphi[iloc,jloc]*dxhat_dx + dyphi[iloc,jloc]*dyhat_dx)
                        _v_dyu[_idx] = (dxphi[iloc,jloc]*dxhat_dy + dyphi[iloc,jloc]*dyhat_dy)
                    
            #Generate the matrices from the row/col indexes and values
            u = sparse.coo_matrix((_v_u, (_i, _j)),(N_int,N_dof))
            dxu = sparse.coo_matrix((_v_dxu, (_i, _j)),(N_int,N_dof))
            dyu = sparse.coo_matrix((_v_dyu, (_i, _j)),(N_int,N_dof))
            u = u.tocsc()
            dxu = dxu.tocsc()
            dyu = dyu.tocsc()

        return (np.array(dof),np.array(numDof),u,dxu,dyu,omega)

    def LagrangeFunctions(self, x, y, order):
        """Get the values and of the base functions and its derivatives at the triangle of reference.

        :param x: Array of x positions in the domain of reference
        :type x: array(float)
        :param y: Array of y positions in the domain of reference
        :type y: array(float)
        :param order: Order of the Lagrange function. [1 or 2]
        :type order: int
        :raises Exception: If the Lagrange function order are no implemented
        :return: (value,x derivative,y derivative)
        :rtype: (array(float),array(float),array(float))
        """
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
        else: 
            raise Exception('Lagrande functions of order ' + str(order) + ' not implemented.')
        return np.array(phi), np.array(dxphi), np.array(dyphi)
    
    def M_Matrix(self):
        """ Generate the mass matrix from the support matrices.

        :return: The mass matrix.
        :rtype: CSC Matrix
        """
        u,omega  = self.u,self.omega
        M = u.transpose().dot(u.transpose().multiply(omega).transpose())
        M = M.tocsc()

        #Garantee int64 to UMFPACK
        if 'scikits.umfpack' in sys.modules:
            M.indices = M.indices.astype(np.int64)
            M.indptr = M.indptr.astype(np.int64)
        return M

    def K_Matrix(self):
        """ Generate the stiffness matrix from the support matrices.

        :return: The stiffness matrix.
        :rtype: CSC Matrix
        """
        dxu,dyu,omega  = self.dxu,self.dyu,self.omega
        G1 = dxu.transpose().dot(dxu.transpose().multiply(omega).transpose())
        G2 = dyu.transpose().dot(dyu.transpose().multiply(omega).transpose())
        K = G1+G2
        K = K.tocsc()

        #Garantee int64 to UMFPACK
        if 'scikits.umfpack' in sys.modules:
            K.indices = K.indices.astype(np.int64)
            K.indptr = K.indptr.astype(np.int64)
        return K

    def F_Matrix(self, f):
        """Generate the forcing matrix.

        :param f: The source-term with argument time_index that returns an array with the values for each dof.
        :type f: function(int): Array
        :return: A function that returns the F matrix for each time index
        :rtype: function(int): Array
        """
        u,omega  = self.u,self.omega
        _ret = lambda time_index: u.multiply(f(time_index)).transpose().dot(omega) #TODO: Calculation can be faster
        return _ret