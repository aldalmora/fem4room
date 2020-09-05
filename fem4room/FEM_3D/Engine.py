import numpy as np
import numpy.linalg as la
import scipy.sparse as sparse
import sys

from .Mesh import Mesh

class Engine:
    def __init__(self, mesh: Mesh, order: int, formule: int):
        """Initialize the engine and assembles the support matrices 

        :param mesh: Instance of the mesh.
        :type mesh: FEM_3D.Mesh
        :param order: Order of the elements. Only order 1 is available.
        :type order: int
        :param formule: Formule for the numerical integration. Only 1 is available.
        :type formule: int
        """
        self.order = order
        self.formule = formule
        self.mesh = mesh
        self.dof,self.numDof,self.u,self.dxu,self.dyu,self.dzu,self.omega = self.initializeMatrices('Lagrange')
        
    def IntegrationPointsHat(self, formule: int):
        """Return the integration points of the reference tetrahedra

        :param formule: Formule for the integration. Only 1 is available.
        :type formule: int
        :return: (x,y,z,omega) ; x, y and z being the coordinates at the reference domain. Omega the weight of integration.
        :rtype: Tuple of arrays
        """
        xhat = []
        yhat = []
        zhat = []
        omega = []
        if (formule==1):
            a = (5-np.sqrt(5))/20
            b = (5+3*np.sqrt(5))/20
            xhat = [a,a,a,b]
            yhat = [a,a,b,a]
            zhat = [a,b,a,a]
            omega = [1/24,1/24,1/24,1/24]
        elif (formule==2):
            xhat = [1/4,1/2,1/6,1/6,1/6]
            yhat = [1/4,1/6,1/6,1/6,1/2]
            zhat = [1/4,1/6,1/6,1/2,1/6]
            omega = [-.8/6,.45/6,.45/6,.45/6,.45/6]
        return (np.array(xhat), np.array(yhat), np.array(zhat), np.array(omega))

    def initializeMatrices(self, typeEF: str):
        """Generate the matrices with the values of the base functions, the dx, dy, dz and the weights for integration.

        :param typeEF: Type of inteporlation. Only 'Lagrange'.
        :type typeEF: str
        :return: (dof,numDof,u,dxu,dyu,dzu,omega);
        dof is the position of the degrees of freedom
        numDof is the dofs that composes each element
        u is the value of the interpolation function for each integration point
        dxu is the derivative in x
        dyu is the derivative in y
        dzu is the derivative in z
        omega is the quadrature weight for each integration point
        :rtype: (Array n x 3, Array e x 3, Matrix i x e, Matrix i x e, Matrix i x e, Array i); 
        n being the number of degrees of freedom; 
        e being the number of elements;
        i being the number of integration points
        """
        mesh,order,formule = self.mesh, self.order, self.formule
        xhat, yhat, zhat, omegaloc = self.IntegrationPointsHat(formule)

        if (typeEF=='Lagrange'):
            phi, dxphi, dyphi, dzphi = self.LagrangeFunctions(xhat, yhat, zhat, order)

            if (order==1 or order==2): #GMSH generate the nodes
                dof = mesh.vertices
                numDof = np.array(mesh.tetrahedrons,dtype=np.int32)

            mesh.calcJacobians()
            detJ = mesh.determinants
            J_inv = mesh.inverses
                
            dxhat_dx = J_inv[:,0,0]
            dxhat_dy = J_inv[:,0,1]
            dxhat_dz = J_inv[:,0,2]
            dyhat_dx = J_inv[:,1,0]
            dyhat_dy = J_inv[:,1,1]
            dyhat_dz = J_inv[:,1,2]
            dzhat_dx = J_inv[:,2,0]
            dzhat_dy = J_inv[:,2,1]
            dzhat_dz = J_inv[:,2,2]

            N_dof = len(dof)
            N_int = len(mesh.tetrahedrons)*len(xhat)
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
            _v_dzu = np.zeros(N_int*len(phi[0]))

            for iloc in range(0,len(xhat)): #Points of Integration
                iglob = np.arange(iloc,N_int,len(xhat),dtype=np.int64)
                omega[iglob] = omegaloc[iloc] * detJ
                for jloc in range(0,len(phi[0])): #Base functions
                    _idx = jloc*N_int + iloc*len(iglob) + np.arange(0,len(iglob)) #Global stacked index for the sparse matrix
                    jglob = numDof[:,jloc]
                    _i[_idx] = iglob
                    _j[_idx] = jglob
                    _v_u[_idx] = np.ones(len(iglob))*phi[iloc,jloc]
                    _v_dxu[_idx] = (dxphi[iloc,jloc]*dxhat_dx + dyphi[iloc,jloc]*dyhat_dx + dzphi[iloc,jloc]*dzhat_dx)
                    _v_dyu[_idx] = (dxphi[iloc,jloc]*dxhat_dy + dyphi[iloc,jloc]*dyhat_dy + dzphi[iloc,jloc]*dzhat_dy)
                    _v_dzu[_idx] = (dxphi[iloc,jloc]*dxhat_dz + dyphi[iloc,jloc]*dyhat_dz + dzphi[iloc,jloc]*dzhat_dz)
            
            
            #Generate the matrices from the row/col indexes and values
            u = sparse.coo_matrix((_v_u, (_i, _j)),(N_int,N_dof))
            dxu = sparse.coo_matrix((_v_dxu, (_i, _j)),(N_int,N_dof))
            dyu = sparse.coo_matrix((_v_dyu, (_i, _j)),(N_int,N_dof))
            dzu = sparse.coo_matrix((_v_dzu, (_i, _j)),(N_int,N_dof))
            u = u.tocsc() 
            dxu = dxu.tocsc()
            dyu = dyu.tocsc()
            dzu = dzu.tocsc()

        return np.array(dof),np.array(numDof),u,dxu,dyu,dzu,omega

    def LagrangeFunctions(self, x, y, z, order: int):
        """ Get the values and of the base functions and its derivatives at the tetrahedra of reference.

        :param x: Array of x positions in the domain of reference
        :type x: array(float)
        :param y: Array of y positions in the domain of reference
        :type y: array(float)
        :param z: Array of z positions in the domain of reference
        :type z: array(float)
        :param order: Order of the Lagrange function. Only 1 is available.
        :type order: int
        :raises Exception: If the Lagrange function order are no implemented
        :return: (value,x derivative,y derivative,z derivative)
        :rtype: (array(float),array(float),array(float),array(float))
        """
        phi = []
        dxphi = []
        dyphi = []
        dzphi = []
        if (order==1): #Ordering (0,0,0), (1,0,0), (0,1,0) et (0,0,1)
            for i in range(0,len(x)):
                phi.append([1-x[i]-y[i]-z[i],x[i],y[i],z[i]])
                dxphi.append([-1.,1.,0.,0])
                dyphi.append([-1.,0.,1.,0])
                dzphi.append([-1.,0.,0.,1])
        else: 
            raise Exception('Lagrande functions of order ' + str(order) + ' not implemented.')
        return np.array(phi), np.array(dxphi), np.array(dyphi), np.array(dzphi)
    
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
        dxu,dyu,dzu,omega  = self.dxu,self.dyu,self.dzu,self.omega
        G1 = dxu.transpose().dot(dxu.transpose().multiply(omega).transpose())
        G2 = dyu.transpose().dot(dyu.transpose().multiply(omega).transpose())
        G3 = dzu.transpose().dot(dzu.transpose().multiply(omega).transpose())
        K = G1+G2+G3
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
        _ret = lambda time_index: u.multiply(f(time_index)).transpose().dot(omega)
        return _ret

    def getValueByPosition(self, dofValues, position):
        """Given the values of the degrees of freedom, it interpolates the lagrange functions to give the estimated value. Only works for Lagrange P1 elements.

        :param dofValues: The values at all degrees of freedom
        :type dofValues: Array
        :param position: Position to be evaluated using intepolation
        :type position: Array(3)
        :return: The interpolated value at the position.
        :rtype: float
        """
        etags = []
        coords_u = []
        coords_v = []
        coords_w = []
        for pos in position:
            x = pos[0]
            y = pos[1]
            z = pos[2]
            etag,etype,nodeTags,u,v,w = self.mesh.model.mesh.getElementByCoordinates(x,y,z,3)
            etags.append(etag)
            coords_u.append(u)
            coords_v.append(v)
            coords_w.append(w)
        tetraIndex = [np.where(self.mesh.tetraTags==etag)[0][0] for etag in etags]
        nodeIndex = self.mesh.tetrahedrons[tetraIndex]
        lFunction = self.LagrangeFunctions(coords_u,coords_v,coords_w,1)[0][0]
        return np.dot(lFunction, np.transpose(dofValues[:,nodeIndex],(0,2,1)))

    