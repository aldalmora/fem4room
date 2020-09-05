import gmsh
import numpy as np
import matplotlib.pyplot as plt

class Mesh:
    @staticmethod
    def MeshSquare(name,Lx,Ly,h,order=1):
        """Create and return an instance of a square mesh.

        :param name: Name of the mesh.
        :type name: str
        :param Lx: x length of the square
        :type Lx: float
        :param Ly: x length of the square
        :type Ly: float
        :param h: element size
        :type h: float
        :param order: Order of the element [1 or 2], defaults to 1
        :type order: int, optional
        :return: The instance of the Mesh.
        :rtype: FEM_2D.Mesh
        """
        m = Mesh(name)
        elements = m.createSurface([[0,0,0],[Lx,0,0],[Lx,Ly,0],[0,Ly,0]],h=h)
        m.fac.addPlaneSurface([elements[-1]])
        m.generate(order=order)
        return m

    @staticmethod
    def MeshByTriangles(name,nodeTags,triangles,vertices):
        """Create and return a mesh instance given the triangles and its vertices. 
        The input must be ordered and numerated correctly.

        :param name: Name of the mesh
        :type name: str
        :param nodeTags: Identifier for each node. Used to map between different domains.
        :type nodeTags: Array
        :param triangles: The index of the vertices for each triangle.
        :type triangles: Array t x 3
        :param vertices: The coordinates of the vertices
        :type vertices: Array v x 3
        :return: The instance of the mesh.
        :rtype: FEM_2D.Mesh
        """
        m = Mesh(name)
        m.nodeTags = nodeTags
        m.vertices = vertices
        m.triangles = triangles
        return m

    def __init__(self, name):
        """Initialize GMSH and the mesh attributes

        :param name: Mesh name
        :type name: str
        """
        gmsh.initialize()
        gmsh.model.add(name)
        self.name = name
        self.model = gmsh.model
        self.fac = self.model.geo
        self.pos = gmsh.view
        self.vertices = np.array([])
        self.triangles = np.array([])

    def calcAreasRatio3D(self):
        """ Calculate the area of each triangle(element) for triangles in 3 dimensions. 
        Used for assembling the mass matrix for robin boundary conditions at surfaces.

        :return: The areas ratio between the triangle and the triangle of reference
        :rtype: Array(float)
        """
        K = self.vertices[self.triangles]

        #Structure of K:
        #x = Tk(x') = K(x'); for x' 1(0,0,1), 2(1,0,0), 3(0,1,0):

        #Only need the edges (first 3 vertices) to get the transformation
        K = K[:,0:3,:]
        K = np.moveaxis(K,2,1)
        
        #Area 
        X1 = K[:,:,1] - K[:,:,0]
        X2 = K[:,:,2] - K[:,:,0]
        x1=X1[:,0];y1=X1[:,1];z1=X1[:,2]
        x2=X2[:,0];y2=X2[:,1];z2=X2[:,2]
        A_paralelo = np.sqrt((y1*z2-y2*z1)**2 +  (x1*z2-x2*z1)**2 + (x1*y2 - x2*y1)**2) #TODO: Computational cost can be reduced?
        return 2 * A_paralelo/2

    def calcJacobians(self):
        """Pre-calculate the determinants and inverse of the jacobian for all triangles.
        Store the values at .determinants and .inverses
        """
        K = self.vertices[self.triangles]

        #Structure of K:
        #x = Tk(x') = K(x'); for x' 1(0,0,1), 2(1,0,0), 3(0,1,0):

        #Transformation Matrix
        xa = K[:,0,0];ya = K[:,0,1]
        xb = K[:,1,0];yb = K[:,1,1]
        xc = K[:,2,0];yc = K[:,2,1]
        a = (xb-xa)
        b = (xc-xa)
        c = (yb-ya)
        d = (yc-ya)
        #Inversion of 2x2 matrices
        self.determinants = a*d - b*c
        self.inverses = np.stack(1/self.determinants * [d,-b,-c,a],axis=1).reshape(-1,2,2)
        
    def createSurface(self, orderedVertices, h):
        """Uses the built-in GMSH mesh factory to construct a surface.

        :param orderedVertices: The coordinates of the surface vertices in order.
        :type orderedVertices: Array v x 3
        :param h: Elements size
        :type h: float
        :return: (p,l,cl) 
        p is the points tags
        l is the lines tags
        cl is the curve loops tag
        :rtype: (Array(int),Array(int),int)
        """
        p = []
        l = []
        for n in orderedVertices:
            p.append(self.fac.addPoint(n[0],n[1],n[2],h))

        for n in p:
            if (n < p[-1]):
                l.append(self.fac.addLine(n,n+1))
            else:
                l.append(self.fac.addLine(n,p[0]))

        cl = self.fac.addCurveLoop(l)

        return (p,l,cl)

    def createCircle(self, x, y, z, radius, h=0.03, addInterior = True):
        """Add a circle to the mesh. If addInterior = True, the interior of the circle will be meshed.

        :param x: The center coordinate x
        :type x: float
        :param y: The center coordinate y
        :type y: float
        :param z: The center coordinate z
        :type z: float
        :param radius: Circle radius
        :type radius: float
        :param h: Elements size, defaults to 0.03
        :type h: float, optional
        :param addInterior: If the elements will be meshed at the interior of the circle, defaults to True
        :type addInterior: bool, optional
        :return: (p,ca,cl,s)
        p are the points tags
        ca are the circle-arc tags
        cl is the curve loop tag
        s is the surface tag
        :rtype: (Array,Array,int,int)
        """
        p1 = self.fac.addPoint(x,y+radius,z,h)
        p2 = self.fac.addPoint(x,y,z,h)
        p3 = self.fac.addPoint(x,y-radius,z,h)
        ca1=self.fac.addCircleArc(p1,p2,p3)
        ca2=self.fac.addCircleArc(p3,p2,p1)

        cl = self.fac.addCurveLoop([ca1,ca2])

        s = self.fac.addPlaneSurface([cl]) if addInterior else 0

        return ([p1,p2,p3],[ca1,ca2],cl,s)

    def generate(self,order=1):
        """Generate the triangle elements of the mesh with GMSH

        :param order: Order of the elements [1 or 2], defaults to 1
        :type order: int, optional
        """
        self.fac.synchronize()
        self.model.mesh.generate(2)
        self.model.mesh.setOrder(order)
        self.model.mesh.removeDuplicateNodes()
        self.__fillVtxTri(order)
        
    def readMsh(self,file):
        """Read a mshFile with GMSH. The elements must be triangles of order 1.

        :param file: File path/name
        :type file: str
        """
        gmsh.open(file)
        self.fac.synchronize()
        self.__fillVtxTri(1)

    def __fillVtxTri(self,order):
        """Load the mesh infos(nodes and elements) from GMSH.

        :param order: Order of the elements [1 or 2]
        :type order: int
        """
        if order==1:
            tnodes = self.model.mesh.getNodesByElementType(2)
            unique, unique_indexes, unique_inverse = np.unique(tnodes[0],return_index=True,return_inverse=True)
            self.nodeTags = unique
            self.vertices = tnodes[1].reshape(-1,3)[unique_indexes]
            self.triangles = unique_inverse.reshape(-1,3)
        elif order==2:
            tnodes = self.model.mesh.getNodesByElementType(9)
            unique, unique_indexes, unique_inverse = np.unique(tnodes[0],return_index=True,return_inverse=True)
            self.nodeTags = unique
            self.vertices = tnodes[1].reshape(-1,3)[unique_indexes]
            self.triangles = unique_inverse.reshape(-1,6)

        self.vertice_group = np.zeros(len(self.vertices))
        for n in self.model.getPhysicalGroups():
            self.vertice_group[self.model.mesh.getNodesForPhysicalGroup(n[0],n[1])[0]-1] = n[1]

    def FLTKRun(self):
        """Opens the GMSH interface
        """
        gmsh.fltk.run()