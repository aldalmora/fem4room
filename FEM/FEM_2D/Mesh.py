import gmsh
import numpy as np
import matplotlib.pyplot as plt

class Mesh:
    @staticmethod
    def MeshSquare(name,Lx,Ly,h,order=1):
        """ Create and return an instance of a square mesh. """
        m = Mesh(name)
        elements = m.createSurface([[0,0,0],[Lx,0,0],[Lx,Ly,0],[0,Ly,0]],h=h)
        m.fac.addPlaneSurface([elements[-1]])
        m.generate(order=order)
        return m

    @staticmethod
    def MeshByTriangles(name,nodeTags,triangles,vertices):
        """ Create and return a mesh instance given the triangles and its vertices. The input must be ordered and numerated correctly. """
        m = Mesh(name)
        m.nodeTags = nodeTags
        m.vertices = vertices
        m.triangles = triangles
        return m

    def __init__(self, name):
        gmsh.initialize()
        gmsh.model.add(name)
        self.name = name
        self.model = gmsh.model
        self.fac = self.model.geo
        self.pos = gmsh.view
        self.vertices = np.array([])
        self.triangles = np.array([])

    def calcAreas3D(self):
        """ Calculate the area of each triangle(element) for triangles in 3 dimensions. Used for assembling the mass matrix for robin boundary conditions at surfaces. """
        K = self.vertices[self.triangles]
        self.area_triangle_ref = .5

        #Structure of K:
        #x = Tk(xhat) = K(xhat); for Xhat 1(0,0,1), 2(1,0,0), 3(0,1,0):

        #Only need the edges to get the transformation
        K = K[:,0:3,:]
        K = np.moveaxis(K,2,1)
        
        #Area 
        X1 = K[:,:,1] - K[:,:,0]
        X2 = K[:,:,2] - K[:,:,0]
        x1=X1[:,0];y1=X1[:,1];z1=X1[:,2]
        x2=X2[:,0];y2=X2[:,1];z2=X2[:,2]
        A_paralelo = np.sqrt((y1*z2-y2*z1)**2 +  (x1*z2-x2*z1)**2 + (x1*y2 - x2*y1)**2) #TODO: Costly?
        self.areas=A_paralelo/2

    def calcJacobians(self):
        """Pre-calculate the jacobian of all triangles."""
        K = self.vertices[self.triangles]
        self.area_triangle_ref = .5

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
        """Uses the built-in GMSH mesh factory to construct a surface."""
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
        """ Add a circle to the mesh. If addInterior = True, the interior of the circle will be meshed. """
        p1 = self.fac.addPoint(x,y+radius,z,h)
        p2 = self.fac.addPoint(x,y,z,h)
        p3 = self.fac.addPoint(x,y-radius,z,h)
        ca1=self.fac.addCircleArc(p1,p2,p3)
        ca2=self.fac.addCircleArc(p3,p2,p1)

        cl = self.fac.addCurveLoop([ca1,ca2])

        s = self.fac.addPlaneSurface([cl]) if addInterior else 0

        return ([p1,p2,p3],[ca1,ca2],cl,s)

    def generate(self,order=1,mesh_algorithm=6):
        """Generate the mesh with GMSH"""
        self.fac.synchronize()
        self.model.mesh.generate(2)
        self.model.mesh.setOrder(order)
        gmsh.option.setNumber('Mesh.Algorithm',mesh_algorithm)
        self.__fillVtxTri(order)
        
    def readMsh(self,file):
        """Read a mshFile with GMSH"""
        gmsh.open(file)
        self.fac.synchronize()
        self.__fillVtxTri(1)

    def __fillVtxTri(self,order):
        """Load the mesh infos(nodes and elements) in GMSH"""
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

    def plotMesh(self, opt):
        """Show the mesh with matplotlib.
        If opt has 'v' it numbers the vertices
        If opt has 't' it numbers the triangles
        If opt has 'p' it numbers the groups/domains
        """
        plt.triplot(self.vertices[:,0],self.vertices[:,1],self.triangles[:,0:3])
        plt.gca().set_aspect('equal')
        if opt.find('v') != -1:
            for n in range(0,len(self.vertices)):
                plt.text(self.vertices[n,0],self.vertices[n,1],n)
        if opt.find('t') != -1:
            for n in range(0,len(self.triangles)):
                x = [self.vertices[self.triangles[n,0],0],self.vertices[self.triangles[n,1],0],self.vertices[self.triangles[n,2],0]]
                y = [self.vertices[self.triangles[n,0],1],self.vertices[self.triangles[n,1],1],self.vertices[self.triangles[n,2],1]]
                coord = np.mean([x,y],axis=1)
                plt.text(coord[0]-0.03,coord[1],n)
        if opt.find('p') != -1:
            for n in range(0,len(self.vertices)):
                if (self.vertice_group[n] != 0):
                    plt.text(self.vertices[n,0],self.vertices[n,1],int(self.vertice_group[n]))

    def FLTKRun(self):
        """Opens the GMSH interface."""
        gmsh.fltk.run()