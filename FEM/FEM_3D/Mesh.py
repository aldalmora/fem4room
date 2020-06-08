import gmsh
import numpy as np
import matplotlib.pyplot as plt

class Mesh:
    @staticmethod
    def MeshCube(name,Lx,Ly,Lz,h,order=1):
        m = Mesh('Mesh3D')
        cTag = m.createCube(Lx,Ly,Lz)
        m.fac.synchronize()
        dt_Boundary = m.model.getBoundary(cTag,recursive=True)
        m.model.occ.setMeshSize(dt_Boundary,h)
        m.fac.synchronize()
        m.generate(order=order)
        return m

    def __init__(self, name):
        gmsh.initialize()
        gmsh.model.add(name)
        self.name = name
        self.model = gmsh.model
        self.fac = self.model.occ
        self.pos = gmsh.view
        self.vertices = np.array([])
        self.tetra = np.array([])

    def calcJacobians(self):
        """Pre-calculate the jacobian of all tetrahedrons."""
        K = self.vertices[self.tetrahedrons]
        
        #Structure of K:
        #x = Tk(xhat) = Ax + (x1) ; for Xhat 1(0,0,0), 2(1,0,0), 3(0,1,0) and 4(0,0,1):
        #A = (x2 x3 x4) - (x1)
        K = np.moveaxis(K,2,1)
        K[:,:,1] = K[:,:,1] - K[:,:,0] 
        K[:,:,2] = K[:,:,2] - K[:,:,0]
        K[:,:,3] = K[:,:,3] - K[:,:,0]
        K = K[:,:,1:4]

        #Inverse of 3x3
        a = K[:,0,0];b = K[:,0,1];c = K[:,0,2]
        d = K[:,1,0];e = K[:,1,1];f = K[:,1,2]
        g = K[:,2,0];h = K[:,2,1];i = K[:,2,2]
        self.determinants = ((a*e*i) + (b*f*g) + (c*d*h) - (c*e*g) - (b*d*i) - (a*f*h))
        aux = [[e*i - f*h, c*h - b*i, b*f - c*e],
                [f*g - d*i, a*i - c*g, c*d - a*f],
                [d*h - e*g, b*g - a*h, a*e - b*d]]
        aux = np.array(aux)
        self.inverses = np.moveaxis(1/self.determinants * aux,2,0)

    def createCube(self, dx,dy,dz):
        bTag = self.fac.addBox(0,0,0,dx,dy,dz)
        return (3,bTag)

    def createSphere(self, x, y, z, radius):
        sTag = self.fac.addSphere(x,y,z,radius)
        return (3,sTag)

    def generate(self,order=1,mesh_algorithm=6):
        """Generate the mesh with GMSH"""
        self.model.mesh.generate(3)
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
            tnodes = self.model.mesh.getNodesByElementType(4)
            unique, unique_indexes, unique_inverse = np.unique(tnodes[0],return_index=True,return_inverse=True)
            self.nodeTags = unique
            self.vertices = tnodes[1].reshape(-1,3)[unique_indexes]
            self.tetrahedrons = unique_inverse.reshape(-1,4)
        elif order==2:
            tnodes = self.model.mesh.getNodesByElementType(11)
            unique, unique_indexes, unique_inverse = np.unique(tnodes[0],return_index=True,return_inverse=True)
            self.nodeTags = unique
            self.vertices = tnodes[1].reshape(-1,3)[unique_indexes]
            self.tetrahedrons = unique_inverse.reshape(-1,10)

        self.vertice_group = np.zeros(len(self.vertices))
        for n in self.model.getPhysicalGroups():
            self.vertice_group[self.model.mesh.getNodesForPhysicalGroup(n[0],n[1])[0]-1] = n[1]
        return

    def FLTKRun(self):
        """Opens the GMSH interface."""
        gmsh.fltk.run()