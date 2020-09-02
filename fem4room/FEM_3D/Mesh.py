import gmsh
import ezdxf
import numpy as np
import matplotlib.pyplot as plt

class Mesh:
    @staticmethod
    def MeshCube(name,Lx,Ly,Lz,h,order=1):
        """ Create and return an instance of a cubic mesh. """
        m = Mesh('Mesh3D')
        m.main_dTag = m.createCube(Lx,Ly,Lz)
        m.fac.synchronize()
        dt_Boundary = m.model.getBoundary(m.main_dTag,recursive=True)
        m.model.occ.setMeshSize(dt_Boundary,h)
        m.fac.synchronize()
        m.generate(order=order)
        return m

    def __init__(self, name):
        gmsh.initialize()
        gmsh.clear()
        gmsh.model.add(name)
        self.name = name
        self.model = gmsh.model
        self.fac = self.model.occ
        self.pos = gmsh.view
        self.vertices = np.array([])
        self.tetra = np.array([])

    def calcJacobians(self):
        """Pre-calculate the determinants and inverse of the jacobian for all tetrahedrons."""
        K = self.vertices[self.tetrahedrons]
        
        #Structure of K:
        #x = Tk(xhat) = A(xhat) + (x1) ; for Xhat 1(0,0,0), 2(1,0,0), 3(0,1,0) and 4(0,0,1):

        #A = (x2 x3 x4) - (x1)
        K = np.moveaxis(K,2,1)
        K[:,:,1] = K[:,:,1] - K[:,:,0]
        K[:,:,2] = K[:,:,2] - K[:,:,0]
        K[:,:,3] = K[:,:,3] - K[:,:,0]
        K = K[:,:,1:4]
        self.transform_matrix = K

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
        """ Add a cube to the mesh. Return the dimTag."""
        bTag = self.fac.addBox(0,0,0,dx,dy,dz)
        self.main_dTag = (3,bTag)
        return self.main_dTag

    def createSphere(self, x, y, z, radius):
        """ Add a sphere to the mesh. Return the dimTag."""
        sTag = self.fac.addSphere(x,y,z,radius)
        self.main_dTag = (3,sTag)
        return self.main_dTag

    def addNamedPoint(self, x, y, z, name):
        """ Add a point to the mesh and a physical group to it. Return the point dimTag."""
        pTag = self.fac.addPoint(x,y,z)
        self.fac.synchronize()
        pg_point = self.model.addPhysicalGroup(0,[pTag])
        self.model.setPhysicalName(0,pg_point,name)
        return (0,pTag)

    def generate(self,order=1):
        """Generate the mesh with GMSH"""
        self.model.mesh.generate(3)
        self.model.mesh.setOrder(order)
        self.model.mesh.removeDuplicateNodes()
        self.__fillVtxTri(order)

    def readGeo(self,file):
        """ Open a .GEO file for geometry descriptions """
        gmsh.open(file)

    def readMsh(self,file):
        """Read a mshFile with GMSH. The elements must be tetrahedrons of order 1."""
        gmsh.open(file)
        self.fac.synchronize()
        self.__fillVtxTri(1)

#region DXF Reading
    def __DXFReadPoint(self,e):
        # print('Point: ', e.dxf.layer)
        point_id = self.fac.addPoint(e.dxf.location[0],e.dxf.location[1],e.dxf.location[2])
        self.dxf_point_reference = np.array([e.dxf.location[0],e.dxf.location[1],e.dxf.location[2]])
        
        self.dxf_points_ids.append(point_id)
        self.dxf_points_names.append(e.dxf.layer)

    def __DXFReadPolyMesh(self,e):
        # print('Physical Group: ', e.dxf.layer)
        for f in e.faces():
            for v in f[0:3]: #Only for three node polymeshs
                coord1=np.round(v.dxf.location[0],9)
                coord2=np.round(v.dxf.location[1],9)
                coord3=np.round(v.dxf.location[2],9)
                self.dxf_vertices.append([coord1,coord2,coord3])

            #Calculate the cross product of the triangle vectors
            p1 = np.array(self.dxf_vertices[-3])
            p2 = np.array(self.dxf_vertices[-2])
            p3 = np.array(self.dxf_vertices[-1])
            v1 = p2 - p1 
            v2 = p3 - p1
            crs =  np.dot(np.cross(v1,v2),self.dxf_point_reference)

            if crs!=0:
                self.dxf_faces_names.append(e.dxf.layer)
                # self.dxf_faces.append([len(self.dxf_vertices)-3,len(self.dxf_vertices)-2,len(self.dxf_vertices)-1])
                if crs>0:
                    self.dxf_faces.append([len(self.dxf_vertices)-3,len(self.dxf_vertices)-2,len(self.dxf_vertices)-1])
                elif crs<0:
                    self.dxf_faces.append([len(self.dxf_vertices)-1,len(self.dxf_vertices)-2,len(self.dxf_vertices)-3])
                
        
    def readDXF(self,file):
        self.dxf_points_ids = []
        self.dxf_points_names = []
        self.dxf_faces = []
        self.dxf_faces_names = []
        self.dxf_vertices = []

        doc = ezdxf.readfile(file)
        msp = doc.modelspace()
        for e in msp.query('POINT'): #First read the points
            if e.dxf.layer!='Layer0': #Ignore not named SketchUp points
                self.__DXFReadPoint(e)

        for e in msp.query('POLYLINE'): #Then the polymeshs
            if (e.is_poly_face_mesh):
                self.__DXFReadPolyMesh(e)

        self.dxf_vertices = np.array(self.dxf_vertices)
        self.dxf_faces = np.array(self.dxf_faces)

        #Remove duplicated points
        new_faces = np.zeros(self.dxf_faces.shape,dtype=int)
        self.dxf_vertices,inverse = np.unique(self.dxf_vertices, axis=0, return_inverse=True)
        for i in range(0,len(inverse)):
            new_faces[self.dxf_faces==i] = inverse[i]
        self.dxf_faces = new_faces

        #Add read points to the mesh. These points are used to form the surfaces
        points_surfaces_ids = []
        for v in self.dxf_vertices:
            points_surfaces_ids.append(self.fac.addPoint(v[0],v[1],v[2]))
        self.dxf_faces,idx = np.unique(self.dxf_faces,axis=0,return_index=True)
        self.dxf_faces_names = np.array(self.dxf_faces_names)[idx]

        #Stores the links between points to not recreate them when linking points to make a surface
        curves = np.zeros((len(points_surfaces_ids),len(points_surfaces_ids)))

        surface_ids = []
        for i in range(0,len(self.dxf_faces)):
            face_points_idx=self.dxf_faces[i]
            lines = []
        
            #Check if the lines of the surface already exists or create new ones
            #TODO: This part can can be simplified
            if curves[face_points_idx[0],face_points_idx[1]] != 0:
                lines.append(curves[face_points_idx[0],face_points_idx[1]])
            else:
                lId = self.fac.addLine(points_surfaces_ids[face_points_idx[0]],points_surfaces_ids[face_points_idx[1]])
                curves[face_points_idx[0],face_points_idx[1]] = lId
                curves[face_points_idx[1],face_points_idx[0]] = lId
                lines.append(lId)

            if curves[face_points_idx[1],face_points_idx[2]] != 0:
                lines.append(curves[face_points_idx[1],face_points_idx[2]])
            else:
                lId = self.fac.addLine(points_surfaces_ids[face_points_idx[1]],points_surfaces_ids[face_points_idx[2]])
                curves[face_points_idx[1],face_points_idx[2]] = lId
                curves[face_points_idx[2],face_points_idx[1]] = lId
                lines.append(lId)

            if curves[face_points_idx[0],face_points_idx[2]] != 0:
                lines.append(curves[face_points_idx[2],face_points_idx[0]])
            else:
                lId = self.fac.addLine(points_surfaces_ids[face_points_idx[0]],points_surfaces_ids[face_points_idx[2]])
                curves[face_points_idx[0],face_points_idx[2]] = lId
                curves[face_points_idx[2],face_points_idx[0]] = lId
                lines.append(lId)

            cl = self.fac.addCurveLoop(lines)
            s = self.fac.addPlaneSurface([cl])
            surface_ids.append(s)

        self.fac.synchronize()
        sl = self.fac.addSurfaceLoop(surface_ids)
        vl = self.fac.addVolume([sl])
        self.fac.healShapes()
        self.fac.synchronize()
        #Embed the defined points in DXF to the mesh. This enables the source and receiving points to be degrees of freedom in the finite elements formulation.
        self.model.mesh.embed(0,self.dxf_points_ids,3,vl)

        #Name the points and surfaces
        surface_ids = np.array(surface_ids)
        self.dxf_points_names = np.array(self.dxf_points_names)
        self.dxf_points_ids = np.array(self.dxf_points_ids)
        for n in np.unique(self.dxf_faces_names):
            idx = np.where(self.dxf_faces_names==n)
            pg = self.model.addPhysicalGroup(2,surface_ids[idx])
            self.model.setPhysicalName(2,pg,n)
        for n in np.unique(self.dxf_points_names):
            idx = np.where(self.dxf_points_names==n)
            pg = self.model.addPhysicalGroup(0,self.dxf_points_ids[idx])
            self.model.setPhysicalName(0,pg,n)
#endregion

    def __fillVtxTri(self,order):
        """Load the mesh infos(nodes and elements) in GMSH"""
        if order==1:
            tnodes = self.model.mesh.getNodesByElementType(4)
            unique, unique_indexes, unique_inverse = np.unique(tnodes[0],return_index=True,return_inverse=True)
            self.nodeTags = unique
            self.tetraTags = self.model.mesh.getElementsByType(4)[0]
            self.vertices = tnodes[1].reshape(-1,3)[unique_indexes]
            self.tetrahedrons = unique_inverse.reshape(-1,4)
        elif order==2:
            tnodes = self.model.mesh.getNodesByElementType(11)
            unique, unique_indexes, unique_inverse = np.unique(tnodes[0],return_index=True,return_inverse=True)
            self.nodeTags = unique
            self.tetraTags = self.model.mesh.getElementsByType(11)[0]
            self.vertices = tnodes[1].reshape(-1,3)[unique_indexes]
            self.tetrahedrons = unique_inverse.reshape(-1,10)

        self.vertice_group = np.zeros(len(self.vertices))

        #Points
        for n in self.model.getPhysicalGroups(0):
            nodeTags = self.model.mesh.getNodesForPhysicalGroup(n[0],n[1])[0]
            idx_nodes = np.where(nodeTags[:,None]==self.nodeTags)[1]
            self.vertice_group[idx_nodes] = n[1]

        #Surfaces
        for n in self.model.getPhysicalGroups(2):
            nodeTags = self.model.mesh.getNodesForPhysicalGroup(n[0],n[1])[0]
            idx_nodes = np.where(nodeTags[:,None]==self.nodeTags)[1]
            self.vertice_group[idx_nodes] = n[1]
        return

    def FLTKRun(self):
        """Opens the GMSH interface."""
        gmsh.fltk.run()