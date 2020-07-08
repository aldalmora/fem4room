import numpy as np
import gmsh
from . import FEM_2D as fem2d

class Boundary:
    @staticmethod
    def Apply_Dirichlet(engine, physical_group, K_full, M_full, f, g):
        #Get DDLs which are not in the Dirichlet B.
        ddl_interior_idx = np.argwhere(engine.mesh.vertice_group!=physical_group)[:,0]

        #Remove DDLs from boundary
        K = K_full[ddl_interior_idx,:]
        K = K[:,ddl_interior_idx]
        M = M_full[ddl_interior_idx,:]
        M = M[:,ddl_interior_idx]

        #Forcing function
        F = engine.F_Matrix(f)
        F = F[ddl_interior_idx]

        #Get only the DDls from the boundary and load the value of the function
        ddl_boundary_idx = np.argwhere(engine.mesh.vertice_group==physical_group)[:,0]
        G_Boundary = np.array(g(engine.ddl[ddl_boundary_idx,0],engine.ddl[ddl_boundary_idx,1],engine.ddl[ddl_boundary_idx,2]))

        #Extra term that handles the Inhomogeneous Dirichlet B.C.
        A_Boundary = (K_full[ddl_interior_idx,:]+ M_full[ddl_interior_idx,:])[:,ddl_boundary_idx]
        G = A_Boundary.dot(G_Boundary)

        #Add the extra term to the RHS
        F = F - G

        return K,M,F,G_Boundary,ddl_interior_idx,ddl_boundary_idx

    @staticmethod
    def Impedance_Damping_Matrix(m3d, physical_group):
        """ Calculates the damping matrix for surface impedance boundary condition, given the physical group(GMSH) of the surface. Only 3D """
        triangles,vertices = Boundary.getPhysicalGroupTriangles(m3d,physical_group)

        unique, unique_indexes, unique_inverse = np.unique(triangles,return_index=True,return_inverse=True)
        nodeTags = unique
        vertices = vertices[unique_indexes]
        triangles = unique_inverse.reshape(-1,3)

        m2d = fem2d.Mesh.MeshByTriangles('ImpedanceSurface',nodeTags,triangles,vertices)
        engine2d = fem2d.Engine(m2d,1,1,calcForMass3D=True)
        M_2d = engine2d.M_Matrix().tocoo()

        gmsh.model.remove()
        return M_2d,nodeTags
        
    @staticmethod
    def getPhysicalGroupTriangles(m, physical_group):
        """ Return the triangles that belongs to the surface of given physical group(GMSH). """
        nodes = m.model.mesh.getNodes()
        arg_sort = np.argsort(nodes[0])
        nodes_coords = nodes[1].reshape(-1,3)[arg_sort]

        surface_tags = m.model.getEntitiesForPhysicalGroup(2,physical_group)
        triangles=[]
        vertices=[]
        for st in surface_tags:
            elements = m.model.mesh.getElements(2,st)
            triangles.extend(elements[2][0]-1)
            vertices.extend(nodes_coords[elements[2][0]-1])

        return np.array(triangles),np.array(vertices)

