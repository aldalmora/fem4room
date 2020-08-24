import numpy as np
import gmsh
from . import FEM_2D as fem2d
from . import FEM_3D as fem3d

class Boundary:
    @staticmethod
    def Apply_Dirichlet(engine, physical_group, M_full, C_full, K_full, f=None, g=None): #TODO: what do you expect for g(time and dof or only time)?
        """ Apply the dirichlet boundary condition given the matrices, the physical group and f,g being functions in time index returning vectors with the values at the ddls. """
        
        #Get DDLs which are not in the Dirichlet B.
        ddl_interior_idx = np.argwhere(engine.mesh.vertice_group!=physical_group)[:,0]

        #Remove DDLs from boundary
        M = M_full[ddl_interior_idx,:]
        M = M[:,ddl_interior_idx]
        C = C_full[ddl_interior_idx,:]
        C = C[:,ddl_interior_idx]
        K = K_full[ddl_interior_idx,:]
        K = K[:,ddl_interior_idx]

        #Forcing function
        f = engine.F_Matrix(f)
        F = lambda time_index: f(time_index)[ddl_interior_idx]

        #Get only the DDls from the boundary and load the value of the function
        ddl_boundary_idx = np.argwhere(engine.mesh.vertice_group==physical_group)[:,0]
        G_Boundary = lambda time_index: g(time_index)[ddl_boundary_idx]

        #Extra term that handles the Inhomogeneous Dirichlet B.C.
        A_Boundary = (K_full[ddl_interior_idx,:]+ M_full[ddl_interior_idx,:])[:,ddl_boundary_idx]
        G = lambda time_index: A_Boundary.dot(G_Boundary(time_index))

        #Add the extra term to the RHS
        f_ret = lambda time_index: F(time_index) - G(time_index)

        return M,C,K,f_ret,G_Boundary,ddl_interior_idx,ddl_boundary_idx

    @staticmethod
    def Impedance_Damping_Matrix(m3d, physical_group):
        """ Calculates the damping matrix for surface impedance boundary condition, given the physical group(GMSH) of the surface. Only 3D (Lagrange P1) """
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
        nodes = m.nodeTags
        nodes_coords = m.vertices

        surface_tags = m.model.getEntitiesForPhysicalGroup(2,physical_group)
        triangles=[]
        vertices=[]
        for st in surface_tags:
            elements = m.model.mesh.getElements(2,st)
            _triangles = elements[2][0]
            _vertices = nodes_coords[np.where(_triangles[:,None]==nodes)[1]]
            vertices.extend(_vertices)
            triangles.extend(_triangles)

        return np.array(triangles),np.array(vertices)

