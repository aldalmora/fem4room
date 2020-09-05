import numpy as np
import gmsh
from . import FEM_2D as fem2d
from . import FEM_3D as fem3d

class Boundary:
    @staticmethod
    def Apply_Dirichlet(engine, physical_group_id, M_full, C_full, K_full, f=None, g=None):
        """Apply the dirichlet boundary condition given the matrices, 
        the physical group and f,g being functions in time index returning vectors with the values at the dofs.

        :param engine: The fem engine.
        :type engine: FEM_2D.Engine, FEM_3D.Engine
        :param physical_group_id: Physical group ID of the surface (GMSH)
        :type physical_group_id: int
        :param M_full: The mass matrix.
        :type M_full: CSC Matrix
        :param C_full: The "viscous damping" matrix.
        :type C_full: CSC Matrix
        :param K_full: The stiffness matrix.
        :type K_full: CSC Matrix
        :param f: Function of the forcing function, defaults to None
        :type f: function(time_index): Array, optional
        :param g: Function of the inhomogeneous dirichlet function, defaults to None
        :type g: function(time_index): Array, optional
        :return: M,C,K,f_ret,G_Boundary,dof_interior_idx,dof_boundary_idx
        :rtype: (CSC, CSC, CSC, function(time_index): Array, function(time_index): Array, Array, Array)
        """
        
        #Get DOFs which are not in the Dirichlet B.
        dof_interior_idx = np.argwhere(engine.mesh.vertice_group!=physical_group_id)[:,0]

        #Remove DOFs from boundary
        M = M_full[dof_interior_idx,:]
        M = M[:,dof_interior_idx]
        C = C_full[dof_interior_idx,:]
        C = C[:,dof_interior_idx]
        K = K_full[dof_interior_idx,:]
        K = K[:,dof_interior_idx]

        #Forcing function
        f = engine.F_Matrix(f)
        F = lambda time_index: f(time_index)[dof_interior_idx]

        #Get only the DOFs from the boundary and load the value of the function
        dof_boundary_idx = np.argwhere(engine.mesh.vertice_group==physical_group_id)[:,0]
        G_Boundary = lambda time_index: g(time_index)[dof_boundary_idx]

        #Extra term that handles the Inhomogeneous Dirichlet B.C.
        A_Boundary = (K_full[dof_interior_idx,:]+ M_full[dof_interior_idx,:])[:,dof_boundary_idx]
        G = lambda time_index: A_Boundary.dot(G_Boundary(time_index))

        #Add the extra term to the RHS
        F_ret = lambda time_index: F(time_index) - G(time_index)

        return M,C,K,F_ret,G_Boundary,dof_interior_idx,dof_boundary_idx

    @staticmethod
    def Surface_Mass_Matrix(m3d, physical_group_id):
        """Calculates the 2D mass matrix(surface integral) for a 3D surface,
        given the physical group(GMSH) of the surface. Only 3D (Lagrange P1)

        :param m3d: Mesh instance in 3D domain.
        :type m3d: FEM_3D.Mesh
        :param physical_group_id: Physical group id(GMSH) of the surface.
        :type physical_group_id: int
        :return: The mesh instance of the surface in 2D, the nodetags of the 3D domain for each 2D dof. 
        :rtype: FEM_2D.Mesh, Array
        """
        triangles,vertices = Boundary.getPhysicalGroupTriangles(m3d,physical_group_id)

        unique, unique_indexes, unique_inverse = np.unique(triangles,return_index=True,return_inverse=True)
        nodeTags = unique
        vertices = vertices[unique_indexes]
        triangles = unique_inverse.reshape(-1,3)

        m2d = fem2d.Mesh.MeshByTriangles('Surface',nodeTags,triangles,vertices)
        engine2d = fem2d.Engine(m2d,1,1,calcForMass3D=True)
        M_2d = engine2d.M_Matrix().tocoo()

        gmsh.model.remove()
        return M_2d,nodeTags
        
    @staticmethod
    def getPhysicalGroupTriangles(m, physical_group_id):
        """Return the triangles that belongs to the surface of given physical group(GMSH).

        :param m: The mesh instance.
        :type m: FEM_2D.Mesh, FEM_3D.Mesh
        :param physical_group_id: Physical group id(GMSH)
        :type physical_group_id: int
        :return: Array with the vertices(indexes), Array of vertice coordinates.
        :rtype: Array, Array
        """
        nodes = m.nodeTags
        nodes_coords = m.vertices

        surface_tags = m.model.getEntitiesForPhysicalGroup(2,physical_group_id)
        triangles=[]
        vertices=[]
        for st in surface_tags:
            elements = m.model.mesh.getElements(2,st)
            _triangles = elements[2][0]
            _vertices = nodes_coords[np.where(_triangles[:,None]==nodes)[1]]
            vertices.extend(_vertices)
            triangles.extend(_triangles)

        return np.array(triangles),np.array(vertices)

