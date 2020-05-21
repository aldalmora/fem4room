import numpy as np

class Boundary:
    def __init__(self, engine):
        self.engine = engine

    def Apply_Dirichlet(self, physical_group, K_full, M_full, f, g):
        engine = self.engine

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
        ddl_boundary_idx = np.argwhere(engine.mesh.vertice_group==physical_group)[:,0] #TODO Can be optimized considering that the interior idxs is given
        G_Boundary = np.array(g(engine.ddl[ddl_boundary_idx,0],engine.ddl[ddl_boundary_idx,1],engine.ddl[ddl_boundary_idx,2]))

        #Extra term that handles the Inhomogeneous Dirichlet B.C.
        A_Boundary = (K_full[ddl_interior_idx,:]+ M_full[ddl_interior_idx,:])[:,ddl_boundary_idx]
        G = A_Boundary.dot(G_Boundary)

        #Add the extra term to the RHS
        F = F - G

        return K,M,F,G_Boundary,ddl_interior_idx,ddl_boundary_idx