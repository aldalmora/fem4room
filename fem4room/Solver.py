import scipy.sparse.linalg as sla
from scipy.sparse.csgraph import reverse_cuthill_mckee
import scikits.umfpack

class umfLA(sla.LinearOperator):
    """ Linear operator which uses UMFPack to solve a linear system. """
    def __init__(self, M):
        self.solve = sla.factorized(M)
        self.shape = M.shape

    def _matvec(self, x):
        return self.solve(x)

class Solver():
    def __init__(self,engine):
        self.engine = engine
        sla.use_solver(useUmfpack=True)

    def eig(self,A,M,k,sigma,which):
        """ Solve the eigenproblem A*x = a*M*x around sigma returning k eigenvalues/eigenvectors. """
        mi = umfLA(A)
        return sla.eigsh(A,k,M,sigma=sigma,which=which,OPinv=mi)

    def solve(self,f):
        """ Solves (K+M)x = F """
        A = self.engine.K_Matrix() + self.engine.M_Matrix()
        F = self.engine.F_Matrix(f)
        
        return sla.spsolve(A,F)

    def solve(self,A,F):
        """ Solves (A)x = F """
        
        return sla.spsolve(A,F)