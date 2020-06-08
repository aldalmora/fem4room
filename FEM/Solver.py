import scipy.sparse.linalg as sla
from scipy.sparse.csgraph import reverse_cuthill_mckee
import scikits.umfpack

class umfLA(sla.LinearOperator):
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
        mi = umfLA(A)
        return sla.eigsh(A,k,M,sigma=sigma,which=which,OPinv=mi)

    def solve(self,f):
        A = self.engine.K_Matrix() + self.engine.M_Matrix()
        F = self.engine.F_Matrix(f)

        idx_rcm = reverse_cuthill_mckee(A)
        A = A[idx_rcm,:]
        A = A[:,idx_rcm]
        F = F[idx_rcm]
        
        return sla.spsolve(A,F)