import scipy.sparse.linalg as sla
from scipy.sparse.csgraph import reverse_cuthill_mckee
import sys


class umfLA(sla.LinearOperator):
    """Linear operator which can use UMFPack to solve a linear system.   """
    def __init__(self, M):
        """Initialize the linear operator

        :param M: The matrix to be factorized/solved.
        :type M: Sparse Matrix
        """
        self.solve = sla.factorized(M)
        self.shape = M.shape

    def _matvec(self, x):
        return self.solve(x)

class Solver():
    def __init__(self,engine):
        """Initialize the solver. Load umfpack is available

        :param engine: FEM Engine
        :type engine: FEM_2D.Engine / FEM_3D.Engine
        """
        self.engine = engine
        if 'scikits.umfpack' in sys.modules:
            sla.use_solver(useUmfpack=True)

    def eig(self,A,M,k,sigma,which):
        """Solve the eigenproblem A*x = a*M*x around sigma returning k eigenvalues/eigenvectors.

        :param A: Matrix A
        :type A: Sparse Matrix
        :param M: [description]
        :type M: Sparse Matrix
        :param k: Amount of eigenvalues to be calculated
        :type k: int
        :param sigma: Number around which the values will be searched
        :type sigma: float
        :param which: Type of solving. See scipy.sparse.eigsh
        :type which: string
        :return: k Eigenvalues, eigenvectors
        :rtype: Array, Matrix
        """
        mi = umfLA(A)
        return sla.eigsh(A,k,M,sigma=sigma,which=which,OPinv=mi)

    def solve(self,f):
        """Solves (K+M)x = F

        :param f: Forcing values for each time-step and degree of freedom
        :type f: function(int): Array
        :return: Solution x
        :rtype: Array
        """
        A = self.engine.K_Matrix() + self.engine.M_Matrix()
        F = self.engine.F_Matrix(f)
        
        return sla.spsolve(A,F)

    def solve(self,A,F):
        """Solves (A)x = F

        :param A: Matrix A
        :type A: Sparse matrix
        :param F: Matrix F
        :type F: Sparse matrix
        :return: Solution x
        :rtype: Array
        """
        
        return sla.spsolve(A,F)