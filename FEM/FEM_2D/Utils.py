import numpy as np
import numpy.linalg as la

def calcL2(u,omega,result,exact):
    """For a giver result and the exact solution, returns the L2-norm of the error."""
    integrand = (exact-result)**2
    return np.sqrt(u.dot(integrand).dot(omega))

def calcH1(u,dxu,dyu,omega,result,exact,grad_exact):
    """For a giver result and the exact solution, returns the H1-norm of the error."""
    integral1 = u.dot((exact-result)**2).dot(omega)
    dif_grad = np.array(u.dot(grad_exact)) - np.array([dxu.dot(result),dyu.dot(result)]).transpose()
    integral2 = (la.norm(dif_grad,axis=1)*la.norm(dif_grad,axis=1)).dot(omega)
    return np.sqrt(integral1 + integral2)