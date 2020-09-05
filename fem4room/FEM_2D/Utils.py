import numpy as np
import numpy.linalg as la

def calcL2(u,omega,result,exact):
    """For a given result and the exact solution, returns the L2-norm of the error.

    :param u: The support matrix u, with the values of the interpolating functions at the integration points.
    :type u: CSC Matrix
    :param omega: Weights of the quadrature formula at the integration points
    :type omega: Array
    :param result: Result given from the solver. Value at each degree of freedom.
    :type result: Array
    :param exact: Exact solution at each degree of freedom
    :type exact: Array
    :return: The L2-norm of the error.
    :rtype: float
    """
    integrand = (exact-result)**2
    return np.sqrt(u.dot(integrand).dot(omega))

def calcH1(u,dxu,dyu,omega,result,exact,grad_exact):
    """For a given result and the exact solution, returns the H1-norm of the error.

    :param u: The support matrix u, with the values of the interpolating functions at the integration points.
    :type u: CSC Matrix
    :param dxu: The support matrix dxu, with the x derivative values
    :type dxu: CSC Matrix
    :param dyu: The support matrix dyu, with the x derivative values
    :type dyu: CSC Matrix
    :param omega: Weights of the quadrature formula at the integration points
    :type omega: Array
    :param result: Result given from the solver. Value at each degree of freedom.
    :type result: Array
    :param exact: Exact solution at each degree of freedom
    :type exact: Array
    :param grad_exact: Gradient of the xact solution at each degree of freedom
    :type grad_exact: Array
    :return: The H1-norm of the error.
    :rtype: float
    """
    integral1 = u.dot((exact-result)**2).dot(omega)
    dif_grad = np.array(u.dot(grad_exact)) - np.array([dxu.dot(result),dyu.dot(result)]).transpose()
    integral2 = (la.norm(dif_grad,axis=1)*la.norm(dif_grad,axis=1)).dot(omega)
    return np.sqrt(integral1 + integral2)