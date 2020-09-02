import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sla
from . import Tools
import scikits.umfpack

class Newmark_iterative:
    # M ddu + C du + K u = f
    # a0 = 1/(alpha*dt**2)
    # a1 = delta/(alpha*dt)
    # a2 = 1/(alpha*dt)
    # a3 = 1/(2*alpha) - 1
    # a4 = delta/alpha - 1
    # a5 = dt/2 * (delta/(alpha-2))
    # a6 = dt*(1-delta)
    # a7 = delta*dt
    # Keff = K + a0*M + a1*C
    # R = f(t+dt) + M(a0*u + a2*du + a3*ddu) + C(a1*u + a4*du + a5*ddu)

    # Systeme : Keff u(t+dt) = R(t+dt)
    # delta > 0.5
    # alpha > 0.25(0.5 + delta)^2
    # Triangularize Keff
    def __init__(self,M,C,K,dt,alpha,delta):
        self.alpha = alpha
        self.delta = delta
        self.dt = dt
        self.a0 = 1/(alpha*(dt**2))
        self.a1 = delta/(alpha*dt)
        self.a2 = 1/(alpha*dt)
        self.a3 = 1/(2*alpha) - 1
        self.a4 = delta/alpha - 1
        self.a5 = (dt/2) * ((delta/alpha)-2)
        self.a6 = dt*(1-delta)
        self.a7 = delta*dt
        self.tspan = []
        self.u = []

        self.M = M
        self.C = C
        sla.use_solver(useUmfpack=True)
        self.Keff = K + M.multiply(self.a0) + C.multiply(self.a1)
        self.Keff.indices = self.Keff.indices.astype(np.int)
        self.Keff.indptr = self.Keff.indptr.astype(np.int)

        #Initial conditions
        self.u0 = np.zeros(M.shape[0])
        self.du0 = np.zeros(M.shape[0])
        self.ddu0 = np.zeros(M.shape[0])

    def setInitialConditions(self,u0,du0,ddu0):
        self.u0 = u0
        self.du0 = du0
        self.ddu0 = ddu0

    def preconditionate(self,Keff):
        ilu = sla.spilu(Keff)
        Mx = lambda x: ilu.solve(x)
        K_prec = sla.LinearOperator(Keff.shape, Mx)
        return K_prec
        
    def solve(self,tspan,F,main_dofs,saved_dofs,save_time_step): #TODO: Check if the F_Matrix is being passed
        self.tspan=tspan

        Keff_solve = 0
        self.Keff_prec = self.preconditionate(self.Keff)

        ret_u = [self.u0[saved_dofs]]
        ret_main_dofs=[self.u0[main_dofs]]
        u = self.u0
        du = self.du0
        ddu = self.ddu0
        for i in range(1,len(tspan)):
            f_new = -F(i)

            u,du,ddu = self.iterate(Keff_solve,f_new,u,du,ddu)
            ret_main_dofs.append(u[main_dofs])

            if i % save_time_step < 1e-20:
                ret_u.append(u[saved_dofs])

            Tools.Other.printInline('Time Step: ' + str(i+1) + '/' + str(len(tspan)))

        ret_u = np.array(np.real(ret_u))
        ret_main_dofs = np.array(np.real(ret_main_dofs))
        Tools.Other.printInline('\n\r')
        self.u = ret_u
        return ret_u,ret_main_dofs

    def iterate(self,Keff_solve,f_new,u,du,ddu):
        R_new = f_new + self.M.dot(self.a0*u + self.a2*du + self.a3*ddu) + self.C.dot(self.a1*u + self.a4*du + self.a5*ddu)

        u_new = sla.gcrotmk(self.Keff,R_new,M=self.Keff_prec,x0=u,atol=1e-5)[0]

        ddu_new = self.a0*(u_new - u) - self.a2*du - self.a3*ddu
        du_new = du + self.a6*ddu + self.a7*ddu_new
        return u_new,du_new,ddu_new

class Newmark:
    # M ddu + C du + K u = f
    # a0 = 1/(alpha*dt**2)
    # a1 = delta/(alpha*dt)
    # a2 = 1/(alpha*dt)
    # a3 = 1/(2*alpha) - 1
    # a4 = delta/alpha - 1
    # a5 = dt/2 * (delta/(alpha-2))
    # a6 = dt*(1-delta)
    # a7 = delta*dt
    # Keff = K + a0*M + a1*C
    # R = f(t+dt) + M(a0*u + a2*du + a3*ddu) + C(a1*u + a4*du + a5*ddu)

    # Systeme : Keff u(t+dt) = R(t+dt)
    # delta > 0.5
    # alpha > 0.25(0.5 + delta)^2
    # Triangularize Keff
    def __init__(self,M,C,K,dt,alpha,delta):
        self.alpha = alpha
        self.delta = delta
        self.dt = dt
        self.a0 = 1/(alpha*(dt**2))
        self.a1 = delta/(alpha*dt)
        self.a2 = 1/(alpha*dt)
        self.a3 = 1/(2*alpha) - 1
        self.a4 = delta/alpha - 1
        self.a5 = (dt/2) * ((delta/alpha)-2)
        self.a6 = dt*(1-delta)
        self.a7 = delta*dt
        self.tspan = []
        self.u = []

        self.M = M
        self.C = C
        sla.use_solver(useUmfpack=True)
        self.Keff = K + M.multiply(self.a0) + C.multiply(self.a1)
        self.Keff.indices = self.Keff.indices.astype(np.int64)
        self.Keff.indptr = self.Keff.indptr.astype(np.int64)

        #Initial conditions
        self.u0 = np.zeros(M.shape[0])
        self.du0 = np.zeros(M.shape[0])
        self.ddu0 = np.zeros(M.shape[0])

    def setInitialConditions(self,u0,du0,ddu0):
        self.u0 = u0
        self.du0 = du0
        self.ddu0 = ddu0
        
    def solve(self,tspan,f,main_dofs,saved_dofs,save_time_step):
        self.tspan=tspan

        Keff_solve = sla.factorized(self.Keff)

        ret_u = [self.u0[saved_dofs]]
        ret_main_dofs=[self.u0[main_dofs]]
        u = self.u0
        du = self.du0
        ddu = self.ddu0
        for i in range(1,len(tspan)):
            f_new = -f(i)

            u,du,ddu = self.iterate(Keff_solve,f_new,u,du,ddu)
            ret_main_dofs.append(u[main_dofs])

            if i % save_time_step < 1e-20:
                ret_u.append(u[saved_dofs])

            Tools.Other.printInline('Time Step: ' + str(i+1) + '/' + str(len(tspan)))

        ret_u = np.array(ret_u)
        ret_main_dofs = np.array(ret_main_dofs)
        Tools.Other.printInline('\n\r')
        self.u = ret_u
        return ret_u,ret_main_dofs

    def iterate(self,Keff_solve,f_new,u,du,ddu):
        R_new = f_new + self.M.dot(self.a0*u + self.a2*du + self.a3*ddu) + self.C.dot(self.a1*u + self.a4*du + self.a5*ddu)

        u_new = Keff_solve(R_new)

        ddu_new = self.a0*(u_new - u) - self.a2*du - self.a3*ddu
        du_new = du + self.a6*ddu + self.a7*ddu_new
        return u_new,du_new,ddu_new

class LeapFrog():
    #(M + dt*C + (dt**2)*K) u_new = (2M + dt*C) u1 - M u0 - (dt**2)f[n+1]
    def __init__(self,M,C,K,dt):
        self.dt = dt
        self.tspan = []
        self.u = []

        self.M = M
        self.C = C
        self.K = K
        sla.use_solver(useUmfpack=True)
        self.sys = M + dt*C + (dt**2)*K
        self.sys.indices = self.sys.indices.astype(np.int64)
        self.sys.indptr = self.sys.indptr.astype(np.int64)

        self.a0 = self.M.multiply(2) + self.C.multiply(self.dt)
        self.a1 = self.dt**2

    def solve(self,tspan,f,u0,du0,ddu0,main_dofs,saved_dofs,save_time_step):
        self.tspan=tspan
        sys_solve = sla.factorized(self.sys)
        ret_u = [u0[saved_dofs]]
        ret_main_dofs=[u0[main_dofs]]
        u0 = u0
        u1 = u0
        for i in range(1,len(tspan)):
            f_new = -f(i)
            u_new = self.iterate(sys_solve,f_new,u0,u1)
            ret_main_dofs.append(u_new[main_dofs])
            if i % save_time_step < 1e-20:
                ret_u.append(u_new[saved_dofs])
            u0=u1
            u1=u_new

        ret_u = np.array(ret_u)
        ret_main_dofs = np.array(ret_main_dofs)
        self.u = ret_u
        return ret_u,ret_main_dofs

    def iterate(self,sys_solve,f_new,u0,u1):
        R_new = self.a0*u1 - self.M.dot(u0) - self.a1*f_new
        u_new = sys_solve(R_new)
        return u_new