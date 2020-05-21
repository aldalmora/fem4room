import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sla

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
        self.a0 = 1/(alpha*dt**2)
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
        self.Keff = K + self.a0*M + self.a1*C
        self.Keff.indices = self.Keff.indices.astype(np.int64)
        self.Keff.indptr = self.Keff.indptr.astype(np.int64)

    def solve(self,f,t,u0,du0,ddu0,main_ddl,saved_ddls,save_time_step):
        self.tmax = t
        self.t = 0
        Keff_solve = sla.factorized(self.Keff)
        tspan = [self.t]
        ret_u = [u0[saved_ddls]]
        ret_main_ddl=[u0[main_ddl]]
        u = u0
        du = du0
        ddu= ddu0
        tcount=0
        while (self.t <= self.tmax):
            f_new = f(self.t+self.dt)
            u,du,ddu = self.iterate(Keff_solve,f_new,u,du,ddu)
            tcount+=1
            ret_main_ddl.append(u[main_ddl])
            if tcount % save_time_step < 1e-20:
                ret_u.append(u[saved_ddls])
            self.t += self.dt
            tspan.append(self.t)

        ret_u = np.array(ret_u)
        ret_main_ddl = np.array(ret_main_ddl)
        tspan = np.array(tspan)
        self.u = ret_u
        self.tspan = tspan
        return tspan,ret_u,ret_main_ddl

    def iterate(self,Keff_solve,f_new,u,du,ddu):
        R_new = f_new + self.M*(self.a0*u + self.a2*du + self.a3*ddu) + self.C*(self.a1*u + self.a4*du + self.a5*ddu)
        u_new = Keff_solve(R_new)
        ddu_new = self.a0*(u_new - u) - self.a2*du - self.a3*ddu
        du_new = du + self.a6*ddu + self.a7*ddu_new
        return u_new,du_new,ddu_new