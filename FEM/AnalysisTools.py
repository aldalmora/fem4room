import numpy as np
import scipy.signal as signal
import scipy.fft as fft
import pickle as pk
import gmsh
import numpy.linalg as la
from . import TimeEngine

class Sources():
    @staticmethod
    def monopole_by_band(t_,band,sigma=None):
        """ Return a forcing function representing a monopole source. The signal is a gaussian with the given cutoff frequencies (3dB). """
        freq = (np.max(band)+np.min(band))/2 #Center frequency
        omega = 2*np.pi*freq

        if sigma==None:
            sigma=1/(omega/2) #1/4 of the band

        t=t_-5*sigma #Shifts the gaussian so it can start at zero.

        _signal = np.sin(omega*t) * np.exp(-t**2/(2*sigma**2))

        _signal = _signal * 1/(sigma*np.sqrt(2*np.pi))
        #_signal = _signal/np.max(np.abs(_signal))

        return 4 * np.pi * _signal

class Visualization():
    @staticmethod
    def addView(mesh,viewName,timeData,nodeTags):
        """timeData must be (Timesteps)x(#ddl)"""
        viewTag = mesh.pos.add(viewName)
        for i_timeData in range(0,len(timeData)):
            mesh.pos.addModelData(viewTag,i_timeData,mesh.name,'NodeData',nodeTags,timeData[i_timeData].reshape(-1,1),numComponents=1)

    @staticmethod
    def showTime(mesh,viewNames,data,view_ddls,fltkrun=True):
        """data must be (Views)x(Timesteps)x(#ddl)"""
        for i_view in range(0,len(data)):
            nodeTags = mesh.nodeTags[view_ddls[i_view]]
            timeData = data[i_view]
            Visualization.addView(mesh,viewNames[i_view],timeData,nodeTags)

        if (fltkrun):
            mesh.FLTKRun()

class SaveLoad():
    def save(self,name,saved_ddls,ddl,sln,sln_main,tspan):
        """Save the mesh and the main infos about the numerical result."""
        self.ddl = ddl
        self.sln = sln
        self.sln_main = sln_main
        self.tspan = tspan
        self.saved_ddls = saved_ddls
        gmsh.option.setNumber('Mesh.SaveAll',1)
        gmsh.write('data/' + name + '.msh')
        with open('data/' + name + '.pickle', 'wb') as f:
            pk.dump(self, f, pk.HIGHEST_PROTOCOL)

    def load(self,name,m):
        """Load the mesh and the main infos about a numerical result."""
        gmsh.open('data/' + name + '.msh')
        m.readMsh('data/' + name + '.msh')
        with open('data/' + name + '.pickle', 'rb') as f:
            T = pk.load(f)
            self.ddl = T.ddl
            self.sln = T.sln
            self.sln_main = T.sln_main
            self.tspan = T.tspan
            self.saved_ddls = T.saved_ddls

class Other():
    @staticmethod
    def nearest_ddl(ddls,x):
        """Return the index of the nearest 3D degree of freedom"""
        idx = np.argmin(la.norm(ddls - x,axis=1))
        return idx

class RoomResponse():
    """ Class that handles the calculation of the impulse response. """

    def __init__(self,M,C,K,c,dt,tf,f_band,idx_xS,idx_xR):
        self.M = M
        self.C = C
        self.K = K
        self.c = c
        self.dt = dt
        self.tf = tf
        self.f_band = f_band
        self.idx_xS = idx_xS  #TODO: uses the interpolation to get the value
        self.idx_xR = idx_xR
        self.nddls = M.shape[0]
        
        self.tspan = np.arange(0,self.tf,self.dt)
        self.tengine = TimeEngine.Newmark(self.M,self.C,self.K,self.dt,alpha=.25,delta=.5)

        self.time_monopole = Sources.monopole_by_band(self.tspan,self.f_band)

        select_ddl = np.zeros(self.nddls)
        select_ddl[idx_xS] = 1
        self.forcing_function = lambda time_index: self.time_monopole[time_index] * select_ddl

    def run(self):
        """ Calculate the room response and return: Discretized frequencies, input spectrum, output spectrum, frequency response, output time response. """
        u0 = np.zeros(self.nddls)
        du0 = np.zeros(self.nddls)
        ddu0 = np.zeros(self.nddls)
        s,s_main = self.tengine.solve(self.tspan,self.forcing_function,u0,du0,ddu0,self.idx_xR,[],1)

        all_frequencies=fft.rfftfreq(len(self.tspan),self.dt)
        valid_band_idx = np.arange((np.abs(all_frequencies-self.f_band[0])).argmin(),(np.abs(all_frequencies-self.f_band[1])).argmin()+1)
        frequencies = all_frequencies[valid_band_idx]
        forcing = [self.forcing_function(i)[self.idx_xS] for i in range(0,len(self.tspan))]
        f_forcing = fft.rfft(forcing)[valid_band_idx]
        f_result = []
        f_ratio = []
        for idx in range(0,len(self.idx_xR)):
            f_result.append(fft.rfft(s_main[:,idx])[valid_band_idx])
            f_ratio.append(fft.rfft(s_main[:,idx])[valid_band_idx]/f_forcing)

        return frequencies,f_forcing,f_result,f_ratio,s_main