import sys
import numpy as np
import scipy.sparse as sparse
import scipy.signal as signal
import scipy.fft as fft
import pickle as pk
import gmsh
import numpy.linalg as la
from . import TimeEngine,Boundary
from . import FEM_3D as fem3d
from . import FEM_2D as fem2d

class Sources():

    @staticmethod
    def monopole_by_band(t_,band,sigma=None):
        """Return a forcing function representing a monopole source with max amplitude 1 at 1 meter. 
        The signal is a gaussian with the given cutoff frequencies (3dB).

        :param t_: Time-steps
        :type t_: Array
        :param band: Cutoff frequencies of the band to be covered
        :type band: int;int
        :param sigma: Bandwidth, defaults to None
        :type sigma: float, optional
        :return: Forcing function in time
        :rtype: Array
        """
        freq = (np.max(band)+np.min(band))/2 #Center frequency
        omega = 2*np.pi*freq

        if sigma==None:
            fc = (np.max(band)-np.min(band))
            sigma=1/(2*np.pi*fc/4) #TODO: What is the best sigma? To be studied

        t=t_-5*sigma #Shifts the gaussian so it can start at zero.

        _signal = np.sin(omega*t) * np.exp(-t**2/(2*sigma**2))
        _signal = _signal/np.max(np.abs(_signal))
        return 4 * np.pi * _signal

class Visualization():
    @staticmethod
    def addView(mesh,viewName,timeData,nodeTags):
        """Add the view to GMSH.

        :param mesh: Mesh instance
        :type mesh: FEM_2D.Mesh; FEM_3D.Mesh
        :param viewName: name of the new view
        :type viewName: str
        :param timeData: Values at the DOF to be shown
        :type timeData: Array (Timesteps)x(#dof)
        :param nodeTags: Node tags for each dof
        :type nodeTags: Array
        """
        viewTag = mesh.pos.add(viewName)
        for i_timeData in range(0,len(timeData)):
            mesh.pos.addModelData(viewTag,i_timeData,mesh.name,'NodeData',nodeTags,timeData[i_timeData].reshape(-1,1),numComponents=1)

    @staticmethod
    def showTime(mesh,viewNames,data,view_dofs,fltkrun=True):
        """Add the solution to the GMSH viewer.

        :param mesh: Mesh instance
        :type mesh: FEM_2D.Mesh; FEM_3D.Mesh
        :param viewName: name of the new view
        :type viewName: str
        :param data: Values at the DOF to be shown
        :type data: Array (Views)x(Timesteps)x(#dof)
        :param view_dofs: Index of the degrees of freedom that will be shown
        :type view_dofs: Array
        :param fltkrun: If the GMSH interface will be open automatically, defaults to True
        :type fltkrun: bool, optional
        """
        for i_view in range(0,len(data)):
            nodeTags = mesh.nodeTags[view_dofs[i_view]]
            timeData = data[i_view]
            Visualization.addView(mesh,viewNames[i_view],timeData,nodeTags)

        if (fltkrun):
            mesh.FLTKRun()

class Other():
    @staticmethod
    def nearest_dof(dofs,x):
        """Return the index of the nearest 3D degree of freedom

        :param dofs: The DOF coordinates
        :type dofs: Array n x 3
        :param x: The coordinate
        :type x: Array(3)
        :return: The index of the nearest degree of freedom
        :rtype: int
        """
        idx = np.argmin(la.norm(dofs - x,axis=1))
        return idx

    @staticmethod
    def printInline(text):
        sys.stdout.write(text + '                                            \r')

    @staticmethod
    def wiener_deconvolution(output, input, SNR=1):
        """Does a deconvolution of signals using the Wiener filter. The signals must have the same length

        :param output: The output signal
        :type output: Array
        :param input: The input signal
        :type input: Array
        :param SNR: The signal-to-noise ratio, defaults to 1
        :type SNR: float, optional
        :return: The deconvolved signal. A representation of the impulse response of the system
        :rtype: Array
        """
        H = fft.fft(input)
        deconvolved = np.real(fft.ifft(fft.fft(output)*np.conj(H)/(H*np.conj(H) + SNR**2)))
        return deconvolved