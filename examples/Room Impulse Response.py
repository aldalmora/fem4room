"""

"""
from fem4room import Simulation
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import pickle as pkl

nameSource = "Source_Genelec"
nameReceivers = ["Receiver_Ambi",
                "Receiver_Array1",
                "Receiver_Array2",
                "Receiver_Array3",
                "Receiver_Array4",
                "Receiver_Array5",
                "Receiver_Array6",
                "Receiver_Array7",
                "Receiver_Array8",
                "Receiver_Array9",
                "Receiver_Array10",
                "Receiver_Array11",
                "Receiver_Array12",
                "Receiver_Array13",
                "Receiver_Array14",
                "Receiver_Array15",
                "Receiver_Array16",
                "Receiver_Array17",
                "Receiver_Array18",
                "Receiver_Array19",
                "Receiver_Array20",
                "Receiver_Array21",
                "Receiver_Array22",
                "Receiver_Array23",
                "Receiver_Array24",
                "Receiver_Array25"]

simulationsTime = {
    '50'   : 6
    ,'63'   : 4
    ,'80'   : 4
    ,'100'  : 3
    ,'125'  : 3
    ,'160'  : 2
    ,'200'  : 2
    ,'250'  : 1
    ,'315'  : 0.6
    ,'400'  : 0.6
    ,'500'  : 0.6
    ,'630'  : 0.6
    ,'800'  : 0.6
}

# simulationsTime = {
#     '50'    : 0
#     ,'63'   : 0 
#     ,'80'   : 0
#     ,'100'  : 0
#     ,'125'  : 0
#     ,'160'  : 0
#     ,'200'  : 0
#     ,'250'  : 0
#     ,'315'  : 0
#     ,'400'  : 0
#     ,'500'  : 0
#     ,'630'  : 0
#     ,'800'  : 0
# }

Parphonic100_Abs = {
    '50'   : 0.08,
    '63'   : 0.14,
    '80'   : 0.25,
    '100'  : 0.46,
    '125'  : 0.77,
    '160'  : 1,
    '200'  : 1,
    '250'  : 1,
    '315'  : 1,
    '400'  : 1,
    '500'  : 1,
    '630'  : 1,
    '800'  : 1
}

Parphonic60_Abs = {
    '50'   : 0.04,
    '63'   : 0.05,
    '80'   : 0.09,
    '100'  : 0.16,
    '125'  : 0.26,
    '160'  : 0.43,
    '200'  : 0.64,
    '250'  : 0.82,
    '315'  : 0.91,
    '400'  : 0.94,
    '500'  : 0.94,
    '630'  : 0.94,
    '800'  : 0.95
}

Ultimate_Abs = {
    '50'   : 0.05,
    '63'   : 0.06,
    '80'   : 0.07,
    '100'  : 0.05,
    '125'  : 0.10,
    '160'  : 0.42,
    '200'  : 0.94,
    '250'  : 1,
    '315'  : 1,
    '400'  : 1,
    '500'  : 1,
    '630'  : 1,
    '800'  : 0.98
}

Rigid_Abs = {
    '50'   : 0.01,
    '63'   : 0.01,
    '80'   : 0.01,
    '100'  : 0.01,
    '125'  : 0.01,
    '160'  : 0.01,
    '200'  : 0.01,
    '250'  : 0.01,
    '315'  : 0.01,
    '400'  : 0.02,
    '500'  : 0.02,
    '630'  : 0.02,
    '800'  : 0.02
}

Rigid_Abs = {
    '50'   : 0.012,
    '63'   : 0.020,
    '80'   : 0.020,
    '100'  : 0.032,
    '125'  : 0.024,
    '160'  : 0.019,
    '200'  : 0.026,
    '250'  : 0.028,
    '315'  : 0.029,
    '400'  : 0.032,
    '500'  : 0.034,
    '630'  : 0.033,
    '800'  : 0.038
}

simulation = Simulation(simulationsTime,nameSource,nameReceivers)
# simulation.readGeometryDXF('examples/RoomWool.dxf')
# simulation.readGeometryDXF('examples/ReverberantRoom.dxf')
simulation.readGeometryGEO('examples/RoomWool.geo')

simulation.setOutputPath('C:/Simulations/NP4_Genelec(GMSH Adj)/')
simulation.setMaxStepSize(0.4)
simulation.setParameters(c=340, elementsPerWavelength=6, cfl=.8)

simulation.setAbsorption("Parphonic100", Parphonic100_Abs)
simulation.setAbsorption("Parphonic60",  Parphonic60_Abs)
simulation.setAbsorption("Ultimate",     Ultimate_Abs)
simulation.setAbsorption("Vieillelaine", Ultimate_Abs)
simulation.setAbsorption("Rigid_Surface", Rigid_Abs)
times,source_signals,receiver_signals = simulation.run()
frequencies_all, response_all = simulation.calcFrequencyResponse()