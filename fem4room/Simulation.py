
import numpy as np
from . import TimeEngine,Boundary,Sources
from . import FEM_3D as fem3d
import gmsh
import scipy.sparse as sparse
import scipy.signal as signal
import scipy.fft as fft
import scipy.io.wavfile as wavfile
import time
import pickle as pkl

#TODO: Userfriendly errors and instructions
class Simulation():
    """ Class that handles the calculation of the room response (time and frequency) for a given mesh with defined impedances at a surface. """

    def __init__(self,simulationsTime,nameSource,nameReceivers):
        """ Initialize the simulation configurations. """
        self.nameSource = nameSource
        self.nameReceivers = nameReceivers
        self.simulationsTime = simulationsTime #TODO: Check sizes before simulation

        #Default parameters
        self.c = 340.
        self.cfl = .8
        self.elementsPerWavelength = 6
        self.element_order = 1
        self.max_h = 1e10

        #Initializations
        self.impedancesByBand = {}
        self.GroupTags = {}
        self.outputPath = None
        self.refine_BC = False

        #Default bands
        self.bands_freq = {
        '50'   : [ 44.194 , 55.681 ],
        '63'   : [ 55.681 , 70.154 ],
        '80'   : [ 70.154 , 88.388 ],
        '100'  : [ 88.388 , 111.362 ],
        '125'  : [ 111.362 , 140.308 ],
        '160'  : [ 140.308 , 176.777 ],
        '200'  : [ 176.777 , 222.725 ],
        '250'  : [ 222.725 , 280.616 ],
        '315'  : [ 280.616 , 353.553 ],
        '400'  : [ 353.553 , 445.449 ],
        '500'  : [ 445.449 , 561.231 ],
        '630'  : [ 561.231 , 707.107 ],
        '800'  : [ 707.107 , 890.899 ]} 

    def defineFrequencyBands(self,bands):
        """ Change the dictionary of bands and its cut-off frequencies. """
        self.bands_freq = bands

    def defineParameters(self,c=340.,elementsPerWavelength=6,cfl=.8):
        """ Change the default parameters for sound speed, elements per wavelength and the CFL condition number. """
        self.c = c
        self.elementsPerWavelength = elementsPerWavelength
        self.cfl = cfl

    def setMaxStepSize(self, h):
        """ Define a max-size for the elements size. """
        self.max_h = h

    def setBCRefinement(self, max_h):
        """ Set specific mesh-size for the meshes at boundaries. """
        self.refine_BC = True
        self.BC_max_h = max_h

    def defineAbsorption(self, physicalGroup, absorptionCoefsByBand):
        """ For a given surface name, set the absorption coefficients for each band. """
        self.impedancesByBand[physicalGroup] = {}
        for band in absorptionCoefsByBand:
            ref_coef=np.sqrt(1-absorptionCoefsByBand[band])
            impedancesByBand = (1+ref_coef)/(1-ref_coef)
            self.impedancesByBand[physicalGroup][band] = impedancesByBand

    def defineImpedance(self, physicalGroup, impedancesByBand):
        """ For a given surface name, set the impedance for each band. """
        self.impedancesByBand[physicalGroup] = impedancesByBand

    def readGeometryDXF(self,file):
        """ Read a geometry described in the DXF format. """
        m = fem3d.Mesh(file)
        m.readDXF(file)
        self.defineMesh(m)

    def readGeometryGEO(self,file):
        """ Read a geometry described in GEO format. """
        m = fem3d.Mesh(file)
        m.readGeo(file)
        self.defineMesh(m)

    def defineMesh(self, mesh: fem3d.Mesh):
        """ Set the mesh instance and load the surfaces. """
        self.mesh = mesh
        for dimTag in self.mesh.model.getPhysicalGroups():
            self.GroupTags[self.mesh.model.getPhysicalName(dimTag[0],dimTag[1])] = dimTag

    def setOutputPath(self, path):
        """ Set the output to the WAV files from the source and receivers. """
        self.outputPath = path

    def run(self):
        """ Run the simulation for each band and stores its results. """
        source_signals = {}
        time_arrays = {}
        receiver_signals = {}
        t1 = time.time()
        self.saveParameters()
        groups_tags = self.GroupTags
        
        for band in self.bands_freq:
            t1_band = time.time()

            band_range = np.array(self.bands_freq[band])
            print('Starting band ' + band)
            self.mesh.model.mesh.clear()

            #Set the parameters specific for the band frequencies
            wavelengths=self.c/band_range
            h=np.min(wavelengths)/self.elementsPerWavelength
            h=np.min([h,self.max_h])
            dt=self.cfl*h/self.c #CFL Condition
            tf = self.simulationsTime[band]

            if tf==0: continue

            self.mesh.fac.synchronize()
            pTags = self.mesh.model.getEntities(0)
            self.mesh.model.occ.setMeshSize(pTags,h)
            self.mesh.model.mesh.setSize(pTags,h)

            #Impedance Surfaces Refinement
            if self.refine_BC:
                h_bc = np.min([h,self.BC_max_h])
                for material in self.impedancesByBand:
                    material_tag = groups_tags[material]
                    materialSurfaceTag = self.mesh.model.getEntitiesForPhysicalGroup(material_tag[0],material_tag[1])
                    pSurfaceTag = self.mesh.model.getBoundary([(2,i) for i in materialSurfaceTag],recursive=True)
                    self.mesh.model.occ.setMeshSize(pSurfaceTag,h_bc)
                    self.mesh.model.mesh.setSize(pSurfaceTag,h_bc)


            self.mesh.fac.synchronize()
            self.mesh.generate()

            #Assemble the FEM matrices
            engine = fem3d.Engine(self.mesh,self.element_order,self.element_order)
            nddls = len(engine.ddl)
            print('h: ',str(h), ' -- ddls:', nddls)
            M = engine.M_Matrix()/(self.c**2)
            C = sparse.csc_matrix(M.shape)
            K = engine.K_Matrix()

            #Impedance B.C
            material_impedances = self.impedancesByBand
            for material in self.impedancesByBand:
                Z = material_impedances[material][band]
                material_tag = groups_tags[material][1]

                M_2d,nodeTags_imp = Boundary.Impedance_Damping_Matrix(self.mesh,material_tag)
                nodeTags_idx = np.where(nodeTags_imp[:,None]==self.mesh.nodeTags)[1]
                idx_row = nodeTags_idx[M_2d.row]
                idx_col = nodeTags_idx[M_2d.col]
                C_material = sparse.coo_matrix((M_2d.data, (idx_row, idx_col)), M.shape).tocsc()
                C_material = C_material/(self.c*Z)

                C = C + C_material

            #DOF index of the source:
            sourceDimTag = groups_tags[self.nameSource]
            sourceNodeTag = self.mesh.model.getEntitiesForPhysicalGroup(sourceDimTag[0],sourceDimTag[1])
            idx_xS = np.where(self.mesh.nodeTags==sourceNodeTag)[0]

            #DOF indexes of the receivers
            idx_xR = []
            for name in self.nameReceivers:
                receiverDimTag = groups_tags[name]
                receiverNodeTag = self.mesh.model.getEntitiesForPhysicalGroup(receiverDimTag[0],receiverDimTag[1])
                idx_xR.extend(np.where(self.mesh.nodeTags==receiverNodeTag)[0])

            #Setup the time scheme
            tspan = np.arange(0,tf,dt)
            # tengine = TimeEngine.Newmark(M,C,K,dt,alpha=.25,delta=.5)
            tengine = TimeEngine.Newmark_iterative(M,C,K,dt,alpha=.25,delta=.5)

            time_monopole = Sources.monopole_by_band(tspan,band_range)

            select_ddl = np.zeros(nddls)
            select_ddl[idx_xS] = 1
            forcing_function = lambda time_index: time_monopole[time_index] * select_ddl
            F = engine.F_Matrix(forcing_function)

            s,s_main = tengine.solve(tspan,F,idx_xR,[],1)
            receiver_signals[band] = s_main.T
            time_arrays[band] = tspan
            source_signals[band] = - time_monopole/(4*np.pi) #TODO Describe the negative sign

            self.saveOutput(band, self.nameSource,source_signals[band],receiver_signals[band],16000)

            t2_band = time.time()
            print('Simulation time (' + band + '): ', str(t2_band-t1_band))
        t2=time.time()
        print('Total simulation time: ', str(t2-t1))
        self.time_arrays = time_arrays
        self.source_signals = source_signals
        self.receiver_signals = receiver_signals
        return time_arrays,self.source_signals,self.receiver_signals
    
    def saveParameters(self):
        """ Store in a file parameters.txt the parameters used in the simulation. """
        if (self.outputPath!=None):
            with open(self.outputPath + 'parameters.txt', 'a') as the_file:
                the_file.write('Source:' + str(self.nameSource) + '\n')
                the_file.write('Receivers:' + str(self.nameReceivers) + '\n')
                the_file.write('Elements per Wavelength:' + str(self.elementsPerWavelength) + '\n')
                the_file.write('Bands:' + str(self.bands_freq) + '\n')
                the_file.write('Simulation times:' + str(self.simulationsTime) + '\n')
                the_file.write('Impedances:' + str(self.impedancesByBand) + '\n')

    def saveOutput(self, band, source_name, source_signal, receiver_signals, save_fs): 
        """ Save the signals as WAV. Source and Receivers for each band. """
        if (self.outputPath!=None):
            source_signal = signal.resample(source_signal,int(self.simulationsTime[band]*save_fs)).astype(np.float32)
            normalize_by = np.max(source_signal)
            source_signal = source_signal/normalize_by
            wavfile.write(self.outputPath + str(band) + "_" + source_name + ".wav",save_fs,source_signal) 
            
            for i_rec in range(0,len(self.nameReceivers)):
                receiver_name = self.nameReceivers[i_rec]
                receiver_signal = receiver_signals[i_rec]
                receiver_signal = signal.resample(receiver_signal,int(self.simulationsTime[band]*save_fs)).astype(np.float32)
                receiver_signal = receiver_signal/normalize_by
                wavfile.write(self.outputPath + str(band) + "_" + receiver_name + ".wav",save_fs,receiver_signal) 

    def calcFrequencyResponse(self, receiverIndex = 0): #TODO EXPLAIN
        """ Return the frequency response given by the simulation for the specific receiver. """
        resampled_tspan = {}
        resampled_source_signals = {}
        resampled_receiver_signals = {}
        sources_fft = {}
        receivers_fft = {}

        #Resample every signal to the max sampling frequency of the bands
        min_band = str(np.min(list(map(int,self.source_signals.keys())))) #Band name/key
        max_band = str(np.max(list(map(int,self.source_signals.keys())))) #Band name/key
        min_freq = np.min(self.bands_freq[min_band]) #Frequence value
        max_freq = np.max(self.bands_freq[max_band]) #Frequence value

        #Get the max sampling frequency of the simulated signals
        max_fs = 1/(self.time_arrays[max_band][1]-self.time_arrays[max_band][0])

        #Resample all the signals from the source and the selected receiver 
        for band in self.source_signals:
            new_size = int(np.max(self.time_arrays[band])*max_fs)
            rsource, rtspan = signal.resample(self.source_signals[band],new_size,self.time_arrays[band])
            rreceiver = signal.resample(self.receiver_signals[band][receiverIndex],new_size)

            resampled_tspan[band] = rtspan
            resampled_source_signals[band] = rsource
            resampled_receiver_signals[band] = rreceiver
            sources_fft[band] = fft.rfft(rsource)
            receivers_fft[band] = fft.rfft(rreceiver)
        
        #Include the response for each band, only for its frequencies
        frequencies = []
        response = []
        for band in self.source_signals:
            _band_freqs = fft.rfftfreq(len(resampled_tspan[band]),1/max_fs)
            idx_from = np.argmin(np.abs(_band_freqs-self.bands_freq[band][0]))
            idx_to = np.argmin(np.abs(_band_freqs-self.bands_freq[band][1]))
            band_response = receivers_fft[band]/sources_fft[band]
            frequencies.extend(_band_freqs[idx_from:idx_to])
            response.extend(band_response[idx_from:idx_to])

        frequencies = np.array(frequencies)
        response = np.array(response)

        return frequencies, response