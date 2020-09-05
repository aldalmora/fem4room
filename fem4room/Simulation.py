
import numpy as np
from . import TimeEngine,Boundary,Sources,Other
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
    """Class that handles the calculation of the room response (time and frequency) for a 
    given mesh with defined impedances at a surface.
    """

    def __init__(self,simulationsTime,nameSource,nameReceivers):
        """Initialize the simulation configurations.

        :param simulationsTime: Dictionary of bands with the simulation times.
        :type simulationsTime: Dict(String;float)
        :param nameSource: Physical group name(GMSH) of the source node.  
        :type nameSource: string
        :param nameReceivers: Physical group name(GMSH) of the receivers nodes. 
        :type nameReceivers: Array(string)
        """
        self.nameSource = nameSource
        self.nameReceivers = nameReceivers
        self.simulationsTime = simulationsTime

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

    def setFrequencyBands(self,bands):
        """Change the dictionary of bands and its cut-off frequencies.

        :param bands: Dictionary of the band edges
        :type bands: Dict(string;Array)
        """
        self.bands_freq = bands

    def setParameters(self,c=340.,elementsPerWavelength=6,cfl=.8):
        """Change the default parameters for sound speed, elements per wavelength and the CFL condition number.

        :param c: Sound speed, defaults to 340.
        :type c: float, optional
        :param elementsPerWavelength: Number of elements per wavelength, defaults to 6
        :type elementsPerWavelength: int, optional
        :param cfl: CFL condition number, defaults to .8
        :type cfl: float, optional
        """
        self.c = c
        self.elementsPerWavelength = elementsPerWavelength
        self.cfl = cfl

    def setMaxStepSize(self, h):
        """Define a max-size for the elements size.

        :param h: Elements size
        :type h: float
        """
        self.max_h = h

    def setBCRefinement(self, max_h):
        """Set specific mesh-size for the meshes at boundaries.

        :param max_h: Maximum size for the elements in the boundaries.
        :type max_h: float
        """
        self.refine_BC = True
        self.BC_max_h = max_h

    def setAbsorption(self, physicalGroupName, absorptionCoefsByBand):
        """For a given surface name, set the absorption coefficients for each band.

        :param physicalGroupName: Physical group name(GMSH) of the impedance surface
        :type physicalGroupName: string
        :param absorptionCoefsByBand: Dictionary with the absorption coef. for each band
        :type absorptionCoefsByBand: Dict(string;float)
        """
        self.impedancesByBand[physicalGroupName] = {}
        for band in absorptionCoefsByBand:
            ref_coef=np.sqrt(1-absorptionCoefsByBand[band])
            impedancesByBand = (1+ref_coef)/(1-ref_coef)
            self.impedancesByBand[physicalGroupName][band] = impedancesByBand

    def setImpedance(self, physicalGroupName, impedancesByBand):
        """For a given surface name, set the impedance for each band.

        :param physicalGroupName: Physical group name(GMSH) of the impedance surface
        :type physicalGroupName: string
        :param impedancesByBand: Dictionary with the complex impedance for each band
        :type impedancesByBand: Dict(string;float)
        """
        self.impedancesByBand[physicalGroupName] = impedancesByBand

    def readGeometryDXF(self,file):
        """Read a geometry described in the DXF format.

        :param file: File name/path
        :type file: String
        """
        m = fem3d.Mesh(file)
        m.readDXF(file)
        self.setMesh(m)

    def readGeometryGEO(self,file):
        """Read a geometry described in GEO format.

        :param file: File name/path
        :type file: String
        """
        m = fem3d.Mesh(file)
        m.readGeo(file)
        self.setMesh(m)

    def setMesh(self, mesh: fem3d.Mesh):
        """Set the mesh instance and load the surfaces.

        :param mesh: Mesh instance.
        :type mesh: FEM_3D.Mesh
        """
        self.mesh = mesh
        for dimTag in self.mesh.model.getPhysicalGroups():
            self.GroupTags[self.mesh.model.getPhysicalName(dimTag[0],dimTag[1])] = dimTag

    def setOutputPath(self, path):
        """Set the output to the WAV files from the source and receivers.

        :param path: Folder path
        :type path: string
        """
        self.outputPath = path

    def run(self): #TODO: Write status?
        """Run the simulation for each band and stores its results.

        :return: (For each band)The array with the time steps, 
        The source signals (monopole, pressure at 1 meter distant), 
        The "measured" signals at the receiver points
        :rtype: Dict(String;Array), Dict(String;Array), Dict(String;Array(Array))
        """
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
            self.mesh.model.mesh.setSize(pTags,h)
            self.mesh.model.mesh.setSize(pTags,h)

            #Impedance Surfaces Refinement
            if self.refine_BC:
                h_bc = np.min([h,self.BC_max_h])
                for material in self.impedancesByBand:
                    material_tag = groups_tags[material]
                    materialSurfaceTag = self.mesh.model.getEntitiesForPhysicalGroup(material_tag[0],material_tag[1])
                    pSurfaceTag = self.mesh.model.getBoundary([(2,i) for i in materialSurfaceTag],recursive=True)
                    self.mesh.model.mesh.setSize(pSurfaceTag,h_bc)
                    self.mesh.model.mesh.setSize(pSurfaceTag,h_bc)

            self.mesh.fac.synchronize()
            self.mesh.generate()

            #Assemble the FEM matrices
            engine = fem3d.Engine(self.mesh,self.element_order,self.element_order)
            ndofs = len(engine.dof)
            print('h: ',str(h), ' -- dofs:', ndofs)
            M = engine.M_Matrix()/(self.c**2)
            C = sparse.csc_matrix(M.shape)
            K = engine.K_Matrix()

            #Impedance B.C
            material_impedances = self.impedancesByBand
            for material in self.impedancesByBand:
                Z = material_impedances[material][band]
                material_tag = groups_tags[material][1]

                M_2d,nodeTags_imp = Boundary.Surface_Mass_Matrix(self.mesh,material_tag)
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

            select_dof = np.zeros(ndofs)
            select_dof[idx_xS] = 1
            F = lambda time_index: time_monopole[time_index] * select_dof

            s,s_main = tengine.solve(tspan,F,idx_xR,[],1)
            receiver_signals[band] = s_main.T
            time_arrays[band] = tspan
            source_signals[band] = - time_monopole/(4*np.pi)

            self.saveWav(band, self.nameSource,source_signals[band],receiver_signals[band],16000)

            t2_band = time.time()
            print('Simulation time (' + band + '): ', str(t2_band-t1_band))
        t2=time.time()
        print('Total simulation time: ', str(t2-t1))
        self.time_arrays = time_arrays
        self.source_signals = source_signals
        self.receiver_signals = receiver_signals
        self.saveSimulation()
        return time_arrays,self.source_signals,self.receiver_signals
    
    def saveParameters(self):
        """Store in a file parameters.txt the parameters used in the simulation. """
        if (self.outputPath!=None):
            with open(self.outputPath + 'parameters.txt', 'a') as the_file:
                the_file.write('Source:' + str(self.nameSource) + '\n')
                the_file.write('Receivers:' + str(self.nameReceivers) + '\n')
                the_file.write('Elements per Wavelength:' + str(self.elementsPerWavelength) + '\n')
                the_file.write('Bands:' + str(self.bands_freq) + '\n')
                the_file.write('Simulation times:' + str(self.simulationsTime) + '\n')
                the_file.write('Impedances:' + str(self.impedancesByBand) + '\n')

    def saveWav(self, band, source_name, source_signal, receiver_signals, save_fs): 
        """Save the signals as WAV. Source and Receivers for each band.

        :param band: Band that is being saved
        :type band: string
        :param source_name: Name of the source
        :type source_name: string
        :param source_signal: Source signal
        :type source_signal: Array
        :param receiver_signals: Receivers signals
        :type receiver_signals: Array(Array)
        :param save_fs: Sampling frequency
        :type save_fs: int
        """
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

    def saveSimulation(self):
        """Save the simulated signals
        """
        if (self.outputPath!=None):
            with open(self.outputPath + 'simulation.pkl', 'wb') as f:
                pkl.dump((self.time_arrays,self.source_signals,self.receiver_signals), f, pk.HIGHEST_PROTOCOL)
    
    def loadSimulation(self,pkl_path):
        """Load the simulated signals

        :param pkl_path: path to the simulation.pkl
        :type pkl_path: str
        """
        with open(pkl_path, 'rb') as f:
            self.time_arrays,self.source_signals,self.receiver_signals = pkl.load(f)

    def __ressample_pad(self,new_fs):
        """Return all the simulated signals ressampled to the given frequency and with same size.

        :param new_fs: New sampling frequency
        :type new_fs: int
        :return: Ressampled source signals and receiver signals
        :rtype: Dict(String;Array), Dict(String;Array(Array))
        """
        resampled_source_signals = {}
        resampled_receiver_signals = {}

        durations = [self.simulationsTime[b] for b in self.simulationsTime]
        max_duration = np.max(durations)
        new_size = int(max_duration*new_fs)

        #Resample all the signals from the source and the selected receiver 
        for band in self.source_signals:
            new_ressample_size = int(np.max(self.time_arrays[band])*new_fs)
            rsource = np.zeros(new_size) #Pad with zeros
            _rsource = signal.resample(self.source_signals[band],new_ressample_size)
            rsource[:len(_rsource)] = _rsource
            rreceivers = []
            for rsig in self.receiver_signals[band]:
                rsig_new = np.zeros(new_size) #Pad with zeros
                rsig = signal.resample(rsig,new_ressample_size)
                rsig_new[:len(rsig)] = rsig
                rreceivers.append(rsig_new)

            resampled_source_signals[band] = rsource
            resampled_receiver_signals[band] = rreceivers

        return resampled_source_signals,resampled_receiver_signals, new_size

    def calcFrequencyResponse(self, receiverIndex = 0): #TODO EXPLAIN
        """Return the frequency response given by the simulation for the specific receiver.

        :param receiverIndex: Index of the receiver to be used, defaults to 0
        :type receiverIndex: int, optional
        :return: Frequencies and the complex frequency response
        :rtype: Array; Array
        """
        # resampled_tspan = {}
        resampled_source_signals = {}
        resampled_receiver_signals = {}
        sources_fft = {}
        receivers_fft = {}

        #Resample all the signals from the source and the selected receiver 
        fs=16000
        resampled_source_signals,resampled_receiver_signals, new_size = self.__ressample_pad(fs)
        
        for band in self.source_signals:
            sources_fft[band] = fft.rfft(resampled_source_signals[band])
            receivers_fft[band] = fft.rfft(resampled_receiver_signals[band][receiverIndex])
        
        #Include the response for each band, only for its frequencies
        frequencies = []
        response = []
        for band in self.source_signals:
            _band_freqs = fft.rfftfreq(new_size,1/fs)
            idx_from = np.argmin(np.abs(_band_freqs-self.bands_freq[band][0]))
            idx_to = np.argmin(np.abs(_band_freqs-self.bands_freq[band][1]))
            band_response = receivers_fft[band]/sources_fft[band]
            frequencies.extend(_band_freqs[idx_from:idx_to])
            response.extend(band_response[idx_from:idx_to])

        frequencies = np.array(frequencies)
        response = np.array(response)

        return frequencies, response

    def assembleResponse(self,receiverIndex=0,fs=16000):
        """Using the simulations for each band, assembles a "impulse response"(limited to the simulated bands)

        :param receiverIndex: The index of the receiver whose response will be calculated, defaults to 0
        :type receiverIndex: int, optional
        :param fs: Sampling frequency of the output, defaults to 0
        :type fs: int, optional
        :return: The limited "impulse response"
        :rtype: Array
        """
        resampled_source_signals,resampled_receiver_signals,new_size = self.__ressample_pad(fs)

        assembled_source_signal = np.zeros(new_size)
        assembled_receiver_signal = np.zeros(new_size)

        for b in self.bands_freq.keys():
            if self.simulationsTime[b]==0: continue
            cutoff = self.bands_freq[b][0]
            ssig = resampled_source_signals[b]

            filter_low = signal.firwin(1001,cutoff,pass_zero='lowpass',fs=fs)
            filter_high = signal.firwin(1001,cutoff,pass_zero='highpass',fs=fs)
            assembled_source_signal = signal.filtfilt(filter_low,[1],assembled_source_signal) + signal.filtfilt(filter_high,[1],ssig)
            rsig = resampled_receiver_signals[b][receiverIndex]
            assembled_receiver_signal = signal.filtfilt(filter_low,[1],assembled_receiver_signal) + signal.filtfilt(filter_high,[1],rsig)
        
        #Filter the last band
        filter_low = signal.firwin(1001,self.bands_freq[b][1],pass_zero='lowpass',fs=fs)
        assembled_source_signal = signal.filtfilt(filter_low,[1],assembled_source_signal)
        assembled_receiver_signal = signal.filtfilt(filter_low,[1],assembled_receiver_signal)

        return Other.wiener_deconvolution(assembled_receiver_signal,assembled_source_signal,1)