"""
Toothy 

Purpose: Input data selection and processing pipeline

Authors: Amanda Schott, Kathleen Esfahany

Last updated: 2025-09-02
"""
# Import standard libraries
import sys
import os
import shutil
from pathlib import Path

import scipy.io as sio
import h5py
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtCore, QtGui
import time as timelib
import pickle
import probeinterface as prif

# Import custom modules
import QSS
import pyfx
import qparam
import ephys
import gui_items as gi
import data_processing as dp
from probe_handler import ProbeObjectPopup
import resources_rc
import resources_v2

#################################################
#################################################
######     Section 1: Worker Objects    #########
#################################################
#################################################

class ArrayReader(QtCore.QObject):
    """
    Reads data from NumPy and MATLAB files and returns data (and optional metadata) in dictionaries via emitted signals.
    - Checks for 2-dimensional arrays in the file ("data") and special keywords relating to sampling rate or units ("metadata").
    - Produces errors if files cannot be loaded or no 2-dimensional arrays are found.
    - Subsequent functions outside this class handle selection of data variable and metadata entry/confirmation.
    """
    # Define PyQt signals to communicate with the main thread
    progress_signal = QtCore.pyqtSignal(str)            # Progress signal used to update spinner text throughout loading process (later connected to RawRecordingSelectionPopup's SpinnerWindow's "report_progress_string" slot)
    data_signal = QtCore.pyqtSignal(str, dict, dict)    # Emits (filepath, data dictionary, metadata dictionary) at the end; later connected to "enter_array_metadata" which is the final storage/resting place of the data and metadata before it goes into the data extraction stage
    error_signal = QtCore.pyqtSignal(str, str)          # Emits (filepath, error message) if loading fails
    finished = QtCore.pyqtSignal()                      # Signal to indicate that the worker has finished its task
    
    def __init__(self, file_path):
        """
        Initialize path to input data file
        """
        super().__init__()
        self.file_path = file_path
    
    def run(self):
        """
        Load input data; emit data_signal with LFP data and metadata if successful
        """

        # Extract file name and extension to use in progress message
        file_path = self.file_path # Copy the file path
        file_name = os.path.basename(file_path) # Extract the file name ("basename" removes any preceding directory and only leaves the last component, i.e. recording.npy)
        file_ext = os.path.splitext(file_path)[-1][1:] # Get the file extension (e.g., .npy, .mat) ("splittext" splits a pathname into a pair (root, ext) where ext includes the leading dot) and index to remove the leading dot
        
        # Emit a progress signal indicating the start of file loading with the file extension in uppercase
        display_ext_dict = {'npy':'NumPy', 'mat':'MATLAB'}
        self.progress_signal.emit(f'Loading {display_ext_dict[file_ext]} file ...')
        
        error_msg = ''
        try:
            # Load NumPy file
            if file_ext == 'npy':
                file_data = np.load(file_path, allow_pickle=True)

            # Load MATLAB file
            elif file_ext == 'mat':
                file_data = sio.loadmat(file_path, squeeze_me=True) # squeeze_me=True removes dimensions of size 1 from the array 
            
            # If the loaded data is a NumPy n-dimensional array, add it to a dictionary with key 'data'
            if isinstance(file_data, np.ndarray):
                
                if file_data.ndim == 2: # Check if the data is 2-dimensional
                    data_dict = {"data" : np.array(file_data)}
                    meta = {'fs':None, 'units':None}
                else:
                    error_msg = 'Error: Data must be a 2-dimensional array.'

            # If the loaded data is a dictionary, parse it to extract data arrays and metadata
            elif isinstance(file_data, dict):
                self.progress_signal.emit('Parsing data dictionary...')

                try:  # get data array (required) and SR/unit metadata (optional)
                    data_dict, meta = self.read_data_from_dict(file_data)
                    if len(data_dict) == 0:
                        error_msg = f'Error: No 2-dimensional data arrays found in "{file_name}".'
                except: 
                    error_msg = f'Error: Could not parse data from "{file_name}".'
            else: 
                error_msg = 'Error: File must contain data array or dictionary.'

        except: 
            error_msg = f'Error: Unable to load "{file_name}".'
        
        # Emit signals to return data or error message
        if error_msg == '': # No errors, return data by emitting signals
            self.data_signal.emit(str(file_path), data_dict, meta)
            self.progress_signal.emit('Done!')
        else: # Errors occurred, emit error signal
            self.error_signal.emit(str(file_path), error_msg)

        # Emit a finished signal to indicate that the worker has completed its task
        self.finished.emit()

    def read_data_from_dict(self, input_data_dict):
        """
        Parse imported data dictionary to find keys with 2D array values (for LFP data) and keys for metadata (sampling rate/data SI units).
        """
        # Initialize two dictionaries: one for data arrays and one for metadata
        data_dict = {}
        metadata_dict = {'fs':None, 'units':None}

        # Define keywords for sampling rate and units
        fs_keys = ['fs', 'sr', 'sampling_rate', 'sample_rate', 'sampling_freq', 
                'sample_freq', 'sampling_frequency', 'sample_frequency']
        unit_keys = ['unit', 'units']

        # Iterate through the input dictionary to find 2D arrays and metadata
        for k, v in input_data_dict.items():

            # Skip MATLAB metadata fields (i.e. __header__, __version__, etc.)
            if k.startswith('__'): 
                continue

            # Look for sampling rate and units keys (note: if there are multiple matches, the last one will overwrite previous ones)
            if k.lower() in fs_keys:
                metadata_dict['fs'] = float(v)
            elif k.lower() in unit_keys:
                metadata_dict['units'] = str(v)

            else:
                if not hasattr(v, '__iter__'): # True for iterables (arrays, lists, strings), False for ints, floats
                    continue
                if len(v) == 0: # If empty iterable, skip
                    continue
                if np.array(v).ndim != 2: # If not 2D array, skip
                    continue
                data_dict[k] = v # Add key and value if array is 2D. Selection of which key occurs later in the pipeline.
        return data_dict, metadata_dict
        
        
class ExtractorWorker(QtCore.QObject):
    """ Returns spikeinterface extractor objects for raw recordings """
    progress_signal = QtCore.pyqtSignal(str)
    data_signal = QtCore.pyqtSignal(object, object)
    error_signal = QtCore.pyqtSignal(str, str)
    finished = QtCore.pyqtSignal()
    
    def __init__(self, fpath, **kwargs):
        """ Initialize recording filepath and raw data (if previously loaded) """
        super().__init__()
        self.fpath = fpath
        self.kwargs = {'data_array' : kwargs.get('data_array'),
                       'metadata' : kwargs.get('metadata', {}),
                       'electrical_series_path' : kwargs.get('electrical_series_path')}
    
    def run(self):
        """ Get recording extractor """
        data_type = dp.get_data_format(self.fpath)
        self.progress_signal.emit(f'Getting {data_type} extractor ...')
        try:  # get spikeinterface extractor for selected data type
            recording, time = dp.get_extractor(self.fpath, data_type, **self.kwargs)
        except Exception as e: # failed to load extractor object
            self.error_signal.emit(str(self.fpath), f'Error: {str(e)}')
        else:   # emit valid extractor
            self.data_signal.emit(recording, time) # Sends recording and time
            self.progress_signal.emit('Done!')
        finally:
            self.finished.emit()
    
    
class DataWorker(QtCore.QObject):
    """ 
    Handles data imports, preprocessing, and exports in processing pipeline
    """
    progress_signal = QtCore.pyqtSignal(str)
    data_signal = QtCore.pyqtSignal()
    error_signal = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()
    
    RECORDING = None    # spikeinterface recording extractor
    PROBE_GROUP = None  # probeinterface ProbeGroup object
    PARAMS = None       # parameter dictionary
    SAVE_DDIR = None    # location of processed data files
    
    def init_source(self, RECORDING, TIME, PROBE_GROUP, PARAMS, SAVE_DDIR=None, DOWNSAMPLE_NEEDED = False):
        """ Initialize data objects and parameter values """
        self.RECORDING = RECORDING
        self.TIME = TIME
        self.PROBE_GROUP = PROBE_GROUP
        self.PARAMS = PARAMS
        if SAVE_DDIR is None:  # auto-generate target directory
            raw_ddir = os.path.dirname(self.RECORDING.get_annotation('ppath'))
            SAVE_DDIR = str(Path(raw_ddir, pyfx.unique_fname(raw_ddir, 'toothy'))) # changed from processed_data
        self.SAVE_DDIR = SAVE_DDIR

        # If True, data will be downsampled. If False, the original data will be used directly.
        self.downsample_bool = DOWNSAMPLE_NEEDED
        
    def quit_thread(self, error_msg, ff=None, KW={}):
        """ Terminate pipeline upon data processing error """
        if ff is not None:
            ff.close()  # close HDF5 datasets
        if 'fid' in KW and KW['fid'] is not None:
            KW['fid'].close()  # close raw data file
        self.error_signal.emit(error_msg)
        self.finished.emit()
    
    def run(self, **kwargs):
        """
        Raw data processing pipeline.
        """
        # Get core data objects (recording, time)
        recording = kwargs.get('recording', self.RECORDING)
        time = kwargs.get("time", self.TIME)

        # Get metadata about the recording
        data_format = recording.get_annotation("data_format")
        recording_metadata_dict = dp.get_meta_from_recording(recording)
        recording_fs, recording_n_samples, TOTAL_CH = recording_metadata_dict['fs'], recording_metadata_dict['nsamples'], recording_metadata_dict['total_ch']

        # If there is no timestamp array, make one for internal use only. Users will only receive indices at the end.
        if type(time) != np.ndarray and time == None:
            # Note: np.linspace takes args (start, stop, n_samples), where start/stop can be floats and stop is inclusive. Example: linspace(0, 1.5, 3) = [0, 0.75, 1.5]
            total_est_time_s = recording_n_samples/recording_fs # Estimate the total time of the recording
            time = np.linspace(0, total_est_time_s, recording_n_samples)

        # Get the probeinterface "probe_group" object and set it for the spikeinterface "recording" object's probe group
        probe_group = kwargs.get('probe_group', self.PROBE_GROUP)
        n_probes = len(probe_group.probes)
        recording.set_probegroup(probe_group, in_place=True)

        # Get the processing parameters (set by user)
        PARAMS = kwargs.get('PARAMS', self.PARAMS)
        t_start_s, t_end_s = PARAMS['trange']   # Selected time points between which the data will be processed
        target_fs = PARAMS['lfp_fs']            # Target downsampling frequency (not used if self.downsample_bool == False)

        # Make the "toothy" folder for output
        toothy_output_dir   = kwargs.get('save_ddir', self.SAVE_DDIR)
        if not os.path.isdir(toothy_output_dir):
            os.mkdir(toothy_output_dir)

        # Set the "load_win" (loading window); default is 10-minute (600 seconds) chunks
        load_win = kwargs.get('load_win', 600)
    
        ##### Convert the selected start/end time into indices ##### 
        if t_end_s == -1: # If the whole recording is being analyzed, t_end_s is the final time (time[-1])
            t_end_s = time[-1]
        indices_in_range = np.nonzero((time >= t_start_s) & (time <= t_end_s))[0] # Indices where the time array is within the selected range
        
        i_start = indices_in_range[0] # i_start is the first in-range index
        i_start = int(min(i_start, recording_n_samples - recording_fs)) # i_start must be at least 1 second before the end of the recording
        
        if t_end_s == time[-1]:
            i_end = recording_n_samples # If using the entire recording, set i_end to the number of samples (e.g. +1 more than the final index)
        else:
            i_end = indices_in_range[-1] + 1 # Add one to i_end so when indexing with ":i_end" it includes the final sample
        i_end = int(min(max(i_end, i_start + recording_fs), recording_n_samples)) # i_end must be at least 1 second after the selected start, but not greater than the total samples
        
        # Count the total number of samples in the selection
        n_samples_input_selection = (i_end - i_start)

        # TODO REMOVE
        # iistart, iiend = dp.get_rec_bounds2(recording_n_samples, recording_fs, t_start_s, t_end_s)
        # n_samples_for_processing_initial = int(iiend-iistart)

        # Calculate size of processed dataset (if no downsampling, equal to the original; if downsampling, need to calculate)
        if self.downsample_bool:
            # Calculate the "downsampling factor", the muliplier between the target downsample rate and the recording's original sampling rate
            # Example A: If the recording is 2kHz and the target is 1kz, the ds_factor would be 2
            # Example B: If the recording is 10kHz and the target is 1.5kHz, the ds_factor would be 6.666...
            ds_factor = recording_fs/target_fs
            
            # Get the number of samples after downsampling
            # Example A: If the selection is 100k samples at 2kHz, the downsampled selection at 1kHz would be 50k samples
            # Example B: If the selection is 100k samples at 10kHz, the downsampled selection at 1.5kHz would be 15k samples
            n_samples_post_processing = int(n_samples_input_selection / ds_factor)
        else:
            # Don't downsample; the number of samples stays the same
            n_samples_post_processing = n_samples_input_selection

        global_dev_idx = probe_group.get_global_device_channel_indices()['device_channel_indices']
        
        
        ################################
        #####      I/O OBJECTS     #####
        ################################
        
        try:  # get IO object (fid, cont, or recording)
            KW = {**dp.get_raw_source_kwargs(recording), 'total_ch':TOTAL_CH}
        except Exception as e:
            self.quit_thread(f'Data Source Error: {str(e)}')
            return
        if KW['fid'] is not None and i_start > 0:
            KW['fid'].seek(int(i_start * TOTAL_CH * 4))
        
        # Create HDF5 datasets for LFP data from each probe
            # Note: This creates a file at the "save_ddir" with the name "DATA.hdf5"
            # Opens the file in "write mode" (parameter "w")
            # "track_order = True" ensures the order in which files are added is preserved
        data_hdf5_file = h5py.File(Path(toothy_output_dir, 'DATA.hdf5'), 'w', track_order =True)
        
        # HDF5 notes
            # Group: analogous to a folder/directory; can hold other groups/datasets
                # [file/group].create_group(<str_name>) to create a group
                # file[<str_path>] to access a group, file.keys() to see groups/datasets
            # Dataset: NumPy array-like structure that can store scalars, vectors, matrices, etc.; can be chunked/partially read without loading entirely into memory
                # [file/group].create_dataset(<str_name>, data = ...) to create a dataset
                # [file/group][<str_name>][:] to load the entire dataset
            # Attributes: metadata attached to groups or datasets
                # [file/group/dataset].attrs[<key>] = <value> to create an attribute
                # [^].attrs to access attrs
                # [^].[<key>] to access specific key/value pair

        # Create attributes of the file for the sampling rate of the input data (recording) and the intended downsampling sampling rate
        data_hdf5_file.attrs['fs'] = recording_fs # FS is the recording, LFP_fs is the downsampled
        data_hdf5_file.attrs['lfp_fs'] = target_fs

        ######################################################################################
        # Create HDF5 dataset for timestamps
        # If downsampling, first downsample the provided time, then add to the file
        if self.downsample_bool:
            interpolation_x_original = np.linspace(i_start, i_end, n_samples_input_selection) # Original indices
            interpolation_y_original = time[i_start:i_end] # Original time
            interpolation_x_new = np.linspace(i_start, i_end, n_samples_post_processing) # New indices (not used as indices; have floats)
            interpolation_y_new = np.interp(interpolation_x_new, interpolation_x_original, interpolation_y_original) # New time
            time_downsample = interpolation_y_new   
            lfp_time = data_hdf5_file.create_dataset("lfp_time", data = time_downsample)
                
        # If not downsampling, add the provided time directly to the file
        else:
            lfp_time = data_hdf5_file.create_dataset("lfp_time", data = time[i_start:i_end])

        ######################################################################################

        # Make HDF5 datasets with the channel mapping for each probe
        ichannels, h5_datasets = [], []
        for i_prb in range(n_probes):
            # Create a group named with the probe index
            h5_probe_group = data_hdf5_file.create_group(str(i_prb), track_order = True)

            # Create an "LFP" group inside the probe group
            h5_probe_lfp_group = h5_probe_group.create_group('LFP', track_order = True)

            # Get the indices of channels on the probe
            idx = np.where(recording.get_channel_groups() == i_prb)[0]

            # Index the "global" to get the global indices of this probe's channels
            ichan = global_dev_idx[idx]

            # Create a dataset in the probe/LFP group called "raw" with shape channels x samples
            dset = h5_probe_lfp_group.create_dataset('raw', (len(ichan), n_samples_post_processing), 
                                              dtype='float32')
            
            ichannels.append(ichan)
            h5_datasets.append(dset)

        ##################################
        #####     DATA EXTRACTION    #####
        ##################################
        lfp_time_resampled = np.zeros(n_samples_post_processing)

        # Extract and downsample LFP in chunks (default of 10 min chunks)
        if self.downsample_bool:
            chunk_target_fs = target_fs
        else:
            chunk_target_fs = None

        #iichunk is the # indices in the original recording per chunk, ichunk is the # indices in the processed recording per chunk
        chunkfunc, (iichunk, ichunk) = dp.get_chunkfunc(
            load_win,
            recording_fs, 
            recording_n_samples,
            target_fs = chunk_target_fs,
            i_start=i_start,
            i_end=i_end
        )
  
        count = 0
        while True:
            (ii,jj), (aa,bb), txt = chunkfunc(count)

            # First, are the indices the same going in and coming out?
            (rec_chunk_start_i, rec_chunk_end_i) = ii, jj
            proc_chunk_start_i, proc_chunk_end_i = aa, bb
           
            
            t_chunk = time[ii:jj]   # section of time from ORIGINAL TIME
            n_in    = len(t_chunk)      # samples going in
            n_out   = bb - aa           # samples coming out
            if n_in >= 2 and n_out > 0:
                # Interpolate time w.r.t. sample index → preserves local irregularities
                x_in  = np.arange(n_in, dtype=np.float64) # Make an integer array of indices spanning the original time array
                x_out = np.linspace(0, n_in - 1, n_out, dtype=np.float64) # Make a linearly spaced array of indices spanning the same range, but in fewer samples
                
                # Downsample time using interpolation over the indices
                time_downsampled = np.interp(x_out, x_in, t_chunk)

                p = count * ichunk
                lfp_time_resampled[p : p + n_out] = time_downsampled

            
            self.progress_signal.emit(txt)
            
     
            for ichan, dset in zip(ichannels, h5_datasets):
                try:  # read in dataset chunk (scaled to mV)
                    snip = dp.load_chunk(data_format, ii=ii, jj=jj, ichan=ichan, **KW)
                except Exception as e:
                    self.quit_thread(f'I/O Error: {str(e)}', ff=data_hdf5_file, KW=KW)
                    return
                
                if self.downsample_bool:
                    try:  # downsample LFP signals
                        snip_dn = dp.resample_chunk(snip, bb-aa)
                        p = count*ichunk
                        dset[:, p:p+(bb-aa)] = snip_dn
                    except Exception as e:
                        self.quit_thread(f'Downsampling Error: {str(e)}', ff=data_hdf5_file, KW=KW)
                        return
                else:
                    p = count * ichunk
                    dset[:, p:p+(bb-aa)] = snip

            if jj >= i_end:
                break
            count += 1

        if KW['fid'] is not None:
            KW['fid'].close()
        
        ####################################
        #####    BANDPASS FILTERING    #####
        ####################################
        
        for iprb, probe in enumerate(probe_group.probes):
            hdr = f'ANALYZING PROBE {iprb+1} / {n_probes}<br>'

            shank_ids = np.array(probe.shank_ids, dtype='int')

            # Get the raw/unfiltered LFP data stored earlier
            lfp_unfiltered = data_hdf5_file[str(iprb)]['LFP']['raw']
            
            # Get the number of channels
            n_channels = len(lfp_unfiltered)

            # Bandpass filter LFP and get mean amplitude per channel

            # Initialize a dictionary to hold filtered signals
            LFP_dict = {}

            # For each band of interest, initialize a numpy array of ones (the same size as the )
            for k in ['theta', 'slow_gamma', 'fast_gamma', 'ds', 'swr']:
                LFP_dict[k] = np.ones(lfp_unfiltered.shape, dtype='float32')
            
            # Emit progress message
            self.progress_signal.emit(hdr + '<br>Bandpass filtering signals ...')

            filter_kwargs = {'lfp_fs': target_fs, 'axis': 1}
            try:
                # Split the array into sections
                arr_list = np.array_split(lfp_unfiltered, int(np.ceil(n_samples_post_processing / ichunk)), axis=1)
                p = 0

                # Go section by section and filter the data - needs to happen, downsampling or not.
                for i, yarr in enumerate(arr_list):
                    q = p + yarr.shape[1]
                    LFP_dict['theta'][:,p:q] = pyfx.butter_bandpass_filter(yarr, *PARAMS['theta'], **filter_kwargs)
                    LFP_dict['slow_gamma'][:,p:q] = pyfx.butter_bandpass_filter(yarr, *PARAMS['slow_gamma'], **filter_kwargs)
                    LFP_dict['fast_gamma'][:,p:q] = pyfx.butter_bandpass_filter(yarr, *PARAMS['fast_gamma'], **filter_kwargs)
                    LFP_dict['swr'][:,p:q] = pyfx.butter_bandpass_filter(yarr, *PARAMS['swr_freq'], **filter_kwargs)
                    LFP_dict['ds'][:,p:q] = pyfx.butter_bandpass_filter(yarr, *PARAMS['ds_freq'], **filter_kwargs)
                    p += yarr.shape[1]
                std_dict = {}
                for k,arr in LFP_dict.items():
                    std_dict[k] = np.std(arr, axis=1)
                    std_dict[f'norm_{k}'] = pyfx.Normalize(std_dict[k])
            except Exception as e:
                self.quit_thread(f'Filtering Error: {str(e)}', ff=data_hdf5_file)
                return
            
            # Store the filtered signal for the dentate spike (DS) and sharp-wave ripple (SPW-R) bands for easy adjustment of event-detection parameters in later steps
            data_hdf5_file[str(iprb)]['LFP']['ds'] = np.array(LFP_dict['ds'])
            data_hdf5_file[str(iprb)]['LFP']['swr'] = np.array(LFP_dict['swr'])
            
            STD = pd.DataFrame(std_dict)
            STD.to_hdf(data_hdf5_file.filename, key=f'/{iprb}/STD')
            
            
            ####################################
            #####      EVENT DETECTION     #####
            ####################################
            
            # Initial ripple and DS detection for each channel
            progress_txt = hdr + f'<br>Detecting DSs and ripples on channel<br>%s / {n_channels} ...'
            SWR_DF, DS_DF, SWR_THRES, DS_THRES = None, None, None, None
            for i,(swr_d, ds_d) in enumerate(zip(LFP_dict['swr'], LFP_dict['ds'])):
                self.progress_signal.emit(progress_txt % (i+1))
                try:  # detect sharp-wave ripples
                    SWR_DF, SWR_THRES = dp.detect_channel('swr', i, swr_d, lfp_time[:],
                                                       DF=SWR_DF, THRES=SWR_THRES, 
                                                       pprint=False, **PARAMS)
                except Exception as e:
                    self.quit_thread(f'Ripple Detection Error: {str(e)}', ff=data_hdf5_file)
                    return
                try:  # detect DSs
                    DS_DF, DS_THRES = dp.detect_channel('ds', i, ds_d, lfp_time[:], 
                                                        DF=DS_DF, THRES=DS_THRES, 
                                                        pprint=False, **PARAMS)
                except Exception as e:
                    self.quit_thread(f'DS Detection Error: {str(e)}', ff=data_hdf5_file)
                    return
            
            # add status columns for later curation
            for DF in [SWR_DF, DS_DF]:
                if DF.size == 0: DF.loc[0] = np.nan
                DF['status'] = 1 # 1=auto-detected; 2=added by user; -1=removed by user
                DF['is_valid'] = 1 # valid events are either auto-detected and not user-removed OR user-added
                tups = [(ch, shank_ids[ch]) for ch in np.unique(DF.index.values)]
                DF['shank'] = 0
                for ch,shkID in tups:
                    DF.loc[ch,'shank'] = shkID
            SWR_DF.to_hdf(data_hdf5_file.filename, key=f'/{iprb}/ALL_SWR')
            DS_DF.to_hdf(data_hdf5_file.filename, key=f'/{iprb}/ALL_DS')
            
            for ek,THRES in [('swr',SWR_THRES),('ds',DS_THRES)]:
                THRES_DF = pd.DataFrame(THRES).T  # save threshold magnitudes
                THRES_DF.to_hdf(data_hdf5_file.filename, key=f'/{iprb}/{ek.upper()}_THRES')
            
            # initialize noise train
            data_hdf5_file[str(iprb)]['NOISE'] = np.zeros(n_channels, dtype='int')
            
        ##########################################
        #####    SAVE PROBE/PARAM SETTINGS   #####
        ##########################################
            
        # initialize event channel dictionary with [None,None,None]
        _ = ephys.init_event_channels(toothy_output_dir, probes=probe_group.probes, psave=True)
        
        # Save params and info file in recording folder
        param_path = Path(toothy_output_dir, pyfx.unique_fname(toothy_output_dir, 'params.pkl'))
        with open(Path(param_path), 'wb') as f:
            pickle.dump(PARAMS, f)
        
        # write probe group to file
        probegroup_path = Path(toothy_output_dir, pyfx.unique_fname(toothy_output_dir, 'probe_group'))
        prif.write_probeinterface(probegroup_path, probe_group)

        # Close the file
        data_hdf5_file.close()
        self.data_signal.emit()
        self.progress_signal.emit('Done!')
        timelib.sleep(1)
        self.finished.emit()

##############################################################################
##############################################################################
################                                              ################
################                RAW DATA MODULES              ################
################                                              ################
##############################################################################
##############################################################################
        
    
class RawRecordingSelectionWidget(gi.FileSelectionWidget):#✅
    """
    Selection and validation of an input data file path.
    Subclass of FileSelectionWidget.

    - "select_filepath" launches a file dialog (restricted to supported file types)
    - "update_filepath" validates the file is a supported type, updates the displayed file path on the GUI, and emits a status signal (if True, drives forward the next step of file extraction)
    - "enter_array_metadata" is run after "update_filepath" emits "read_array_signal" (which calls "load_array_worker", which runs ArrayReader, which emits to "load_array_finished_slot", which calls this)
    """
    
    # Create a signal for broadcasting if an array (NumPy or MATLAB file type) is selected
    read_array_signal = QtCore.pyqtSignal(str)

    def __init__(self, title='', parent=None):
        super().__init__(title=title, parent=parent)
        self.icon_btn.hide() # Hide icon button
    
    def select_filepath(self):
        """
        Launch file dialog for raw recording selection, filter unsupported extensions.
        """
        init_ddir = self.get_init_ddir()
        supported_extensions = [x[1] for x in dp.supported_formats.values()]

        # Create a modal QFileDialog for the user to select a file; return the file path (as type str)
        file_path = ephys.select_raw_recording_file(supported_extensions, init_ddir, parent=self)

        # If a file is not selected above (i.e. the file dialog is closed via hitting "Cancel" or the X button), file_path will be an empty string
        # If a file is selected (non-empty strings evaluate as True), update the data array, metadata, and filepath
        if file_path:
            # Initialize variables for the data array and metadata dictionary
            self.data_array, self.meta = None, {}

            # Send the file path string to "update_filepath" to validate it and (if an array) extract the array
            # "update_filepath" also emits a signal ("signal") connected to a slot "ppath_updated" that triggers the next step of the overall GUI flow
            self.update_filepath(file_path)

        # If a file is not selected, we don't update the data_array/metadata or filepath
        
    def update_filepath(self, ppath, x=None):
        """
        Handle selection of a filepath. Overwrites function in parent class to add handling for NumPy and MATLAB arrays.
        """
        
        if x is not None:
            validation_status = x
            
        if x is None: # (default)
            validation_status = self.validate_path(ppath) # True, False, or "NPY or MAT"

        if validation_status == "NPY or MAT":   # User selected a NPY or MAT file
            self.read_array_signal.emit(ppath)  # Emit signal for reading a NumPy/MATLAB array; connected to "load_array_worker" slot which begins ArrayWorker
            return                              # Exit the function; downstream functions called by "read_array_signal" signal will eventually call "update_filepath" again

        else:
            self.le.setText(ppath)                  # Update QLineEdit widget defined in parent class 
            self.update_status(validation_status)   # Use parent class "update_status" to change the QLineEdit box edge from red to green
            self.signal.emit(self.VALID_PPATH)      # Defined in parent class; most important signal emitted by this class; True or False to indicate if a valid file has been selected 
    
    def enter_array_metadata(self, ppath, data_array, metadata_dict):
        """
        Prompt user to enter contextual metadata for NPY/MAT files.
        If file contained special keys, those values will be populated.
        Important: assigns data_array and metadata_dict to class variables, which are referenced by the next step (extraction).
        """
        # Launch dialog windows
        dialog_window = RawArrayPopup(data_array.shape, **metadata_dict, filename=os.path.basename(ppath))

        # If dialog window is Assign values to class variables
        if dialog_window.exec():
            self.data_array = np.array(data_array)
            self.metadata_dict = {
                'nch' : dialog_window.nch, 
                'nts' : dialog_window.nts,
                'fs' : dialog_window.fs_w.value(),
                'units' : dialog_window.units_w.currentText()
            }

            # Finally, call "update_filepath" with x=True to move the workflow on to the next part (extraction)
            # Concludes the journey from file selection and array processing for NPY/MAT files
            self.update_filepath(ppath, True)
                
    def validate_path(self, path):
        """
        Check if selected data path exists and is a supported file type.
        This is largely achieved prior to this point by using a QFileDialog filered to the supported file types, but worth having as a backup.

        Returns:
            True if the path exists and is a supported data format.
            False if the path doesn't exist, is a directory, or is an unsupported file format.
            "NPY or MAT" if the path points to a NumPy or MATLAB file.
        """
        # Check if the path exists and is a file (not a directory)
        if not os.path.exists(path) or os.path.isdir(path):
            return False
        
        # "get_data_format" either returns a string (one of the six supported formats), or raises an exception
        try: 
            data_format = dp.get_data_format(path)
        except:
            return False
        
        # Special return for NumPy or MATLAB files to prompt further array processing
        if data_format in ['NPY','MAT']:
            return "NPY or MAT"
    
        # For all other supported formats, just return True
        else:
            return True
    
        
class ProbeRow(QtWidgets.QWidget):
    """ Interactive widget representation of a probe object """
    
    def __init__(self, probe, nrows, start_row, mode):
        super().__init__()
        self.probe = probe
        self.nch = probe.get_contact_count()
        self.nrows = nrows
        self.div = int(self.nrows / self.nch)
        
        self.gen_layout()
        self.get_rows(start_row, mode)
        
    def gen_layout(self):
        """ Set up layout """
        # selection button
        self.btn = QtWidgets.QPushButton()
        self.btn.setCheckable(True)
        self.btn.setChecked(True)
        self.btn.setFixedSize(20,20)
        self.btn.setFlat(True)
        self.btn.setStyleSheet('QPushButton'
                               '{border : none;'
                               'image : url(:/icons/white_circle.png);'
                               'outline : none;}'
                               
                               'QPushButton:checked'
                               '{image : url(:/icons/black_circle.png);}')
        # probe info labels
        self.glabel = QtWidgets.QLabel()
        self.glabel_fmt = '<b>{a}</b><br>channels {b}'
        labels = QtWidgets.QWidget()
        self.glabel.setStyleSheet('QLabel {'
                                  'background-color:white;'
                                  'border:1px solid gray;'
                                  'padding:5px 10px;}')
        label_lay = QtWidgets.QVBoxLayout(labels)
        self.qlabel = QtWidgets.QLabel(self.probe.name)
        self.ch_label = QtWidgets.QLabel()
        label_lay.addWidget(self.qlabel)
        label_lay.addWidget(self.ch_label)
        
        # action buttons - delete & copy implemented
        self.bbox = QtWidgets.QWidget()
        policy = self.bbox.sizePolicy()
        policy.setRetainSizeWhenHidden(True)
        self.bbox.setSizePolicy(policy)
        bbox = QtWidgets.QGridLayout(self.bbox)
        bbox.setContentsMargins(0,0,0,0)
        bbox.setSpacing(0)
        toolbtns = [QtWidgets.QToolButton(), QtWidgets.QToolButton(), 
                    QtWidgets.QToolButton(), QtWidgets.QToolButton()]
        self.copy_btn, self.delete_btn, self.edit_btn, self.save_btn = toolbtns
        
        self.delete_btn.setIcon(QtGui.QIcon(":/icons/trash.png"))
        self.copy_btn.setIcon(QtGui.QIcon(":/icons/copy.png"))
        self.edit_btn.setIcon(QtGui.QIcon(":/icons/edit.png"))
        self.save_btn.setIcon(QtGui.QIcon(":/icons/save.png"))
        for btn in toolbtns:
            btn.setIconSize(QtCore.QSize(20,20))
            btn.setAutoRaise(True)
        bbox.addWidget(self.copy_btn, 0, 0)
        bbox.addWidget(self.delete_btn, 0, 1)
        #bbox.addWidget(self.edit_btn, 1, 0)
        #bbox.addWidget(self.save_btn, 1, 1)
        self.btn.toggled.connect(lambda chk: self.bbox.setVisible(chk))
        
        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.setContentsMargins(0,0,0,0)
        self.layout.addWidget(self.btn, stretch=0)
        self.layout.addWidget(self.glabel, stretch=2)
        #self.layout.addWidget(labels)
        self.layout.addWidget(self.bbox, stretch=0)
        #self.layout.addWidget(self.qlabel)
        #self.layout.addWidget(self.ch_label)
        
    def get_rows(self, start_row, mode):
        """ Map probe channels to subset of data rows """
        self.MODE = mode
        if self.MODE == 0:    # M consecutive indices from starting point
            self.ROWS = np.arange(start_row, start_row+self.nch)
            txt = f'{start_row}:{start_row+self.nch}'
        elif self.MODE == 1:  # M indices distributed evenly across M*N total rows
            self.ROWS = np.arange(0, self.nch*self.div, self.div) + start_row
            txt = f'{start_row}::{self.div}::{self.nch*self.div-self.div+start_row+1}'
        self.glabel.setText(self.glabel_fmt.format(a=self.probe.name, b=txt))


class ProbeAssignmentWidget(QtWidgets.QWidget):
    """ Loads, creates and assigns probe objects to ephys data arrays """
    check_signal = QtCore.pyqtSignal()
    MODE = 0  # probes assigned to "contiguous" (0) or "alternating" (1) rows
    
    def __init__(self, nrows):
        super().__init__()
        self.nrows = nrows
        self.remaining_rows = np.arange(self.nrows)
        
        self.gen_layout()
        self.connect_signals()
        self.probe_done_btn = QtWidgets.QPushButton('i have written new text') # PROCESS DATA
        self.probe_done_btn.setEnabled(False)
        
        # KHERE KHERE
    
    def gen_layout(self):
        """ Set up layout """
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0,0,0,0)
        
        # title and status button
        self.row0 = QtWidgets.QHBoxLayout()
        self.row0.setContentsMargins(0,0,0,0)
        self.row0.setSpacing(3)
        # self.prb_icon_btn = gi.StatusIcon(init_state=0)
        probe_lbl = QtWidgets.QLabel('<b>Probe(s)</b>')
        # self.row0.addWidget(self.prb_icon_btn)
        self.row0.addWidget(probe_lbl)
        self.row0.addStretch()
        # load/create buttons
        self.load_prb = QtWidgets.QPushButton('Load')
        self.create_prb = QtWidgets.QPushButton('Create')
        self.row0.addWidget(self.load_prb)
        self.row0.addWidget(self.create_prb)
        # container for probe objects/rows 
        self.data_assign_df = pd.DataFrame({'Row':np.arange(self.nrows), 'Probe(s)':''})
        self.probe_bgrp = QtWidgets.QButtonGroup()
        self.qframe = QtWidgets.QFrame()
        self.qframe.setFrameShape(QtWidgets.QFrame.Panel)
        self.qframe.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.qframe.setLineWidth(3)
        self.qframe.setMidLineWidth(3)
        qframe_layout = QtWidgets.QVBoxLayout(self.qframe)
        qframe_layout.setSpacing(10)
        self.qlayout = QtWidgets.QVBoxLayout()  # probe row container
        qframe_layout.addLayout(self.qlayout, stretch=2)
        #qframe_layout.addLayout(hbox, stretch=0)
        
        # display data dimensions vs probe geometry
        self.row00 = QtWidgets.QHBoxLayout()
        self.row00.setContentsMargins(0,0,0,0)
        self.row00.setSpacing(3)
        self.view_assignments_btn = QtWidgets.QPushButton('View')
        self.row00.addStretch()
        self.row00.addWidget(self.view_assignments_btn)
        data_panel = QtWidgets.QFrame()
        data_panel.setFrameShape(QtWidgets.QFrame.Panel)
        data_panel.setFrameShadow(QtWidgets.QFrame.Sunken)
        data_lay = QtWidgets.QVBoxLayout(data_panel)
        #self.data_lbl = QtWidgets.QLabel(f'DATA: {self.nrows} channels')
        self.data_txt0 = f'{self.nrows} channels'
        self.data_txt_fmt = (f'<code>{self.nrows} data rows<br>'
                             '<font color="%s">%s channels</font></code>')
        self.data_lbl = QtWidgets.QLabel(self.data_txt_fmt % ('red', 0))
        self.data_lbl.setStyleSheet('QLabel {'
                                    'background-color:white;'
                                    'border:1px solid gray;'
                                    'padding:10px;'
                                    '}')
        # assignment mode (blocks vs interlacing)
        assign_vlay = QtWidgets.QVBoxLayout()
        assign_vlay.setSpacing(0)
        assign_lbl = QtWidgets.QLabel('<u>Index Mode</u>')
        self.block_radio = QtWidgets.QRadioButton('Contiguous rows')
        self.block_radio.setChecked(True)
        self.inter_radio = QtWidgets.QRadioButton('Alternating rows')
        assign_vlay.addWidget(assign_lbl)
        assign_vlay.addWidget(self.block_radio)
        assign_vlay.addWidget(self.inter_radio)
        data_lay.addWidget(self.data_lbl)
        data_lay.addStretch()
        data_lay.addLayout(assign_vlay)
        #data_lay.addWidget(self.view_assignments_btn)
        self.vlay0 = QtWidgets.QVBoxLayout()
        self.vlay0.addLayout(self.row0)
        self.vlay0.addWidget(self.qframe)
        self.vlay1 = QtWidgets.QVBoxLayout()
        self.vlay1.addLayout(self.row00)
        self.vlay1.addWidget(data_panel)
        self.hlay = QtWidgets.QHBoxLayout()
        self.hlay.addLayout(self.vlay0, stretch=3)
        self.hlay.addLayout(self.vlay1, stretch=1)
        
        self.layout.addLayout(self.hlay)
        
    def connect_signals(self):
        """ Connect GUI inputs """
        self.load_prb.clicked.connect(self.load_probe_from_file)
        self.create_prb.clicked.connect(self.design_probe)
        self.view_assignments_btn.clicked.connect(self.view_data_assignments)
        self.block_radio.toggled.connect(self.switch_index_mode)
    
    def view_data_assignments(self):
        """ Show probe(s) assigned to each data signal """
        tbl = gi.TableWidget(self.data_assign_df)
        dlg = gi.Popup(widgets=[tbl], title='Data Assignments', parent=self)
        dlg.exec()
        
    def switch_index_mode(self, chk):
        """ Assign probes to contiguous blocks or distributed rows of data """
        self.MODE = int(not chk)  # if block btn is checked, mode = 0
        items = pyfx.layout_items(self.qlayout)
        self.remaining_rows = np.arange(self.nrows)
        start_row = 0
        for i,item in enumerate(items):
            item.get_rows(start_row, self.MODE)
            if self.MODE == 0:
                start_row = item.ROWS[-1] + 1
            elif self.MODE == 1:
                start_row += 1
            self.remaining_rows = np.setdiff1d(self.remaining_rows, item.ROWS)
        self.check_assignments()
        
    def add_probe_row(self, probe):
        """ Add new probe to collection """
        nch = probe.get_contact_count()
        # require enough remaining rows to assign probe channels
        try:
            assert nch <= len(self.remaining_rows)
        except AssertionError:
            msg = f'Cannot map {nch}-channel probe to {len(self.remaining_rows)} remaining data rows'
            gi.MsgboxError(msg, parent=self).exec()
            return
        
        if self.MODE == 1:
            lens = [item.nch for item in pyfx.layout_items(self.qlayout)] + [nch]
            try:
                assert len(np.unique(lens)) < 2  # alternate indexing requires all same-size probes
            except AssertionError:
                msg = 'Alternate indexing requires all probes to be the same size'
                gi.MsgboxError(msg, parent=self).exec()
                return
        # get start row for probe based on prior probe assignment
        start_row = 0
        if self.qlayout.count() > 0:
            prev_rows = self.qlayout.itemAt(self.qlayout.count()-1).widget().ROWS
            start_row = pyfx.Edges(prev_rows)[1-self.MODE] + 1
        probe_row = ProbeRow(probe, self.nrows, start_row, self.MODE)
        self.probe_bgrp.addButton(probe_row.btn)
        probe_row.copy_btn.clicked.connect(lambda: self.copy_probe_row(probe_row))
        probe_row.delete_btn.clicked.connect(lambda: self.del_probe_row(probe_row))
        # probe_row.edit_btn.clicked.connect(lambda: self.edit_probe_row(probe_row))
        # probe_row.save_btn.clicked.connect(lambda: self.save_probe_row(probe_row))
        
        self.qlayout.addWidget(probe_row)
        self.remaining_rows = np.setdiff1d(self.remaining_rows, probe_row.ROWS)
        self.check_assignments()
    
    def del_probe_row(self, probe_row):
        """ Remove assigned probe from collection """
        # position of probe object to be deleted
        idx = pyfx.layout_items(self.qlayout).index(probe_row)
        
        self.probe_bgrp.removeButton(probe_row.btn)
        self.qlayout.removeWidget(probe_row)
        probe_row.setParent(None)
        
        self.remaining_rows = np.arange(self.nrows)
        items = pyfx.layout_items(self.qlayout)
        for i,item in enumerate(items):
            if i==max(idx-1,0): item.btn.setChecked(True) # auto-check row above deleted object
            if i < idx: continue  # probes above deleted object do not change assignment
            # update rows
            if i == 0 : start_row = 0
            else      : start_row = pyfx.Edges(items[i-1].ROWS)[1-self.MODE] + 1
            item.get_rows(start_row, self.MODE)
            self.remaining_rows = np.setdiff1d(self.remaining_rows, item.ROWS)
        self.check_assignments()
    
    def copy_probe_row(self, probe_row):
        """ Duplicate an assigned probe """
        # copy probe configuration to new probe object, add as row
        orig_probe = probe_row.probe
        new_probe = orig_probe.copy()
        new_probe.annotate(**dict(orig_probe.annotations))
        new_probe.set_shank_ids(np.array(orig_probe.shank_ids))
        new_probe.set_contact_ids(np.array(orig_probe.contact_ids))
        new_probe.set_device_channel_indices(np.array(orig_probe.device_channel_indices))
        self.add_probe_row(new_probe)
    
    def load_probe_from_file(self):
        """ Load probe object from saved file, add to collection """
        probe,_ = ephys.select_load_probe_file(parent=self)
        if probe is None: return
        self.add_probe_row(probe)
    
    def design_probe(self):
        """ Launch probe designer popup"""
        probe_popup = ProbeObjectPopup()
        probe_popup.setModal(True)
        probe_popup.accept_btn.setVisible(True)
        probe_popup.accept_btn.setText('CHOOSE PROBE')
        res = probe_popup.exec()
        if res:
            probe = probe_popup.probe_widget.probe
            self.add_probe_row(probe)

    def check_assignments(self):
        """ Check for valid assignment upon probe addition/deletion/reindexing """
        # list probe(s) associated with each data row
        items = pyfx.layout_items(self.qlayout)
        
        # allow different-size probes in block mode, but disable in alternate mode
        x = len(np.unique([item.nch for item in items])) < 2
        self.inter_radio.setEnabled(x)
        
        ALL_ROWS = {}
        for k in np.arange(self.nrows):
            ALL_ROWS[k] = [i for i,item in enumerate(items) if k in item.ROWS]
            
        # probe config is valid IF each row is matched with exactly 1 probe
        matches = [len(x)==1 for x in ALL_ROWS.values()]
        nvalid = len(np.nonzero(matches)[0])
        is_valid = bool(nvalid == self.nrows)
        
        probe_strings = [', '.join(np.array(x, dtype=str)) for x in ALL_ROWS.values()]
        self.data_assign_df = pd.DataFrame({'Row':ALL_ROWS.keys(), # assignment dataframe
                                            'Probe(s)':probe_strings})
        self.probe_done_btn.setEnabled(is_valid)  # require valid config for next step
        self.data_lbl.setText(self.data_txt_fmt % (['red','green'][int(is_valid)], nvalid))
        self.check_signal.emit()
        

class RawArrayPopup(QtWidgets.QDialog):
    """ Interface for user-provided metadata for NPY/MAT recordings """
    supported_units = ['uV', 'mV', 'V', 'kV']
    
    def __init__(self, data_shape, fs=None, units=None, filename='', parent=None):
        super().__init__(parent)
        self.setWindowFlag(QtCore.Qt.WindowMaximizeButtonHint)
        assert len(data_shape) == 2, 'Data array must be 2-dimensional.'
        
        self.gen_layout(data_shape, fs, units)
        self.connect_signals()
        self.setWindowTitle(filename)
        
    def gen_layout(self, data_shape, fs, units):
        """ Set up layout """
        nrows,ncols = data_shape
        if nrows > ncols: # rows == samples
            self.nts, self.nch = data_shape
            row_lbl, col_lbl = ['samples', 'channels']
        else: # rows == channels
            self.nch, self.nts = data_shape
            row_lbl, col_lbl = ['channels', 'samples']
        txt = f'<code>{nrows} {row_lbl} x {ncols} {col_lbl}</code>'
        qlbl = QtWidgets.QLabel(txt)
        qlbl.setStyleSheet('QLabel {background-color : white; padding : 4px;}')
        qtitle = QtWidgets.QLabel('Data dimensions:')
        self.lbl_row = pyfx.get_widget_container('h', qtitle, qlbl, stretch_factors=[2,0],
                                                 widget='widget')
        # create sampling rate and unit inputs
        self.fs_w = gi.LabeledSpinbox('Sampling rate', double=True, minimum=1, 
                                      maximum=9999999999, suffix=' Hz')
        self.dur_w = gi.LabeledSpinbox('Duration', double=True, minimum=0.0001,
                                       maximum=9999999999, suffix=' s', decimals=4)
        self.dur_w.qw.setReadOnly(True)
        if fs is not None:
            self.fs_w.setValue(fs)
            self.dur_w.setValue(self.nts / fs)
        self.units_w = gi.LabeledCombobox('Units')
        self.units_w.addItems(self.supported_units)
        if units in self.supported_units:
            self.units_w.setCurrentText(units)
        
        self.layout = QtWidgets.QVBoxLayout(self)
        hlay = pyfx.get_widget_container('h', self.fs_w, self.dur_w, self.units_w)
        
        # create action buttons
        self.bbox = QtWidgets.QWidget()
        bbox_lay = QtWidgets.QHBoxLayout(self.bbox)
        self.confirm_meta_btn = QtWidgets.QPushButton('Continue')
        self.close_btn = QtWidgets.QPushButton('Cancel')
        bbox_lay.addWidget(self.close_btn)
        bbox_lay.addWidget(self.confirm_meta_btn)
        
        self.layout.addWidget(self.lbl_row)
        self.layout.addWidget(pyfx.DividerLine(lw=1, mlw=1))
        self.layout.addLayout(hlay)
        self.layout.addWidget(self.bbox)
    
    def connect_signals(self):
        """ Connect GUI inputs """
        self.fs_w.qw.valueChanged.connect(lambda x: self.update_fs_dur(x, 0))
        self.dur_w.qw.valueChanged.connect(lambda x: self.update_fs_dur(x, 1))
        
        self.confirm_meta_btn.clicked.connect(self.accept)
        self.close_btn.clicked.connect(self.reject)
        
    def update_fs_dur(self, val, mode):
        """ Update duration from sampling rate (mode=0) or vice versa (mode=1) """
        #nts = self.data.shape[self.cols_w.currentIndex()]
        if mode==0:
            # calculate recording duration from sampling rate
            pyfx.stealthy(self.dur_w.qw, self.nts / self.fs_w.value())
        elif mode==1:
            pyfx.stealthy(self.fs_w.qw, self.nts / self.dur_w.value())


class ProcessingOptionsWidget(QtWidgets.QFrame):
    """
    A widget for a section with radio buttons with different processing approaches, with one recommended based on the sampling rate of the input data.
    Emits a "selection_changed" signal whenever the choice or custom rate changes.
    """

    # Define a custom signal for when the user's selection changes
    selection_changed = QtCore.pyqtSignal(dict)

    def __init__(self, parent = None):
        super().__init__(parent)
        self.input_fs = float(10)

        # Set the stylesheet for the elements in this section
        self.setStyleSheet("""
            QFrame {
                border: 0px solid #D8D8D8;
            }
            QRadioButton {
                spacing: 8px; /* spacing between the button and the adjacent text */
            }
            QRadioButton::indicator {
                width: 16px; height: 16px; /* size of the button */
            }
            .optionTitle {
                color: black;
                border: 0px;
            }
            .hint {
                color: #333333;
                border: 1px solid gray;
                border-radius: 5px;
                padding: 10px 10px;
            }
            .pill {
                border: 1px solid #006400;
                color: #006400;
                background: #E6F7E6;
                border-radius: 9px; /* make rounded */
                padding: 1px 6px; /* top/bottom, left/right*/
                font-weight: bold;
                margin-left: 6px; /* space between text and pill */
            }
        """)

        # Create a vertical layout
        vbox_layout = QtWidgets.QVBoxLayout(self)
        vbox_layout.setContentsMargins(0, 0, 0, 0)
        vbox_layout.setSpacing(15)

        # Add header with "Choose processing approach:"
        header = QtWidgets.QLabel("<b>Choose processing approach:<b>")
        vbox_layout.addWidget(header)

        ##### HINT/RECOMMENDATION TEXT FOR USERS TO READ #####
        self.hint_label = QtWidgets.QLabel("")
        self.hint_label.setWordWrap(True)
        self.hint_label.setProperty("class", "hint")
        vbox_layout.addWidget(self.hint_label)

        ##### RADIO BUTTON OPTIONS #### 
        self.button_group = QtWidgets.QButtonGroup(self)

        # 1) Option 1: Use data directly
        self.rb_opt1 = QtWidgets.QRadioButton()
        self.button_group.addButton(self.rb_opt1, 0)
        opt1_title = QtWidgets.QLabel("Perform event detection on your uploaded data")
        opt1_title.setProperty("class", "optionTitle")
        opt1_row, self.opt1_pill = self.make_radiobutton_row(self.rb_opt1, opt1_title)
        self.opt1_pill.setVisible(False)
        vbox_layout.addLayout(opt1_row, stretch = 0)

        # 2) Option 2: Downsample data to 1000 Hz before further processing
        self.rb_opt2 = QtWidgets.QRadioButton()
        self.button_group.addButton(self.rb_opt2, 1)
        opt2_title = QtWidgets.QLabel("Downsample uploaded data to 1000 Hz before further processing")
        opt2_title.setProperty("class", "optionTitle")
        opt2_row, self.opt2_pill = self.make_radiobutton_row(self.rb_opt2, opt2_title)
        vbox_layout.addLayout(opt2_row, stretch = 0)

        # 3) Option 3: Set a custom downsample rate
        self.rb_opt3 = QtWidgets.QRadioButton("")
        self.button_group.addButton(self.rb_opt3, 2)
        opt3_title = QtWidgets.QLabel("Downsample uploaded data to a custom rate before further processing")
        opt3_title.setProperty("class", "optionTitle")
        opt3_row, self.opt3_pill = self.make_radiobutton_row(self.rb_opt3, opt3_title)
        self.opt3_pill.setVisible(False) # Never recommended
        vbox_layout.addLayout(opt3_row, stretch = 0)

        # Add QSpinBox (number entry field) for setting the custom rate
        custom_line = QtWidgets.QHBoxLayout()
        custom_line.setContentsMargins(30, 0, 0, 0)  # Indent the box right of the radio button
        custom_line.setSpacing(6)
        self.custom_rate_label = QtWidgets.QLabel("Custom downsampled rate (Hz):")
        self.custom_rate_label.setStyleSheet("color: gray;")
        custom_line.addWidget(self.custom_rate_label)
        self.custom_rate = QtWidgets.QSpinBox()
        self.custom_rate.setSingleStep(50)
        self.custom_rate.setEnabled(False)
        custom_line.addWidget(self.custom_rate, 0)
        custom_line.addStretch(1) # Critical to keep box from filling the whole row
        vbox_layout.addLayout(custom_line, stretch = 0)

        # Connect signals and slots
        self.button_group.buttonToggled.connect(self._onChanged)
        self.custom_rate.valueChanged.connect(self._onChanged)

        # Update as part of initialization
        self._onChanged()

    def make_radiobutton_row(self, radio: QtWidgets.QRadioButton, title_lbl: QtWidgets.QLabel):
        """
        Helper function for assembling rows.
        Combine a "blank" radio button with a QLabel title (with the button description). 
        Returns a layout.
        """

        # Make a horizontal layout
        hbox_layout = QtWidgets.QHBoxLayout()
        hbox_layout.setSpacing(6) # Spacing between button and text

        # Add the radio button
        # Note: Use a stretch of 0 to ensure radio button position does not move to fill available horizontal space
        hbox_layout.addWidget(radio, stretch = 0, alignment = QtCore.Qt.AlignTop)

        #  Create another horizontal layout for the text that goes next to the button
        title_box = QtWidgets.QHBoxLayout()
        title_box.setSpacing(6) # Spacing between text and pill
        title_box.addWidget(title_lbl, stretch = 0, alignment = QtCore.Qt.AlignTop)

        # Add a pill 
        pill = QtWidgets.QLabel("Recommended")
        pill.setProperty("class", "pill")
        title_box.addWidget(pill, stretch = 0, alignment = QtCore.Qt.AlignTop)

        # Add a spacer to the end of the row to keep things left-aligned
        # Note: This is *critical* for the other elements' "stretch = 0" to be honored; otherwise if all elements have stretch = 0, they evenly spread across the row
        title_box.addStretch(stretch = 1)

        # Add the label and optional pill next to the radio button in the first HBoxLayout
        hbox_layout.addLayout(title_box) 

        # Return the layout with all the elements (button, label, pill)
        return hbox_layout, pill

    def update_input_fs(self, input_fs):
        """
        The input_fs is initialized as None. 
        When a recording is selected, the sampling rate is sent to this function, resulting in the hint and default selection (along with the "recommended" pill) being updated.
        """
        self.input_fs = input_fs
        self.update_hint()              # Update the top section text with the "hint"
        self.update_suggested_option()  # Update which radio button is clicked and which shows the "recommended" pill

        # Adjust the custom rate to reflect the input sampling rate (if input is below 2.5kHz) or 1000 Hz (if input is above 2.5kHz)
        self.custom_rate.setRange(10, int(max(100000, self.input_fs)))
        if self.input_fs <= 2500:
            self.custom_rate.setValue(int(self.input_fs))
        else:
            self.custom_rate.setValue(1000)

        # Emit the payload for the parameters widget to get updated
        self._onChanged()

    def update_hint(self):    
        """
        The hint is initialized as an empty string. 
        Once a recording is selected, this function is called (via "update_input_fs") and the hint will be filled in depending its sampling rate (input_fs).
        If a different recording is loaded, this function is also called and the hint is updated.
        """    
        if self.input_fs <= 2500:
            hint = f"Your sampling rate ({int(round(self.input_fs))} Hz) is within the recommended range for using Toothy. Downsampling is <b>not</b> needed for your data."
        else:
            hint = f"Your sampling rate ({int(round(self.input_fs))} Hz) <b>exceeds</b> the recommended rate for using Toothy. <b>Downsampling is recommended to significantly reduce time and space consumed.<b>"
        
        self.hint_label.setText(hint)

    def update_suggested_option(self):
        """
        Fills in a radio button for Option 1 or 2 depending on the input data's sampling rate.
        """
        # Default behavior: 
            # If the sampling rate is low (<=2500 Hz), proceed without downsampling
            # Otherwise, suggest downsampling to 1000 Hz

        if self.input_fs <= 2500:
            self.rb_opt1.setChecked(True)
            self.opt1_pill.setVisible(True)
            self.opt2_pill.setVisible(False)
        else:
            self.rb_opt2.setChecked(True)
            self.opt1_pill.setVisible(False)
            self.opt2_pill.setVisible(True)

    def _onChanged(self, *args):
        # Only custom rate editor when Option 3 is checked
        if self.rb_opt3.isChecked():
            self.custom_rate.setEnabled(True)
            self.custom_rate_label.setStyleSheet("color: black;")
        else:
            self.custom_rate.setEnabled(False)
            self.custom_rate_label.setStyleSheet("color: gray;")

        # Emit current selection
        mode = {self.rb_opt1: "keep",
                self.rb_opt2: "downsample_1000",
                self.rb_opt3: "downsample_custom",
                None: "None"}[self.button_group.checkedButton()]

        target_fs = {
            self.rb_opt1: self.input_fs,
            self.rb_opt2: 1000,
            self.rb_opt3: self.custom_rate.value(),
            None: None
        }[self.button_group.checkedButton()]
        
        payload = {
            "mode": mode,
            "fs_in": self.input_fs,
            "target_hz": target_fs
        }

        self.selection_changed.emit(payload)
        

##############################################################################
##############################################################################
################                                              ################
################                 MAIN INTERFACE               ################
################                                              ################
##############################################################################
##############################################################################

class InputDataSelectionPopup(QtWidgets.QDialog):
    """
    Data processing GUI for input recording selection and probe assignment.
    
    QDialog widget (presents a top-level window used to collect a response from the user.)
    """
    recording = None
    last_saved_ddir = None
    
    def __init__(self, parent=None, init_path=None, bulk_mode=False, bulk_probe_file=None):
        super().__init__(parent)
        self._bulk_mode = bulk_mode
        self._bulk_probe_file = bulk_probe_file

        # Set the window title (changes as the user proceeds through the workflow of this QDialog)
        self.setWindowTitle('Select input data')

        # Set icon for upper left corner and toolbar
        self.setWindowIcon(QtGui.QIcon(':/resources/logo.png'))

        # Remove the "?" help button
        gi.remove_help_button(self)

        # Make the QDialog background white
        ss = "QDialog {background-color : white;}"
        self.setStyleSheet(ss)
        
        # Load parameters
        self.PARAMS = ephys.read_params() # Returns a dictionary of validated default parameters

        # Get the initial directories
        input_data_dir_path, probe_config_dir_path, probe_file_path, param_file_path  = ephys.base_dirs()
        self.default_probe_file = probe_file_path
        
        # Set up the window; creates "self.fsw" referenced two lines down
        self.gen_layout()

        # Connect all the signals and slots
        self.connect_signals()

        # Update the file selection widget with the default directory for input data
        # Note: this will display the directory in the QLineEdit, but keep the outline red, since validation will fail on a path that is a directory rather than a file
        self.fsw.update_filepath(input_data_dir_path)

        # Hide downstream sections of the workflow
        self.probe_gbox.hide()
        self.save_gbox.hide()

        # Pre-load path if provided via bulk mode (deferred until event loop starts)
        if init_path:
            QtCore.QTimer.singleShot(0, lambda: self.fsw.update_filepath(init_path))

    def gen_layout(self):
        """
        Set up layout and add widgets for input recording selection.
        """
        # Create a layout with QVBoxLayout (linear vertical layout) and assign this layout to the main QDialog widget (self)
        # Note:
            # This VBoxLayout manages the placement of any child widgets added to the main QDialog widget
            # This syntax is equivalent to creating a layout, then calling self.setLayout(layout)
        self.layout = QtWidgets.QVBoxLayout(self)

        # Vertical spacing between rows (in pixels)
        self.layout.setSpacing(10)
        
        #######################################
        ##### INPUT DATA SOURCE SELECTION #####
        #######################################
    
        # Create a QFrame widget for the input data source section. 
        # We were previously using a QGroupBox, which provides a box frame with a title passed as a str, but we don't have a title and it would leave an undesirable tiny margin
        self.recording_selection_frame = QtWidgets.QFrame()

        # Set stylesheet for the input data QFrame to have a light gray background
        qframe_ss = 'QFrame {background-color : #E8E8E8;}'
        self.recording_selection_frame.setStyleSheet(qframe_ss)

        ##### FILE SELECTION WIDGET #####
        # Get a nested QVBoxLayout to add file selection widget
            # Interwidget takes a parent widget and orientation ('h' or 'v') and returns a layout of a child widget added to the parent widget
            # Send our parent QGroupBox to interwidget
            # Applies a vertical layout to our parent QGroupBox ("interlayout")
            # Adds a QWidget ("interwidget") with a vertical layout (QVBoxLayout, specified by the "v" parameter) to the parent QGroupBox
            # Returns a tuple of (interlayout, interwidget, layout), where layout is the interwidget's layout
            # We index into the tuple and just take "layout"
        ddir_vbox = pyfx.InterWidgets(self.recording_selection_frame, 'v')[2] # QVBoxLayout

        # Create directory selection widget (instance of gui_items.FileSelectionWidget)
        self.fsw = RawRecordingSelectionWidget(title='<b>Select a recording:</b>')

        # Add directory selecton widget to layout
        ddir_vbox.addWidget(self.fsw)

        ##### LISTING SUPPORTED FILE FORMATS #####
        # Add a label for the heading "Supported file formats:"
        header_label = QtWidgets.QLabel("<i>Supported file formats:</i>")
        header_label.setContentsMargins(0, 10, 0, 0)    # Add space to the top margin (left, top, right, bottom)
        self.fsw.vlay.addWidget(header_label)           # Add header label to the bottom of the directory selection widget's layout
        
        # Add labels listing the supported data formats
        data_format_layout = QtWidgets.QVBoxLayout()
        data_format_layout.setSpacing(3) # Reduce row spacing
        for k, (file_format, file_ext) in dp.supported_formats.items():
            format_label = QtWidgets.QLabel(f'• {file_format} ({file_ext})')
            format_label.setContentsMargins(25, 0, 0, 0)    # Add space to the left margin (left, top, right, bottom) to appear indented
            data_format_layout.addWidget(format_label)      # Add label to vertical layout
        self.fsw.vlay.addLayout(data_format_layout)         # Add data format layout to the bottom of the directory selection widget's layout
                
        ##### DISPLAYING RESULT OF SUCCESSFULLY SELECTING & EXTRACTING FILE #####
        # Create a QFrame widget for displaying result of file selection and make the background white
        self.metadata_frame = QtWidgets.QFrame() 
        ss = 'QFrame {background-color : white;}'
        self.metadata_frame.setStyleSheet(ss)
        self.metadata_frame_layout = pyfx.InterWidgets(self.metadata_frame, 'v')[2] # Get a QVBoxLayout of a child widget in the QGroupBox

        # File loaded message
        self.file_loaded_label = QtWidgets.QLabel("") # Later, update the text once extractor runs
        self.file_loaded_label.setStyleSheet("color: darkgreen;")
        self.metadata_frame_layout.addWidget(self.file_loaded_label)

        # Filename
        self.file_name_label = QtWidgets.QLabel("") # Later, update the text once extractor runs
        self.metadata_frame_layout.addWidget(self.file_name_label)

        # Recording metadata
        self.recording_metadata_header = QtWidgets.QLabel("<b>Recording properties:</b>")
        self.metadata_frame_layout.addWidget(self.recording_metadata_header)
        self.metadata_display_label = QtWidgets.QLabel("") # Later, update the text once extractor runs
        self.metadata_display_label.setContentsMargins(25, 0, 0, 0)  # Add padding to the left so it looks indented
        # self.metadata_display_label.setFont(QtGui.QFont("Courier New") # Text needs to be monospace for alignment; use <code> tags later instead of setting the font for monospacing
        self.metadata_frame_layout.addWidget(self.metadata_display_label)

        # Time metadata
        self.time_metadata_header = QtWidgets.QLabel("<b>Timestamps properties:</b>")
        self.metadata_frame_layout.addWidget(self.time_metadata_header)
        self.time_metadata_label = QtWidgets.QLabel("") # Later, update the text once extractor runs
        self.time_metadata_label.setContentsMargins(25, 0, 0, 0)  # Add padding to the left so it looks indented
        self.metadata_frame_layout.addWidget(self.time_metadata_label)

        # Hide the section to start; make visible later in "extractor_finished_slot"
        self.metadata_frame.hide() 

        ##### DISPLAYING RESULT OF EXTRACTION ISSUE #####
        # Create a QFrame widget for displaying result of file selection and make the background white
        self.extraction_error_frame = QtWidgets.QFrame() 
        ss = 'QFrame {background-color : white;}'
        self.extraction_error_frame.setStyleSheet(ss)
        self.extraction_error_frame_layout = pyfx.InterWidgets(self.extraction_error_frame, 'v')[2] # Get a QVBoxLayout of a child widget in the QGroupBox

        # File loaded message
        self.file_issue_label = QtWidgets.QLabel("") # Later, update the text once extractor runs
        self.file_issue_label.setStyleSheet("color: #991F08;")
        self.extraction_error_frame_layout.addWidget(self.file_issue_label)

        # Filename
        self.file_name_label_copy = QtWidgets.QLabel("") # Later, update the text once extractor runs
        self.extraction_error_frame_layout.addWidget(self.file_name_label_copy) # Same label as used for regular metadata

        # Hide the section to start; make visible later in "extractor_error_slot"
        self.extraction_error_frame.hide() 

        ##### SETTINGS (DOWNSAMPLE) RECOMMENDATION & CONFIRMATION #####
        # Create a QFrame widget for the processing settings selection section 
        self.downsample_frame = QtWidgets.QFrame()
        self.downsample_frame.setStyleSheet(qframe_ss) # Give it the same gray background as the top section
        self.downsample_frame_layout = pyfx.InterWidgets(self.downsample_frame, 'v')[2] # Get a QVBoxLayout of a child widget in the QGroupBox

        # Create the widget for displaying the hint and processing options
        self.processing_options_widget = ProcessingOptionsWidget(parent = self)
        self.downsample_frame_layout.addWidget(self.processing_options_widget)

        # Hide the section to start; make visible later in "extractor_finished_slot"
        self.downsample_frame.hide()

        #####################################################################################################################

        ### settings widget
        self.settings_w = QtWidgets.QWidget()
        settings_vlay = QtWidgets.QVBoxLayout(self.settings_w)
        settings_vlay.setContentsMargins(0,0,0,0)
        
        # Initialize main parameter input widget, embed in scroll area
        self.params_widget = qparam.ParamObject(params=dict(self.PARAMS), mode='data_processing', parent=self)
        self.qscroll = QtWidgets.QScrollArea()
        self.qscroll.horizontalScrollBar().hide()
        self.qscroll.setWidgetResizable(True)
        self.qscroll.setWidget(self.params_widget)
        qh = pyfx.ScreenRect(perc_height=0.25, keep_aspect=False).height()
        self.qscroll.setMaximumHeight(qh)
        self.qscroll.hide()

        # create settings button to show/hide param widgets
        self.params_bar = QtWidgets.QPushButton('Confirm processing parameters')
        self.params_bar.setCheckable(True)
        self.params_bar.setFocusPolicy(QtCore.Qt.NoFocus)
        # self.params_bar.setStyleSheet(pyfx.dict2ss(QSS.EXPAND_PARAMS_BTN))
        # create icon to indicate warnings about 1 or more parameter inputs
        self.params_warning_btn = QtWidgets.QPushButton()
        self.params_warning_btn.setObjectName('params_warning')
        self.params_warning_btn.setCheckable(True)
        self.params_warning_btn.setFlat(True)
        self.params_warning_btn.setFixedSize(25,25)
        self.params_warning_btn.setEnabled(False)
        self.params_warning_btn.setStyleSheet(pyfx.dict2ss(QSS.ICON_BTN))
        self.params_warning_btn.setFlat(True)
        self.params_warning_btn.hide()
        
        # Bar to put settings/base folders/metadata icon (two down, 1 to go)
        bbar = QtWidgets.QHBoxLayout()
        bbar.addWidget(self.params_warning_btn, stretch=0)
        bbar.addWidget(self.params_bar, stretch=2)

        settings_vlay.addLayout(bbar)
        
        self.settings_container = QtWidgets.QSplitter()
        self.settings_container.addWidget(self.qscroll)
        settings_vlay.addWidget(self.settings_container)
        #####################################################################################################################
        
        ##### PROBE ASSIGNMENT SECTION #####
        self.probe_gbox = QtWidgets.QGroupBox() 
        self.probe_gbox.setStyleSheet('QGroupBox {border-width : 10px;' # WAS 0 ; fix
                                      'font-weight : bold; text-decoration : underline;}')
        self.probe_gbox.setStyleSheet( 'QGroupBox {background-color : rgba(230,230,230,255);}')
        self.probe_vbox = pyfx.InterWidgets(self.probe_gbox, 'v')[2]
        self.paw = None
        
        ##### RESULTS DIRECTORY LOCATION #####
        save_lbl = QtWidgets.QLabel('<b>Select results directory:</b>')
        self.save_le = QtWidgets.QLineEdit()
        self.save_le.setReadOnly(True)
        self.save_ddir_btn = QtWidgets.QPushButton() # file dialog launch button
        self.save_ddir_btn.setIcon(QtGui.QIcon(':/icons/folder.png'))
        self.save_ddir_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.save_gbox = pyfx.get_widget_container('h', save_lbl, self.save_le, 
                                                   self.save_ddir_btn, spacing=5, 
                                                   widget='frame')

        ##### Bottom row action buttons #####
        bbox = QtWidgets.QHBoxLayout()

        # Go to probe mapping step (from file selection step)
        self.probe_map_btn = QtWidgets.QPushButton('Map to probe(s)')
        self.probe_map_btn.setEnabled(False)    # Visible but disabled at the start

        # Go back to file selection step (from probe mapping step)
        self.back_btn = QtWidgets.QPushButton('Back')
        self.back_btn.setVisible(False)         # Inivisble at the start
        
        # Begin processing data
        self.pipeline_btn = QtWidgets.QPushButton('Process data!')
        self.pipeline_btn.setVisible(False)
        self.pipeline_btn.setEnabled(False)     # Invisible at the start

        # Add buttons to QHBoxLayout
        for button in [self.probe_map_btn, self.back_btn, self.pipeline_btn]:
            bbox.addWidget(button)
        
        ##### ADD WIDGETS TO CENTRAL LAYOUT #####
        self.recording_selection_frame.setMinimumWidth(600)        # Sets a minimum width for the overall file selection frame
        self.layout.addWidget(self.recording_selection_frame)      # File selection frame (top section)
        self.layout.addWidget(self.metadata_frame)                  # Metadata frame (hidden; appears under file selection after user selects a file if extraction is successful)
        self.layout.addWidget(self.extraction_error_frame)          # Error message frame (hidden; appears under file selection after user selects a file if an error occurs)
        self.layout.addWidget(self.downsample_frame)
        self.layout.addWidget(self.settings_w)
        self.layout.addWidget(self.probe_gbox)
        self.layout.addWidget(self.save_gbox)
        line0 = pyfx.DividerLine()
        self.layout.addWidget(line0)
        self.layout.addLayout(bbox)                         # Bottom buttons (Map to probes, Back, Process data)
        self.layout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        
        # "loading" spinner animation
        self.spinner_window = gi.SpinnerWindow(self)
        self.spinner_window.spinner.setInnerRadius(25)
        self.spinner_window.spinner.setNumberOfLines(10)
        #self.spinner_window.layout.setContentsMargins(5,5,5,5)
        self.spinner_window.layout.setSpacing(0)
        #self.spinner_window.adjust_labelSize(lw=2.5, lh=0.65, ww=3)
        
    def connect_signals(self):
        """ Connect GUI inputs """
        # File selection and extraction
        self.fsw.ppath_btn.clicked.connect(self.fsw.select_filepath)    # Connect folder icon button (ppath_btn) clicked signal to "select_filepath" slot (launches file selection widget)
        self.fsw.signal.connect(self.ppath_updated)                     # Connect FileSelectionWidget output signal "signal" to "ppath_updated" slot, which launches ExtractorWorker
        self.fsw.read_array_signal.connect(self.load_array_worker)      # Connect signal "read_array_signal" to "load_array_worker" slot, which processed NumPy and MATLAB files so that the user enters/confirms the data array and metadata for those files

        self.params_widget.warning_update_signal.connect(self.update_param_warnings)
        self.probe_map_btn.clicked.connect(self.start_probe_assignment)

        self.pipeline_btn.clicked.connect(self.pipeline_worker)

        self.back_btn.clicked.connect(self.back_to_selection)
        self.save_ddir_btn.clicked.connect(self.set_save_ddir)
        self.params_bar.toggled.connect(lambda x: self.qscroll.setVisible(x))
    
    def create_worker_thread(self):
        """
        Creates a parallel worker thread for long-running processing steps.
        - Generates a worker thread (QThread).
        - Adds worker object (self.worker_object) to the worker thread (self.worker_object a QObject of either ArrayReader, ExtractorWorker, or DataWorker).
        - Connects worker thread's "started" signal to worker object's "run" slot. For each worker object, "run" has been defined to contain the important processing.
        - Connects worker object "progress_signal" (which emits strings) to the spinner window slot to show progress messages.
        - Connects worker thread "finished" signal to slots that d
        """
        # Create a QThread (worker thread)
        self.worker_thread = QtCore.QThread()

        # Move the worker object to the worker thread
        self.worker_object.moveToThread(self.worker_thread)

        # Connect the started signal for the worker thread to the worker object's "run" slot
        # When the thread starts, it will also run the worker object
        self.worker_thread.started.connect(self.worker_object.run)

        # Connect the worker object's progress signal to the spinner window's "report_progress_string" slot
        self.worker_object.progress_signal.connect(self.spinner_window.report_progress_string)

        # Connect the worker thread's finished signal to several slots
        self.worker_object.finished.connect(self.worker_thread.quit)        # Exit event loop
        self.worker_thread.finished.connect(self.worker_object.deleteLater) # Tell PyQt to delete the worker object when the event loop ends
        self.worker_thread.finished.connect(self.worker_thread.deleteLater) # Tell PyQt to delete the worker thread when the event loop ends
        self.worker_thread.finished.connect(self.finished_slot)             # Run "finished_slot" which stops the progress spinner and sets the worker object/thread to None
    
    def start_qthread(self):
        """ Start worker thread """
        # Create worker thread self.worker_thread (also moves worker object to the worker thread)
        self.create_worker_thread()

        # Start the progress spinner
        self.spinner_window.start_spinner()

        # Start the worker thread
        self.worker_thread.start()
    
    @QtCore.pyqtSlot()
    def finished_slot(self):
        """ Worker thread stopped """
        self.spinner_window.stop_spinner() # stop "loading" spinner icon
        self.worker_object = None
        self.worker_thread = None
        
    def ppath_updated(self, status):
        """ 
        Slot called when "signal" is emitted from the RawRecordingSelectionWidget after a file is selected.
        - status: "signal" emits a True/False to indicate if the selected file is valid 
        """

        # If True (path is valid), update the text in the QLineEdit widget and run the ExtractorWorker
        if status:
            ppath = self.fsw.le.text()
            data_type = dp.get_data_format(ppath)
            self.extractor_worker(ppath, data_type) # Load new recording
        else:
            # Set the status of the "Map to probe(s)" button to match "status"
            self.probe_map_btn.setEnabled(status)
            data_type = None 
            self.recording = None # delete previous recording (if any)
            self.params_widget.trange.set_duration(np.inf)
            self.params_widget.lfp_fs.set_fs(None)
    
    #######################
    ##### ArrayReader #####
    #######################

    def load_array_worker(self, ppath):
        """
        Read raw data array into memory. Slot called  
        """
        self.worker_object = ArrayReader(ppath)                                 # Initialize ArrayReader QObject
        self.worker_object.data_signal.connect(self.load_array_finished_slot)   # Connect ArrayReader signal "data_signal" to slot "load_array_finished_slot"
        self.worker_object.error_signal.connect(self.load_array_error_slot)     # Connect ArrayReader signal "error_signal" to slot "load_array_error_slot"
        self.start_qthread()                                                    # Create worker objects and worker thread, add worker object to worker thread, start progress spinner window, and start worker thread
        
    # Decorate with pyqtSlot to ensure function can be connected as a slot to the signal before thread is started    
    @QtCore.pyqtSlot(str, dict, dict)
    def load_array_finished_slot(self, file_path, data_dict, metadata_dict):
        """
        Pass valid arrays and initial metadata to the user.
        Slot function connected to "data_signal" signal of class ArrayReader which emits the fil path, data dictionary, and metadata dictionary.
        If there is more than one 2D array, the user chooses the key of the one they want to use.
        Final line calls a function in the file selection widget for the user to confirm metadata values.
        """
        # If the file contains multiple 2-dimensional arrays, the user will choose one.
        if len(data_dict) > 1:
            # Create a QDialog popup for with buttons for the user to select a key
            lbl = '<u>Select LFP dataset from available keys:<br><br></u><i>(hover over key for array dimensions)<br></i>'
            key_selection_dialog = gi.ButtonPopup(*data_dict.keys(), label=lbl, parent=self)

            # For each key button, add hover text with the dimensions (rows by columns) (appears when user hovers over button with mouse) 
            for b in key_selection_dialog.btns:
                nrows, ncols = data_dict[b.text()].shape
                b.setToolTip(f'{nrows} rows x {ncols} columns')
            
            # User selects a LFP dataset key
            if key_selection_dialog.exec(): 
                data_array = data_dict[key_selection_dialog.result]
            
            # User closes the selection window without selecting a key
            else:
                gi.MsgboxError('Aborted LFP dataset selection.', parent=self).exec() # Show error message
                return # End slot and return to data selection GUI
        
        # If the file contains only one 2-dimensional dataset, it is automatically selected.
        else: 
            data_array = list(data_dict.values())[0]

        # Lastly, the user will manually confirm sampling rate, units, etc.
        # In this function, the data/metadata is then stored within "fsw" variables which are referenced during file extraction.
        self.fsw.enter_array_metadata(file_path, data_array, metadata_dict)
        
    @QtCore.pyqtSlot(str, str)
    def load_array_error_slot(self, ppath, error_msg):
        """ Handle invalid NPY/MAT files """
        gi.MsgboxError(error_msg, parent=self).exec()
        self.fsw.update_filepath(ppath, False)

    #######################
        
    def get_nwb_dataset(self, ppath):
        """ Identify target ephys dataset in NWB file """
        try:  # try getting names of available ElectricalSeries in NWB file
            eseries = dp.get_nwb_eseries(ppath)
        except Exception as e:  # file missing or corrupted
            gi.MsgboxError(f'Error: {str(e)}', parent=self).exec()
            return False
        if len(eseries) == 0:   # no analyzable data
            gi.MsgboxError('NWB file contains no ElectricalSeries with valid electrodes.', parent=self).exec()
            return False
        elif len(eseries) == 1: # return the only available dataset
            return eseries[0]
        elif len(eseries) > 1:  # ask user to select a target dataset
            lbl = '<u>Select an ElectricalSeries dataset</u>'
            dlg = gi.ButtonPopup(*eseries, label=lbl, parent=self)
            if dlg.exec():
                return str(dlg.result)
            else:
                gi.MsgboxError('Aborted NWB dataset selection.', parent=self).exec()
                return False
            
    def extractor_worker(self, file_path, data_type):
        """ Instantiate spikeinterface Extractor for raw recording """
        # Initialize data_array and metadata (only filled in for NumPy/MATLAB; otherwise kept as None/empty when passed to ExtractorWorker)
        data_array, metadata = None, {}

        # Get data_array and metadata from NumPy or MATLAB files; stored in the FileSelectionWidget by "enter_array_metadata", called after ArrayWorker runs
        if data_type in ['NPY','MAT']:
            data_array, metadata = self.fsw.data_array, self.fsw.metadata_dict

            # Transpose the array if needed to ensure the rows are samples and columns are channels
                # .shape returns (rows, columns)
                # if the number of rows is equal to the number of channels, transpose so that the rows = samples and columns = channels
            if data_array.shape[0] == metadata['nch']: 
                data_array = data_array.T
        
        # For NWB files
        electrical_series_path = None
        if data_type == 'NWB':
            es_name = self.get_nwb_dataset(file_path)
            if es_name == False:
                self.fsw.update_filepath(file_path, False)
                return
            electrical_series_path = f'acquisition/{es_name}'
        
        # Make a dictionary where each variable name is a key with the variable as its value
        extractor_input_kwargs = dict(data_array=data_array, metadata=metadata, electrical_series_path=electrical_series_path)
        
        # Create extractor worker. Send the file path and any available data (see "extractor_input_kwargs").
        self.worker_object = ExtractorWorker(file_path, **extractor_input_kwargs)

        # Connect ExtractorWorker signals to slots in this class
            # "data_signal" emits two objects (recording, time) to "extractor_finished_slot" which assigns them to class variables self.recording and self.time
            # "error_signal" sends a filepath and error message to "extractor_error_slot" which displays the error mesaage.
        self.worker_object.data_signal.connect(self.extractor_finished_slot)
        self.worker_object.error_signal.connect(self.extractor_error_slot)

        # Create a worker thread, add self.worker_object (ExtractorWorker) to it, and start the worker thread 
        self.start_qthread()
    
    @QtCore.pyqtSlot(object, object)
    def extractor_finished_slot(self, recording, time):
        """ Load recording data into pipeline """
        # Receive recording and time from ExtractorWorker
        self.recording = recording      # ExtractorWorker emits recording; received here by "extractor_finished_slot" and assigned to self.recording
        self.time = time                # ExtractorWorker also emits time; received here and assigned to self.time
        
        # Get a dictionary of metadata from the recording
        recording_metadata = dp.get_meta_from_recording(self.recording)

        ##### Show and update the metadata section #####
        self.metadata_frame.setVisible(True)
        self.extraction_error_frame.setVisible(False) # Hide the "error" section, since extraction was successful

        file_path = self.fsw.le.text()
        file_name = os.path.basename(file_path)
        data_type = dp.get_data_format(file_path)
        file_type, file_ext = dp.supported_formats[data_type]

        # Line 1: {file type} successfully loaded (in green)
        if data_type == "Neuralynx":
            file_loaded_str = f"<b>{file_type}</b> files successfully loaded."
        else:
            file_loaded_str = f"<b>{file_type}</b> file successfully loaded."
        self.file_loaded_label.setText(file_loaded_str)
        
        # Line 2: File name: {file name}
        if data_type == "Neuralynx":
            folder_name = os.path.dirname(file_path).split("/")[-1]
            file_name_str = f"<b>Folder name (all .ncs files loaded):</b> {folder_name}"
        else:
            file_name_str = f"<b>File name:</b> {file_name}"
        self.file_name_label.setText(file_name_str)

        # Line 3: Recording properties: number of channels, number of samples, and sampling rate
        display_properties = {
            "N channels": recording_metadata["nch"],
            "N samples": recording_metadata["nsamples"],
            "Sampling rate (Hz)": np.round(recording_metadata['fs'], 3)
        }
        recording_properties_str = ""
        for label, val in display_properties.items():
            if len(recording_properties_str) > 0:
                recording_properties_str += "\n" # Add a newline for all rows except the first
            recording_properties_str += f"{f'{label}:'.ljust(25)}{val}" 
        # Introducing html tags (like <code>) ignores newline and whitespace characters, so we'll also add <pre> to respect whitespace
        # Important: <pre> needs to go inside <code> tags to avoid introducing large newline spaces
        self.metadata_display_label.setText("<code><pre>"+recording_properties_str+"</code>")

        # Line 4: Time properties: start time, end time, duration
        if self.time is not None:
            t_start = self.time[0]
            t_end = self.time[-1]
            display_properties = {
                "Start time": np.round(t_start, 3),
                "End time": np.round(t_end, 3),
                "Duration (s)": np.round(t_end - t_start, 3),
            }
            time_properties_str = ""
            for label, val in display_properties.items():
                if len(time_properties_str) > 0:
                    time_properties_str += "\n" # Add a newline for all rows except the first
                time_properties_str += f"{f'{label}:'.ljust(25)}{val}"
            self.time_metadata_label.setText("<code><pre>"+time_properties_str+"</code>")
        else: # time is None
            self.time_metadata_label.setText("<i>Timestamps not available. Toothy will return sample indices for each event.<i>")

        # Make the next part of the workflow (choosing a processing option for downsampling) visible and send metadata to it to update the suggested action
        self.downsample_frame.setVisible(True)
        self.processing_options_widget.selection_changed.connect(self.update_target_fs) # Connect the signal for making a selection to a slot for updating the parameters
        self.processing_options_widget.update_input_fs(recording_metadata["fs"])

        # Set downsample flag
        if int(recording_metadata["fs"]) == int(self.target_fs):
            self.downsample_bool = False
        else:
            self.downsample_bool = True

        # Enable the "Map to probe(s)" button to match "status"
        self.probe_map_btn.setEnabled(True)

        # Update parameter widgets with the recording's metadata
        self.params_widget.lfp_fs.set_fs(float(recording_metadata['fs']))       # Updates lfp_fs with the Fs from the recording (for comparison with desired downsample Fs)
        self.params_widget.trange.set_duration(self.recording.get_duration())   # Updates the duration with the duration from the recording

        # Center the window after expanding it with all this information
        pyfx.center_window(self)

        # Bulk mode: auto-advance through probe assignment and pipeline
        if self._bulk_mode and self._bulk_probe_file:
            QtCore.QTimer.singleShot(150, self._bulk_auto_proceed)

    def _bulk_auto_proceed(self):
        """Auto-advance through probe assignment and pipeline in bulk mode."""
        self.default_probe_file = self._bulk_probe_file
        self.start_probe_assignment()
        if self.pipeline_btn.isEnabled():
            self.pipeline_worker()

    def update_target_fs(self, payload):
        self.target_fs = payload["target_hz"]
        self.params_widget.lfp_fs.update_param(self.target_fs)   # Update param!
        new_params, _ = self.params_widget.get_param_dict_from_gui()
        self.params_widget.update_gui_from_param_dict(new_params)

        # Set downsample flag
        if int(payload["fs_in"]) == int(self.target_fs):
            self.downsample_bool = False
        else:
            self.downsample_bool = True

    
    @QtCore.pyqtSlot(str, str)
    def extractor_error_slot(self, ppath, error_msg):
        """ Handle data extraction errors """
        gi.MsgboxError(error_msg, parent=self).exec()
        self.fsw.update_filepath(ppath, False)

        # Update metadata box
        self.metadata_frame.setVisible(False)   # Hide the metadata box if extraction failed
        self.downsample_frame.setVisible(False)  # Hide the next part of the workflow if extraction failed
        self.extraction_error_frame.setVisible(True)
        file_path = ppath
        file_name = os.path.basename(file_path)
        data_type = dp.get_data_format(file_path)
        file_type, file_ext = dp.supported_formats[data_type]

        # Line 1: Error loading {file type} file. (in red)
        file_loaded_str = f"Error loading <b>{file_type}</b> file."
        self.file_issue_label.setText(file_loaded_str)
        
        # Line 2: File name: {file name}
        if data_type == "Neuralynx":
            folder_name = os.path.dirname(file_path).split("/")[-1]
            file_name_str = f"<b>Folder name (all .ncs files loaded):</b> {folder_name}"
        else:
            file_name_str = f"<b>File name:</b> {file_name}"
        self.file_name_label_copy.setText(file_name_str)

    def test_settings(self, sampling_rate):
        self.downsample_frame.setVisible(True)
        recording_metadata = dp.get_meta_from_recording(self.recording)
        section = ProcessingOptionsWidget(parent = self)
        section.update_hint(recording_metadata["fs"])
        self.downsample_frame_layout.addWidget(section)

    def start_probe_assignment(self):
        """ Initiate probe mapping phase """
        NCH = self.recording.get_num_channels()
        # initialize probe box
        self.paw = ProbeAssignmentWidget(NCH)
        self.paw.check_signal.connect(self.update_probe_config)
        self.probe_vbox.addWidget(self.paw)
        self.probe_gbox.setVisible(True)
        self.save_gbox.setVisible(True)
        self.back_btn.setVisible(True)
        self.probe_map_btn.setVisible(False)
        self.pipeline_btn.setVisible(True)
        
        # initialize save box
        ppath = self.recording.get_annotation('ppath')
        raw_ddir = os.path.dirname(ppath)
        init_save_ddir = str(Path(raw_ddir, pyfx.unique_fname(raw_ddir, 'toothy'))) # changed from processed_data
        self.save_le.setText(init_save_ddir)
        # try loading and adding default probe if it meets the criteria
        dflt_probe = ephys.read_probe_file(self.default_probe_file)
        if (dflt_probe is not None) and (dflt_probe.get_contact_count() <= NCH):
            self.paw.add_probe_row(dflt_probe)
        # disable data loading
        self.recording_selection_frame.setEnabled(False)
        self.setWindowTitle('Map to probe(s)')
        pyfx.center_window(self)
        
    def assemble_probe_group(self):
        """ Return ProbeGroup object from row items in probe assignment widget """
        PROBE_GROUP = prif.ProbeGroup()
        items = pyfx.layout_items(self.paw.qlayout)
        for i,item in enumerate(items):
            prb  = item.probe
            rows = item.ROWS  # group of rows belonging to this probe
            # reorder assigned rows by device indices
            sorted_rows = [rows[dvi] for dvi in prb.device_channel_indices]
            prb.set_contact_ids(rows)
            # device_indices * nprobes + start_row = sorted_rows
            prb.set_device_channel_indices(sorted_rows)
            if i > 0:  # make sure probe boundaries do not overlap
                xmax = max(PROBE_GROUP.probes[-1].contact_positions[:,0])
                cur_xmin = min(prb.contact_positions[:,0])
                prb.contact_positions[:,0] += (xmax - cur_xmin + 1)
            PROBE_GROUP.add_probe(item.probe)
        return PROBE_GROUP
        
    def update_probe_config(self):
        """ Check for valid probe assignment """
        x = bool(self.paw.probe_done_btn.isEnabled())
        # self.paw.prb_icon_btn.new_status(x)
        self.enable_pipeline_btn()
    
    def update_param_warnings(self, n, ddict):
        """ Check for any input parameter warnings """
        x = n > 0
        self.params_warning_btn.setChecked(x)
        self.params_warning_btn.setVisible(x)
        self.enable_pipeline_btn()
    
    def enable_pipeline_btn(self):
        """ Enable processing pipeline """
        a = self.paw is not None and self.paw.probe_done_btn.isEnabled()
        b = not self.params_warning_btn.isChecked()
        self.pipeline_btn.setEnabled(a and b)
    
    ######### processing pipeline
    
    def pipeline_worker(self):
        """
        Run processing pipeline.
        """
        # META = dp.get_meta_from_recording(self.recording)

        # create probe group with all probe objects used in the recording
        self.PROBE_GROUP = self.assemble_probe_group()

        # Get updated analysis parameters
        PARAMS = self.params_widget.DEFAULTS  # Get a parameter dictionary (with default Toothy values)
        param_dict = self.params_widget.get_param_dict_from_gui()[0]  # Get the parameters entered into the GUI and update the dict
        PARAMS.update(param_dict)
        
        # Create empty folder for processed data
        save_ddir = self.save_le.text()
        if os.path.isdir(save_ddir):
            shutil.rmtree(save_ddir)  # delete existing directory
        if os.path.isfile(save_ddir):
            os.remove(save_ddir)  # delete existing file
        os.makedirs(save_ddir)
        
        # Create data processor worker
        self.worker_object = DataWorker()
        self.worker_object.init_source(self.recording, self.time, self.PROBE_GROUP, PARAMS=PARAMS, SAVE_DDIR=save_ddir, DOWNSAMPLE_NEEDED=self.downsample_bool)
        self.worker_object.data_signal.connect(self.pipeline_finished_slot)
        self.worker_object.error_signal.connect(self.pipeline_error_slot)
        self.start_qthread()
    
    @QtCore.pyqtSlot()
    def pipeline_finished_slot(self):
        """ Worker successfully completed the processing pipeline """
        self.last_saved_ddir = str(self.save_le.text())
        if self._bulk_mode:
            self.accept()
            return
        msg = 'Data processing complete!<br><br>Load another recording?'
        res = gi.MsgboxSave(msg, parent=self).exec()
        if res == QtWidgets.QMessageBox.Yes:
            self.back_btn.click()  # select another recording for processing
        else:  # close window
            self.accept()
    
    @QtCore.pyqtSlot(str)
    def pipeline_error_slot(self, error_msg):
        """ Worker encountered an error in the processing pipeline """
        gi.MsgboxError(error_msg, parent=self).exec()
        # delete incomplete recording folder
        save_ddir = self.save_le.text()
        if os.path.isdir(save_ddir):
            folder_name = os.path.basename(save_ddir)
            filestring = ', '.join(map(lambda f: f'"{f}"', os.listdir(save_ddir)))
            print(f'"{folder_name}" contains the following file(s): {filestring}' + os.linesep)
            res = ''
            while res.lower() not in ['y','n']:
                res = input('Delete folder? (y/n) --> ')
            print('')
            if res == 'y':
                shutil.rmtree(save_ddir) # delete incomplete recording folder
                print(f'"{folder_name}" folder deleted!')
    
    def set_save_ddir(self):
        """ Select location of processed data folder """
        init_ddir = os.path.dirname(self.save_le.text())
        init_dirname = os.path.basename(self.save_le.text())

        save_ddir = ephys.select_save_directory(init_ddir=init_ddir, init_dirname=init_dirname,
                                                title='Set processed data directory', parent=self)
        if save_ddir:
            self.save_le.setText(save_ddir)
    
    def back_to_selection(self):
        """ Return to the raw data selection step """
        self.setWindowTitle('Select input data')
        self.back_btn.setVisible(False)
        self.probe_map_btn.setVisible(True)
        self.pipeline_btn.setVisible(False)
        self.pipeline_btn.setEnabled(False)
        self.recording_selection_frame.setEnabled(True)
        self.probe_gbox.setVisible(False)
        self.save_gbox.setVisible(False)
        # delete probe assignment widget
        self.probe_vbox.removeWidget(self.paw)
        self.paw.deleteLater()
        self.paw = None
        
class BulkLoadPopup(QtWidgets.QDialog):
    """
    Popup for bulk-loading multiple recordings sequentially through the
    Step 1 pipeline.  The user adds up to 99 paths, clicks Run All, and
    each path is processed one at a time via InputDataSelectionPopup.
    """

    MAX_ROWS = 99

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Bulk Load Data')
        self.setWindowIcon(QtGui.QIcon(':/resources/logo.png'))
        gi.remove_help_button(self)
        self.setStyleSheet("QDialog { background-color: white; }")
        self._path_rows = []   # list of (frame, line_edit) tuples
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        outer = QtWidgets.QVBoxLayout(self)
        outer.setSpacing(10)
        outer.setContentsMargins(14, 14, 14, 14)

        # Title label
        title = QtWidgets.QLabel('Step 1: Bulk Load Data')
        title.setAlignment(QtCore.Qt.AlignCenter)
        title_font = title.font()
        title_font.setPointSize(13)
        title_font.setBold(True)
        title.setFont(title_font)
        outer.addWidget(title)

        # Probe file config row
        probe_frame = QtWidgets.QFrame()
        probe_frame.setStyleSheet("QFrame { background-color: #E8E8E8; border-radius: 4px; }")
        probe_hlayout = QtWidgets.QHBoxLayout(probe_frame)
        probe_hlayout.setContentsMargins(10, 8, 10, 8)
        probe_hlayout.setSpacing(6)
        probe_lbl = QtWidgets.QLabel('<b>Probe file:</b>')
        probe_lbl.setFixedWidth(78)
        self.probe_le = QtWidgets.QLineEdit()
        self.probe_le.setReadOnly(True)
        self.probe_le.setPlaceholderText('Browse for probe configuration file (.json / .prb / .mat)...')
        probe_browse_btn = QtWidgets.QPushButton('Browse')
        probe_browse_btn.setStyleSheet(pyfx.dict2ss(QSS.TOGGLE_BTN))
        probe_browse_btn.setFixedWidth(70)
        probe_browse_btn.clicked.connect(self._browse_probe)
        probe_hlayout.addWidget(probe_lbl)
        probe_hlayout.addWidget(self.probe_le, stretch=1)
        probe_hlayout.addWidget(probe_browse_btn)
        outer.addWidget(probe_frame)

        # Scroll area that holds the path rows
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)

        self.rows_widget = QtWidgets.QWidget()
        self.rows_widget.setStyleSheet("background-color: white;")
        self.rows_layout = QtWidgets.QVBoxLayout(self.rows_widget)
        self.rows_layout.setSpacing(6)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.addStretch()

        self.scroll_area.setWidget(self.rows_widget)
        outer.addWidget(self.scroll_area, stretch=1)

        # "Add Recording" and "Import from CSV" buttons
        add_import_row = QtWidgets.QHBoxLayout()
        self.add_btn = QtWidgets.QPushButton('+ Add Recording')
        self.add_btn.setStyleSheet(pyfx.dict2ss(QSS.TOGGLE_BTN))
        self.add_btn.clicked.connect(self._add_row)
        import_csv_btn = QtWidgets.QPushButton('Import from CSV')
        import_csv_btn.setStyleSheet(pyfx.dict2ss(QSS.TOGGLE_BTN))
        import_csv_btn.clicked.connect(self._load_from_csv)
        add_import_row.addWidget(self.add_btn)
        add_import_row.addWidget(import_csv_btn)
        outer.addLayout(add_import_row)

        # Bottom buttons
        btn_row = QtWidgets.QHBoxLayout()
        self.cancel_btn = QtWidgets.QPushButton('Cancel')
        self.cancel_btn.setStyleSheet(pyfx.dict2ss(QSS.TOGGLE_BTN))
        self.cancel_btn.clicked.connect(self.reject)

        self.run_btn = QtWidgets.QPushButton('Run All')
        self.run_btn.setStyleSheet(
            pyfx.dict2ss(QSS.TOGGLE_BTN) + "QPushButton { background-color: #c8f0c8; }"
        )
        self.run_btn.clicked.connect(self._run_all)

        btn_row.addWidget(self.cancel_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.run_btn)
        outer.addLayout(btn_row)

        self.setMinimumSize(560, 400)

        # Start with 3 default empty rows
        for _ in range(3):
            self._add_row()

    # ------------------------------------------------------------------
    # Row management
    # ------------------------------------------------------------------

    def _add_row(self):
        if len(self._path_rows) >= self.MAX_ROWS:
            return

        idx = len(self._path_rows)
        num = idx + 1

        frame = QtWidgets.QFrame()
        frame.setStyleSheet(
            "QFrame { background-color: #E8E8E8; border-radius: 4px; }"
        )
        row_layout = QtWidgets.QHBoxLayout(frame)
        row_layout.setContentsMargins(8, 6, 8, 6)
        row_layout.setSpacing(6)

        num_label = QtWidgets.QLabel(f'{num}.')
        num_label.setFixedWidth(28)
        num_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        line_edit = QtWidgets.QLineEdit()
        line_edit.setPlaceholderText('Path to recording folder / file…')

        browse_btn = QtWidgets.QPushButton('Browse')
        browse_btn.setStyleSheet(pyfx.dict2ss(QSS.TOGGLE_BTN))
        browse_btn.setFixedWidth(70)
        browse_btn.clicked.connect(lambda _, le=line_edit: self._browse(le))

        remove_btn = QtWidgets.QPushButton('Remove')
        remove_btn.setStyleSheet(pyfx.dict2ss(QSS.TOGGLE_BTN))
        remove_btn.setFixedWidth(70)
        remove_btn.clicked.connect(lambda _, f=frame: self._remove_row(f))

        row_layout.addWidget(num_label)
        row_layout.addWidget(line_edit, stretch=1)
        row_layout.addWidget(browse_btn)
        row_layout.addWidget(remove_btn)

        # Insert before the trailing stretch
        insert_pos = self.rows_layout.count() - 1
        self.rows_layout.insertWidget(insert_pos, frame)
        self._path_rows.append((frame, line_edit))

        if len(self._path_rows) >= self.MAX_ROWS:
            self.add_btn.setEnabled(False)

    def _remove_row(self, frame):
        for i, (f, le) in enumerate(self._path_rows):
            if f is frame:
                self._path_rows.pop(i)
                self.rows_layout.removeWidget(frame)
                frame.deleteLater()
                break
        self.add_btn.setEnabled(True)
        self._renumber_rows()

    def _renumber_rows(self):
        for i, (frame, _) in enumerate(self._path_rows):
            label = frame.findChild(QtWidgets.QLabel)
            if label:
                label.setText(f'{i + 1}.')

    def _browse(self, line_edit):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Select recording folder', line_edit.text() or os.path.expanduser('~')
        )
        if not path:
            # Also allow file selection for formats like .npy/.mat
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, 'Select recording file', line_edit.text() or os.path.expanduser('~')
            )
        if path:
            line_edit.setText(path)

    def _browse_probe(self):
        """Browse for probe configuration file."""
        fpath, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Select probe configuration file',
            self.probe_le.text() or os.path.expanduser('~'),
            'Probe files (*.json *.prb *.mat)'
        )
        if fpath:
            self.probe_le.setText(fpath)

    def _load_from_csv(self):
        """Import recording paths from a CSV file with a 'Path' column."""
        fpath, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Select CSV file', os.path.expanduser('~'), 'CSV files (*.csv)'
        )
        if not fpath:
            return
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            gi.MsgboxError(f'Could not read CSV:\n{e}', parent=self).exec()
            return
        if 'Path' not in df.columns:
            gi.MsgboxError("CSV must have a 'Path' column.", parent=self).exec()
            return
        paths = df['Path'].dropna().astype(str).tolist()
        if not paths:
            gi.MsgboxError('No paths found in CSV.', parent=self).exec()
            return
        for frame, _ in list(self._path_rows):
            self._remove_row(frame)
        for path in paths:
            self._add_row()
            _, le = self._path_rows[-1]
            le.setText(path)
        self.add_btn.setEnabled(len(self._path_rows) < self.MAX_ROWS)

    def _resolve_trigger_path(self, path):
        """Return trigger file path for InputDataSelectionPopup.
        For NCS folders, auto-selects the first .ncs file found."""
        if os.path.isfile(path):
            return path
        if os.path.isdir(path):
            ncs_files = sorted(f for f in os.listdir(path) if f.lower().endswith('.ncs'))
            if ncs_files:
                return os.path.join(path, ncs_files[0])
            return path  # OpenEphys / NeuroNexus folders are passed as-is
        return None

    # ------------------------------------------------------------------
    # Run all paths sequentially
    # ------------------------------------------------------------------

    def _run_all(self):
        probe_file = self.probe_le.text().strip()
        if not probe_file or not os.path.isfile(probe_file):
            gi.MsgboxError('Please select a probe file before running.', parent=self).exec()
            return

        paths = [le.text().strip() for _, le in self._path_rows if le.text().strip()]
        if not paths:
            gi.MsgboxError('No recording paths entered.', parent=self).exec()
            return

        for path in paths:
            trigger_path = self._resolve_trigger_path(path)
            if trigger_path is None:
                gi.MsgboxError(f'Could not find a valid recording in:\n{path}', parent=self).exec()
                continue
            popup = InputDataSelectionPopup(parent=self, init_path=trigger_path,
                                            bulk_mode=True, bulk_probe_file=probe_file)
            popup.exec()


class BulkChannelSelectionPopup(QtWidgets.QDialog):
    """
    Import a CSV (Path, Fissure Channel, Ripple Channel, Hilus Channel columns)
    and bulk-save event channels for each processed recording without launching the GUI.
    Optional 'Needs processing' column (TRUE/FALSE) filters which rows are processed.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Bulk Channel Selection')
        self._df = None
        self._build_ui()

    def _build_ui(self):
        outer = QtWidgets.QVBoxLayout(self)
        outer.setSpacing(12)
        outer.setContentsMargins(16, 16, 16, 16)

        info_lbl = QtWidgets.QLabel(
            'Select a CSV with columns: Path, Fissure Channel, Ripple Channel, Hilus Channel.\n'
            'Optional: "Needs processing" column (TRUE/FALSE) to filter rows.'
        )
        info_lbl.setWordWrap(True)
        outer.addWidget(info_lbl)

        # CSV file picker row
        csv_frame = QtWidgets.QFrame()
        csv_hlayout = QtWidgets.QHBoxLayout(csv_frame)
        csv_hlayout.setContentsMargins(0, 0, 0, 0)
        csv_lbl = QtWidgets.QLabel('CSV file:')
        csv_lbl.setFixedWidth(60)
        self.csv_le = QtWidgets.QLineEdit()
        self.csv_le.setReadOnly(True)
        self.csv_le.setPlaceholderText('Browse for CSV file...')
        csv_browse_btn = QtWidgets.QPushButton('Browse')
        csv_browse_btn.setStyleSheet(pyfx.dict2ss(QSS.TOGGLE_BTN))
        csv_browse_btn.setFixedWidth(70)
        csv_browse_btn.clicked.connect(self._browse_csv)
        csv_hlayout.addWidget(csv_lbl)
        csv_hlayout.addWidget(self.csv_le, stretch=1)
        csv_hlayout.addWidget(csv_browse_btn)
        outer.addWidget(csv_frame)

        # Status / results area
        self.status_txt = QtWidgets.QTextEdit()
        self.status_txt.setReadOnly(True)
        self.status_txt.setPlaceholderText('Load a CSV file then click "Run" to save channels...')
        self.status_txt.setMinimumHeight(200)
        outer.addWidget(self.status_txt, stretch=1)

        # Bottom buttons
        btn_row = QtWidgets.QHBoxLayout()
        self.close_btn = QtWidgets.QPushButton('Close')
        self.close_btn.setStyleSheet(pyfx.dict2ss(QSS.TOGGLE_BTN))
        self.close_btn.clicked.connect(self.reject)
        self.run_btn = QtWidgets.QPushButton('Run')
        self.run_btn.setStyleSheet(
            pyfx.dict2ss(QSS.TOGGLE_BTN) + "QPushButton { background-color: #c8f0c8; }"
        )
        self.run_btn.clicked.connect(self._run)
        self.run_btn.setEnabled(False)
        btn_row.addWidget(self.close_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.run_btn)
        outer.addLayout(btn_row)

        self.setMinimumSize(520, 400)

    def _browse_csv(self):
        fpath, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Select CSV file', os.path.expanduser('~'), 'CSV files (*.csv)'
        )
        if not fpath:
            return
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            gi.MsgboxError(f'Could not read CSV:\n{e}', parent=self).exec()
            return

        required = {'Path', 'Ripple Channel', 'Fissure Channel', 'Hilus Channel'}
        missing = required - set(df.columns)
        if missing:
            gi.MsgboxError(
                f"CSV is missing required columns: {', '.join(sorted(missing))}", parent=self
            ).exec()
            return

        self.csv_le.setText(fpath)

        if 'Needs processing' in df.columns:
            mask = df['Needs processing'].astype(str).str.strip().str.upper().isin(
                ['TRUE', '1', 'YES']
            )
            n_total = len(df)
            self._df = df[mask].reset_index(drop=True)
            self.status_txt.setText(
                f'Loaded {n_total} rows; {len(self._df)} marked for processing.\n'
                'Click "Run" to save channels.'
            )
        else:
            self._df = df.dropna(subset=['Path']).reset_index(drop=True)
            self.status_txt.setText(
                f'Loaded {len(self._df)} rows (no "Needs processing" column — using all rows).\n'
                'Click "Run" to save channels.'
            )

        self.run_btn.setEnabled(len(self._df) > 0)

    def _run(self):
        if self._df is None or len(self._df) == 0:
            return

        lines = []
        n_ok = n_skip = n_err = 0

        for _, row in self._df.iterrows():
            raw_path = str(row['Path']).strip()
            toothy_ddir = str(Path(raw_path, 'toothy'))

            if not dp.validate_processed_ddir(toothy_ddir):
                lines.append(
                    f'[SKIP]  {raw_path}\n'
                    '        → No valid processed data in toothy subfolder'
                )
                n_skip += 1
                continue

            try:
                # CSV holds display values (1-indexed, matching what the GUI spinboxes show).
                # Subtract 1 to get the 0-indexed values that are stored in the .npy file.
                fissure_disp = int(row['Fissure Channel'])
                ripple_disp  = int(row['Ripple Channel'])
                hil_disp     = int(row['Hilus Channel'])
                fissure_chan = fissure_disp - 1
                ripple_chan  = ripple_disp - 1
                hil_chan     = hil_disp - 1
            except (ValueError, TypeError) as e:
                lines.append(f'[ERROR] {raw_path}\n        → Invalid channel values: {e}')
                n_err += 1
                continue

            try:
                # new_channels order: [theta/fissure, ripple, hilus]
                ephys.save_event_channels(
                    toothy_ddir, iprb=0, ishank=0,
                    new_channels=[fissure_chan, ripple_chan, hil_chan]
                )
                # Create DS_DF_probe0-shank0 so "Classify dentate spikes" becomes enabled
                self._create_ds_df(toothy_ddir, hil_chan)
                lines.append(
                    f'[OK]    {raw_path}\n'
                    f'        → Fissure={fissure_disp}, Ripple={ripple_disp}, Hilus={hil_disp}'
                )
                n_ok += 1
            except Exception as e:
                lines.append(f'[ERROR] {raw_path}\n        → Save failed: {e}')
                n_err += 1

        summary = f'Done: {n_ok} saved, {n_skip} skipped, {n_err} errors\n\n'
        self.status_txt.setText(summary + '\n'.join(lines))
        self.run_btn.setEnabled(False)

    def _create_ds_df(self, toothy_ddir, hil_chan):
        """Create DS_DF_probe0-shank0 from DATA.hdf5 so the classification GUI can be opened."""
        hdf5_path = Path(toothy_ddir, 'DATA.hdf5')
        if not os.path.isfile(hdf5_path):
            return  # Older npz-format recording; skip
        probe = ephys.read_probe_group(toothy_ddir).probes[0]
        DF_ALL = pd.read_hdf(str(hdf5_path), key=ephys.get_h5_key('ALL_DS', iprb=0))
        STD    = pd.read_hdf(str(hdf5_path), key=ephys.get_h5_key('STD',    iprb=0))
        DS_ALL = ephys.clean_event_df(DF_ALL, STD, probe)
        if hil_chan in DS_ALL.index:
            DS_DF = DS_ALL.loc[[hil_chan]]  # double brackets → always a DataFrame
            DS_DF = DS_DF[DS_DF['is_valid'] == 1].reset_index(drop=True)
        else:
            DS_DF = pd.DataFrame(columns=DS_ALL.columns)
        DS_DF.to_csv(Path(toothy_ddir, 'DS_DF_probe0-shank0'), index_label=False)


def _headless_ds_classify(ddir, iprb=0, ishank=0, bad_channels=None, progress_fn=None):
    """
    Run DS CSD calculation + PCA classification without the GUI.
    Replicates DS_CSDWindow.calculate_csd() + save_csd() using default settings.
    bad_channels: list of absolute 0-indexed probe channel numbers to mark as noisy.
    progress_fn: optional callable(str) for live step-by-step progress reporting.
    Saves results to CSDs.hdf5 and DS_DF_probe{iprb}-shank{ishank}.csv.
    Returns (ds1_n, ds2_n).
    """
    from copy import deepcopy
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans, DBSCAN
    import quantities as pq

    def _step(msg):
        if progress_fn is not None:
            progress_fn(msg)

    # Probe / shank geometry
    probe = ephys.read_probe_group(ddir).probes[iprb]
    shank = probe.get_shanks()[ishank]
    ypos = np.array(sorted(shank.contact_positions[:, 1]))
    coord_electrode = pq.Quantity(ypos, probe.si_units).rescale('m', dtype='float32')
    shank_channels = shank.get_indices()
    channels = np.arange(len(shank_channels), dtype='int')

    # Relative index of the fissure/theta channel — CSD window starts here
    event_channels = ephys.load_event_channels(ddir, iprb, ishank=ishank)
    if event_channels is None or None in event_channels:
        raise ValueError('Event channels not set — run bulk channel selection first')
    rel_theta_chan = list(shank_channels).index(event_channels[0])

    # Load LFP and noise annotation
    dmode = dp.validate_processed_ddir(ddir)
    FF = None
    _step(f'  Loading LFP data ({len(shank_channels)} channels)...')
    if dmode == 1:
        FF = h5py.File(str(Path(ddir, 'DATA.hdf5')), 'r+')
        lfp_fs = int(FF.attrs['lfp_fs'])
        lfp_all = ephys.load_h5_lfp(FF, key='raw', iprb=iprb)[shank_channels, :]
        NOISE_ALL = ephys.load_h5_array(FF, 'NOISE', iprb, in_memory=True)
        NOISE_TRAIN = NOISE_ALL[shank_channels].copy()
    elif dmode == 2:
        lfp_all = ephys.load_bp(ddir, key='raw', iprb=iprb)[shank_channels, :]
        lfp_fs = int(np.load(Path(ddir, 'lfp_fs.npy')))
        NOISE_ALL = ephys.load_noise_channels(ddir, iprb=iprb).copy()
        NOISE_TRAIN = NOISE_ALL[shank_channels].copy()
    else:
        raise ValueError('No valid processed data found')
    _step(f'  LFP loaded: {lfp_all.shape[0]} ch × {lfp_all.shape[1]:,} samples  '
          f'({lfp_fs} Hz,  {lfp_all.shape[1]/lfp_fs/60:.1f} min)')

    # Apply bad channels from the CSV to the noise annotation and persist them
    if bad_channels:
        newly_marked = []
        for ch in bad_channels:
            if ch in shank_channels and not NOISE_ALL[ch]:
                NOISE_ALL[ch] = 1
                NOISE_TRAIN[list(shank_channels).index(ch)] = 1
                newly_marked.append(ch)
        if newly_marked:
            _step(f'  Marking {len(newly_marked)} bad channel(s) as noisy: {newly_marked}')
            if dmode == 1:
                FF[f'{iprb}']['NOISE'][:] = NOISE_ALL
            else:
                noise_list = list(np.load(Path(ddir, 'noise_channels.npy'), allow_pickle=True))
                noise_list[iprb] = NOISE_ALL
                np.save(Path(ddir, 'noise_channels.npy'), noise_list, allow_pickle=True)
        already = [ch for ch in bad_channels if ch in shank_channels and ch not in newly_marked]
        if already:
            _step(f'  Already noisy (skipped): {already}')

    # Interpolate noisy channels (mirrors DS_CSDWindow.interp_data)
    noise_idx = np.nonzero(NOISE_TRAIN)[0]
    clean_idx = np.setdiff1d(channels, noise_idx)
    if len(noise_idx):
        _step(f'  Interpolating {len(noise_idx)} noisy channel(s): '
              f'{list(shank_channels[noise_idx])}')
    lfp_interp = deepcopy(lfp_all)
    for i in noise_idx:
        if len(clean_idx) == 0:
            lfp_interp[i] = np.zeros_like(lfp_interp[i])
        elif i < int(min(clean_idx)):
            lfp_interp[i] = lfp_all[int(min(clean_idx))]
        elif i > int(max(clean_idx)):
            lfp_interp[i] = lfp_all[int(max(clean_idx))]
        else:
            s1 = lfp_all[int(pyfx.Closest(i, clean_idx[clean_idx < i]))]
            s2 = lfp_all[int(pyfx.Closest(i, clean_idx[clean_idx > i]))]
            lfp_interp[i] = np.nanmean([s1, s2], axis=0)

    # Load DS events and recording params
    DS_DF = ephys.load_ds_dataset(ddir, iprb, ishank)
    if DS_DF is None or len(DS_DF) < 2:
        raise ValueError(f'Too few DS events ({0 if DS_DF is None else len(DS_DF)}) — cannot classify')
    DS_DF = DS_DF.copy()
    iev = np.atleast_1d(DS_DF.idx.values)
    ddict = dict(ephys.load_recording_params(ddir))
    algo = str(ddict.get('clus_algo', 'kmeans'))
    _step(f'  DS dataset: {len(iev):,} valid events  |  '
          f'event channels — fissure: {event_channels[0]+1}, '
          f'ripple: {event_channels[1]+1}, hilus: {event_channels[2]+1}')

    # CSD channel window: fissure/theta channel → bottom of shank (GUI default)
    csd_chs = np.arange(rel_theta_chan, len(channels))

    def _csd(idx):
        data = lfp_interp[csd_chs, :][:, idx]
        obj = ephys.get_csd_obj(data, coord_electrode[csd_chs], ddict)
        return ephys.csd_obj2arrs(obj)

    _step(f'  Computing CSD  ({len(csd_chs)} channels, {len(iev):,} events)...')
    raw_csd, filt_csd, norm_filt_csd = _csd(iev)

    _step(f'  PCA + {algo.upper()} clustering...')
    pca_fit = PCA(n_components=2).fit_transform(norm_filt_csd.T)

    def _set_ds_type(types):
        rows1, rows2 = np.where(types == 1)[0], np.where(types == 2)[0]
        if not len(rows1) or not len(rows2):
            return types
        c1 = _csd(DS_DF.idx.values[rows1])[2]
        c2 = _csd(DS_DF.idx.values[rows2])[2]
        if np.argmin(np.nanmean(c1, axis=1)) > np.argmin(np.nanmean(c2, axis=1)):
            types[rows1] = 2
            types[rows2] = 1
        return types

    km = KMeans(n_clusters=int(ddict.get('nclusters', 2)), n_init='auto').fit(pca_fit)
    db = DBSCAN(eps=float(ddict.get('eps', 0.2)),
                min_samples=int(ddict.get('min_clus_samples', 3))).fit(pca_fit)
    km_types = _set_ds_type(np.array([{0: 2, 1: 1}.get(x, 0) for x in km.labels_]))
    db_types = _set_ds_type(np.array([{0: 1, 1: 2}.get(x, 0) for x in db.labels_]))

    dstypes = km_types if algo == 'kmeans' else db_types
    DS_DF[['pc1', 'pc2']] = pca_fit
    DS_DF[['k_type', 'db_type', 'type']] = np.array([km_types, db_types, dstypes]).T

    ds1_n = int((DS_DF['type'] == 1).sum())
    ds2_n = int((DS_DF['type'] == 2).sum())

    _step(f'  Saving CSDs.hdf5 and DS_DF CSV...')
    hdf5_path = str(Path(ddir, 'CSDs.hdf5'))
    K = f'/{iprb}/{ishank}'
    scalar_params = {k: v for k, v in ddict.items()
                     if isinstance(v, (int, float, bool, str, np.integer, np.floating))}
    with h5py.File(hdf5_path, 'a') as gg:
        if K in gg:
            del gg[K]
        for key, arr in dict(raw_csd=raw_csd, filt_csd=filt_csd,
                              norm_filt_csd=norm_filt_csd, csd_chs=csd_chs).items():
            gg[f'{K}/{key}'] = arr
        gg[K].attrs.update(scalar_params)
    DS_DF.to_hdf(hdf5_path, key=f'{K}/DS_DF', mode='a')
    DS_DF.to_csv(str(Path(ddir, f'DS_DF_probe{iprb}-shank{ishank}')), index_label=False)

    if FF is not None:
        FF.close()

    return ds1_n, ds2_n


class BulkResultsPopup(QtWidgets.QDialog):
    """
    Import a CSV (same feeder-sheet format) and run headless DS classification
    for each recording: calculates CSD + PCA, saves results, and writes a
    summary CSV with DS Type 1 / Type 2 counts.
    Ask for the output CSV path before running.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Step 3: Bulk DS Classification')
        self._df = None
        self._build_ui()

    def _build_ui(self):
        outer = QtWidgets.QVBoxLayout(self)
        outer.setSpacing(12)
        outer.setContentsMargins(16, 16, 16, 16)

        info_lbl = QtWidgets.QLabel(
            'Select a CSV with a "Path" column (and optionally "Session ID" and\n'
            '"Needs processing"). Then choose where to save the results summary\n'
            'before clicking Run.'
        )
        info_lbl.setWordWrap(True)
        outer.addWidget(info_lbl)

        # Input CSV row
        csv_frame = QtWidgets.QFrame()
        csv_hl = QtWidgets.QHBoxLayout(csv_frame)
        csv_hl.setContentsMargins(0, 0, 0, 0)
        csv_lbl = QtWidgets.QLabel('Input CSV:')
        csv_lbl.setFixedWidth(80)
        self.csv_le = QtWidgets.QLineEdit()
        self.csv_le.setReadOnly(True)
        self.csv_le.setPlaceholderText('Browse for feeder-sheet CSV...')
        csv_btn = QtWidgets.QPushButton('Browse')
        csv_btn.setStyleSheet(pyfx.dict2ss(QSS.TOGGLE_BTN))
        csv_btn.setFixedWidth(70)
        csv_btn.clicked.connect(self._browse_csv)
        csv_hl.addWidget(csv_lbl)
        csv_hl.addWidget(self.csv_le, stretch=1)
        csv_hl.addWidget(csv_btn)
        outer.addWidget(csv_frame)

        # Output CSV row
        out_frame = QtWidgets.QFrame()
        out_hl = QtWidgets.QHBoxLayout(out_frame)
        out_hl.setContentsMargins(0, 0, 0, 0)
        out_lbl = QtWidgets.QLabel('Output CSV:')
        out_lbl.setFixedWidth(80)
        self.out_le = QtWidgets.QLineEdit()
        self.out_le.setReadOnly(True)
        self.out_le.setPlaceholderText('Choose where to save the results summary...')
        out_btn = QtWidgets.QPushButton('Browse')
        out_btn.setStyleSheet(pyfx.dict2ss(QSS.TOGGLE_BTN))
        out_btn.setFixedWidth(70)
        out_btn.clicked.connect(self._browse_output)
        out_hl.addWidget(out_lbl)
        out_hl.addWidget(self.out_le, stretch=1)
        out_hl.addWidget(out_btn)
        outer.addWidget(out_frame)

        # Status / results text
        self.status_txt = QtWidgets.QTextEdit()
        self.status_txt.setReadOnly(True)
        self.status_txt.setPlaceholderText(
            'Load a CSV and choose an output path, then click "Run"...'
        )
        self.status_txt.setMinimumHeight(220)
        outer.addWidget(self.status_txt, stretch=1)

        # Bottom buttons
        btn_row = QtWidgets.QHBoxLayout()
        self.close_btn = QtWidgets.QPushButton('Close')
        self.close_btn.setStyleSheet(pyfx.dict2ss(QSS.TOGGLE_BTN))
        self.close_btn.clicked.connect(self.reject)
        self.run_btn = QtWidgets.QPushButton('Run')
        self.run_btn.setStyleSheet(
            pyfx.dict2ss(QSS.TOGGLE_BTN) + "QPushButton { background-color: #c8f0c8; }"
        )
        self.run_btn.clicked.connect(self._run)
        self.run_btn.setEnabled(False)
        btn_row.addWidget(self.close_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.run_btn)
        outer.addLayout(btn_row)

        self.setMinimumSize(560, 460)

    def _browse_csv(self):
        fpath, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Select CSV file', os.path.expanduser('~'), 'CSV files (*.csv)'
        )
        if not fpath:
            return
        try:
            df = pd.read_csv(fpath)
        except Exception as e:
            gi.MsgboxError(f'Could not read CSV:\n{e}', parent=self).exec()
            return
        if 'Path' not in df.columns:
            gi.MsgboxError("CSV must have a 'Path' column.", parent=self).exec()
            return
        self.csv_le.setText(fpath)
        if 'Needs processing' in df.columns:
            mask = df['Needs processing'].astype(str).str.strip().str.upper().isin(
                ['TRUE', '1', 'YES']
            )
            n_total = len(df)
            self._df = df[mask].reset_index(drop=True)
            self.status_txt.setText(
                f'Loaded {n_total} rows; {len(self._df)} marked for processing.'
            )
        else:
            self._df = df.dropna(subset=['Path']).reset_index(drop=True)
            self.status_txt.setText(
                f'Loaded {len(self._df)} rows (no "Needs processing" column — using all rows).'
            )
        self._update_run_btn()

    def _browse_output(self):
        fpath, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save results CSV', os.path.expanduser('~'), 'CSV files (*.csv)'
        )
        if fpath:
            if not fpath.lower().endswith('.csv'):
                fpath += '.csv'
            self.out_le.setText(fpath)
        self._update_run_btn()

    def _update_run_btn(self):
        self.run_btn.setEnabled(
            self._df is not None and len(self._df) > 0 and bool(self.out_le.text().strip())
        )

    def _log(self, text):
        self.status_txt.append(text)
        QtWidgets.QApplication.processEvents()

    def _run(self):
        if self._df is None or not self.out_le.text().strip():
            return

        out_path = self.out_le.text().strip()
        self.status_txt.clear()
        self.run_btn.setEnabled(False)
        results = []
        n_ok = n_skip = n_err = 0
        n_total = len(self._df)

        for i, (_, row) in enumerate(self._df.iterrows()):
            raw_path = str(row['Path']).strip()
            sid_val = row.get('Session ID', None)
            session_id = str(sid_val) if pd.notna(sid_val) else os.path.basename(raw_path)
            toothy_ddir = str(Path(raw_path, 'toothy'))

            self._log(f'\n[{i+1}/{n_total}] {session_id}\n  {raw_path}')

            # Parse Bad Channels column → list of 0-indexed absolute channel numbers
            bad_channels = []
            bc_val = row.get('Bad Channels', None)
            if pd.notna(bc_val) and str(bc_val).strip():
                try:
                    bad_channels = [int(x.strip()) for x in str(bc_val).split(',')]
                except ValueError:
                    pass
            if bad_channels:
                self._log(f'  Bad channels from CSV: {bad_channels}')

            if not dp.validate_classification_ddir(toothy_ddir, 0, 0):
                self._log(
                    '[SKIP] Not ready for classification (run Bulk channel selection first?)'
                )
                n_skip += 1
                continue

            try:
                ds1_n, ds2_n = _headless_ds_classify(
                    toothy_ddir, iprb=0, ishank=0,
                    bad_channels=bad_channels or None,
                    progress_fn=self._log,
                )
                self._log(f'[OK]  DS Type 1 N = {ds1_n},  DS Type 2 N = {ds2_n}')
                results.append({
                    'Session ID':  session_id,
                    'Path':        raw_path,
                    'DS Type 1 N': ds1_n,
                    'DS Type 2 N': ds2_n,
                })
                n_ok += 1
            except Exception as e:
                self._log(f'[ERROR] {e}')
                n_err += 1

        self._log(f'\nDone: {n_ok} classified, {n_skip} skipped, {n_err} errors')

        if results:
            pd.DataFrame(results).to_csv(out_path, index=False)
            self._log(f'Summary saved → {out_path}')


if __name__ == '__main__':
    app = pyfx.qapp()
    qfd = QtWidgets.QFileDialog()
    init_ddir = str(qfd.directory().path())
    if init_ddir == os.getcwd():
        init_ddir=None
    w = InputDatasSelectionPopup()
    w.show()
    w.raise_()
    sys.exit(app.exec())