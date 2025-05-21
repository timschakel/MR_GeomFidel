#!/usr/bin/env python
"""
MR_GeometricFidelity.py
This module provides analysis tools for evaluating the geometric fidelity of MR images using a phantom. 
It is designed to be used as part of the WADQC software suite for quality assurance in radiotherapy imaging.
Classes and Functions:
----------------------
- applyFilter_createNewModuleData(data, filter_value):
    Filters DICOM series by SeriesDescription, copies the filtered series to a new directory, and creates a new ModuleData object.
- acqdatetime_series(data, results, action):
    Extracts and records the acquisition date and time from the DICOM series.
- MRL_geomfidel_analysis(data, results, action):
    Performs geometric fidelity analysis for MRL (MR-Linac) data using the GeomAcc class.
- MRL_geomfidel_2shift_analysis(data, results, action):
    Performs geometric fidelity analysis for MRL data acquired at two different fat shift directions, averages marker positions, and computes statistics.
- MR_geomfidel_analysis(data, results, action):
    Performs geometric fidelity analysis for MRSim data using the GeomAcc class.
- dataVerification(data, actions):
    Placeholder function for verifying the integrity and suitability of the input data.
Usage:
------
This script is intended to be run as a standalone module. It processes MR DICOM data, detects phantom markers, 
computes geometric deviations, and outputs results and images for quality assurance purposes.
Changelog:
----------
- 20210623: Initial version
- 20210820: Removed obsolete code
- 20250507: Refactored for MRSim data, added phantom types, and improved processing for different table positions.
Author:
-------
Tim Schakel, UMC Utrecht, 2025
References:
-----------
- WADQC Software: https://bitbucket.org/MedPhysNL/wadqc/wiki/Home

"""

from wad_qc.module import pyWADinput
from wad_qc.module import ModuleData
from wad_qc.modulelibs import wadwrapper_lib
import pydicom
import GeomAcc
import numpy as np
import os
import shutil

def applyFilter_createNewModuleData(data, filter_value):
    """
    Filters DICOM series in the provided data object by SeriesDescription, copies the filtered series to a new directory,
    and creates a new ModuleData object from the filtered data.
    Args:
        data: An object containing a 'series_filelist' attribute, which is a list of lists of DICOM file paths.
        filter_value (str): The SeriesDescription value to filter DICOM series by.
    Returns:
        tuple:
            - filtered_data: A new ModuleData object created from the filtered and copied DICOM series.
            - datadir (str): The path to the directory containing the filtered DICOM data.
    Side Effects:
        - Copies the filtered DICOM series to a new directory under a predefined path.
        - Prints status messages during processing.
    Notes:
        - Assumes the existence of the 'ModuleData' class and required imports (e.g., pydicom, shutil, os).
        - The function expects a specific folder structure for ModuleData.
        - Only the first matching series directory is copied.
    """
    print("Filtering data and creating new ModuleData object for: ", filter_value)
    filtered_series = []
    filtered_paths = []
    dataflatten = [item for sublist in data.series_filelist for item in sublist]
    # Iterate through all series in the data object
    for series in dataflatten:
        # Load the first DICOM file in the series
        dcm = pydicom.dcmread(series, stop_before_pixels=True)
        
        # Check if the SeriesDescription matches the filter value
        if dcm.SeriesDescription == filter_value:
            acqname = dcm.SeriesDescription
            studyid = dcm.StudyInstanceUID
            filtered_series.append(series)
            filtered_paths.append(os.path.dirname(series))

    # Create a new ModuleData object with the filtered series
    unique_entries = list(dict.fromkeys(filtered_paths))
    
    # Need to move the selected files to a new location
    # ModuleData expects a certain folder structure (slow and expensive way to do it...)
    #wadpath = '/smb/user/tschakel/DS-Data/Radiotherapie/Research/User/tschakel/projects/wadqc/QAtests/MR_GeomFidel/data/'
    wadpath = '/home/waduser/data/'
    dname = str(studyid)+'_'+acqname
    destdir = wadpath+dname+'/DICOM'
    datadir = wadpath+dname
    shutil.copytree(unique_entries[0],destdir)
    
    filtered_data = ModuleData(wadpath+dname)

    return filtered_data, datadir

def acqdatetime_series(data, results, action):
    """
    Extracts the acquisition date and time from a DICOM series and adds it to the results.

    Parameters:
        data: An object containing the DICOM series file list. Expects `data.series_filelist` to be a list of lists, where the first element is the file path to a DICOM file.
        results: An object with an `addDateTime` method to store the extracted acquisition date and time.
        action: Unused parameter, included for interface compatibility.

    Side Effects:
        - Reads the first DICOM file in the series to extract acquisition date and time.
        - Adds the extracted date and time to the `results` object under the key 'AcquisitionDateTime'.
        - Prints the acquisition date and time to standard output.
    """
    dcmInfile = pydicom.read_file(data.series_filelist[0][0], stop_before_pixels=True)
    dt = wadwrapper_lib.acqdatetime_series(dcmInfile)
    results.addDateTime('AcquisitionDateTime', dt)
    print('Acquisition date and time is: ', dt)

def MRL_geomfidel_analysis(data, results, action):
    """
    Performs geometric fidelity analysis for MRL (Magnetic Resonance Linear accelerator) data.

    This function initializes a geometric accuracy analysis object for MRL, processes the input data to detect and correct marker positions, 
    and then collects 3D metric results and saves relevant images.

    Args:
        data: Input data required for geometric fidelity analysis, typically containing imaging series.
        results: A data structure to store the results of the analysis and generated images.
        action: Specifies the action or mode for the analysis (not used directly in this function).

    Returns:
        None. Results are stored in the provided `results` object and images are saved as side effects.

    Raises:
        Any exceptions raised by the underlying GeomAcc methods.
    """
    print("Performing MRL geometric fidelity analysis...")

    GeoFid = GeomAcc.GeomAcc('MRL')
    GeoFid.loadSeriesAndDetectMarkers(data)
    GeoFid.correctDetectedMarkerPositionsForSetup()

    GeoFid.collectResults3Dmetrics(results)
    GeoFid.createAndSaveImages(results)

def MRL_geomfidel_2shift_analysis(data, results, action):
    """
    Performs geometric fidelity analysis for MR-Linac (MRL) using two shifted datasets.
    This function filters the input data based on two shift descriptions (A and P), detects and corrects marker positions
    in both datasets, matches markers present in both scans, averages their detected positions, and computes geometric
    accuracy metrics. The results are collected and images are generated for reporting. Temporary directories created
    during filtering are cleaned up at the end.
    Args:
        data: The input dataset containing MR images and associated metadata.
        results: An object or structure to store the computed results and generated images.
        action: A dictionary containing filter descriptions for the two shifts under `action["filters"]` with keys
            "shiftA_description" and "shiftP_description".
    Returns:
        None. Results are stored in the provided `results` object.
    Side Effects:
        - Modifies the `results` object with geometric fidelity metrics and images.
        - Creates and deletes temporary directories during processing.
    """
    print("Performing MRL geometric fidelity 2-shift analysis...")
        
    shiftA_description = action["filters"]["shiftA_description"]
    shiftP_description = action["filters"]["shiftP_description"]
    filtered_data_A, datadirA = applyFilter_createNewModuleData(data, shiftA_description)
    filtered_data_P, datadirP = applyFilter_createNewModuleData(data, shiftP_description)

    # Initialize GeomAcc class for A
    GeoFidA = GeomAcc.GeomAcc('MRL')
    GeoFidA.loadSeriesAndDetectMarkers(filtered_data_A)
    GeoFidA.correctDetectedMarkerPositionsForSetup()
    
    # Initialize GeomAcc class for P
    GeoFidP = GeomAcc.GeomAcc('MRL')
    GeoFidP.loadSeriesAndDetectMarkers(filtered_data_P)
    GeoFidP.correctDetectedMarkerPositionsForSetup()
    
    # Get detected positions
    detectedA = GeoFidA.correctedMarkerPositions
    detectedP = GeoFidP.correctedMarkerPositions
    
    # Only match on Markers located in both scans
    closest_idxA = GeoFidA.closestExpectedMarkerIndices
    closest_idxP = GeoFidP.closestExpectedMarkerIndices
    
    # indices in the expectedMarkerLocations list in both scans
    common_idx = list(set(closest_idxA).intersection(set(closest_idxP)))
    
    detected_common_idx_A = []
    detected_common_idx_P = []
    for index in common_idx:
        detected_common_idx_A.append(np.argwhere(index == closest_idxA)[0, 0])
        detected_common_idx_P.append(np.argwhere(index == closest_idxP)[0, 0])
        
    detectedA_common = detectedA[detected_common_idx_A, :]
    detectedP_common = detectedP[detected_common_idx_P, :]
    
    # average common marker detections
    detected = (detectedA_common + detectedP_common)/2
    
    # Overwrite the detected markerpositions in the GeomAcc object and get statistics
    GeoFidA.correctedMarkerPositions = detected
    GeoFidA.closestExpectedMarkerIndices = GeoFidA.closestLocations(GeoFidA.correctedMarkerPositions,GeoFidA.expectedMarkerPositions)
    GeoFidA.differencesCorrectedExpected = GeoFidA.getdifferences(GeoFidA.correctedMarkerPositions,GeoFidA.expectedMarkerPositions)

    GeoFidA.collectResults3Dmetrics(results)
    GeoFidA.createAndSaveImages(results)
    shutil.rmtree(datadirA) #remove temporary directories again
    shutil.rmtree(datadirP) #remove temporary directories again


def MR_geomfidel_analysis(data, results, action):
    """
    Performs MR geometric fidelity analysis using the GeomAcc module.

    This function initializes a GeomAcc object for MR simulation, loads the provided data series,
    detects markers, applies setup corrections to detected marker positions, collects 3D metric results,
    and generates and saves relevant images.

    Args:
        data: Input data required for marker detection and analysis.
        results: Object or structure to store the analysis results and generated images.
        action: Specifies the action or mode for the analysis (usage depends on implementation).

    Returns:
        None. Results are stored in the provided 'results' object and images are saved as side effects.
    """
    print("Performing MR geometric fidelity analysis...")

    GeoFid = GeomAcc.GeomAcc('MRSim')
    GeoFid.loadSeriesAndDetectMarkers(data)
    GeoFid.correctDetectedMarkerPositionsForSetup()

    GeoFid.collectResults3Dmetrics(results)
    GeoFid.createAndSaveImages(results)
    
def dataVerification(data, actions):
    # Placeholder for data verification
    # For now rely on the selectors to only call this module with the right data
    # In the future we could add some checks here
    print("Performing data verification...")
    return True


#### main function
if __name__ == "__main__":
    data, results, config = pyWADinput()

    print(config)
    
    # Log which series are found
    data_series = data.getAllSeries()
    print("The following series are found:")
    for item in data_series:
        print(item[0]["SeriesDescription"].value+" with "+str(len(item))+" instances")
        
    for name,action in config['actions'].items():

        if name == 'acqdatetime':
            acqdatetime_series(data, results, action)

        elif name == 'MRL_geomfidel_analysis':
            MRL_geomfidel_analysis(data, results, action)
            
        elif name == 'MRL_geomfidel_2shift_analysis':
            MRL_geomfidel_2shift_analysis(data, results, action)
        
        if name == 'MR_geomfidel_analysis':
            if dataVerification(data, config['actions']):
                MR_geomfidel_analysis(data, results, action)

    results.write()