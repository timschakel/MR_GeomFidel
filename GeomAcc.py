#!/usr/bin/env python

# Tim Schakel, UMC Utrecht, 2025
# This code is developed to be used as part of an analysis module within the WADQC software
# The GeomAcc class expects MR data from the geometric fidelity phantom
# It performs marker localization and gives some statistics on the deviations
# 
# It has been adapted from the original code of Erik van der Bijl & Stijn van de Schoot
# The WAD Software can be found on https://bitbucket.org/MedPhysNL/wadqc/wiki/Home
#  
#
# Changelog:
#
# 20210623: initial version
# 20210820: remove obsolete code
# 20250507: Refactoring to make it suitable for MRSim data
#           - Add different phantom types with different defaults
#           - Add MRSim specific code (handling data from different table positions)
#           - Largely same processing/analysis/plotting code for both phantom types

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

import numpy as np
from scipy import optimize
from scipy.spatial.distance import cdist
from scipy.cluster.vq import kmeans2
from scipy.ndimage.filters import gaussian_filter
from multiprocessing import Pool
from datetime import datetime
import pydicom

class GeomAcc():

    def __init__(self, phantomType):
        """
        Initializes the GeomAcc class with the specified phantom type and sets up
        default configurations, constants, and properties for geometric accuracy 
        analysis.
        Args:
            phantomType (str): The type of phantom being used. Supported values are:
                - 'MRL': Loads MRL phantom defaults.
                - 'MRSim': Loads MRSim phantom defaults.
                - Other: Issues a warning for unknown phantom type.
        Attributes:
            phantomType (str): The type of phantom being used.
            LR (float): Left-Right default constant from the phantom defaults.
            AP (float): Anterior-Posterior default constant from the phantom defaults.
            CC (float): Cranial-Caudal default constant from the phantom defaults.
            studyDate (str): The date of the study (initialized as None).
            studyTime (str): The time of the study (initialized as None).
            studyScanner (str): The scanner used for the study (initialized as None).
            rigid_transformation_setup (list): Initial rigid transformation setup.
            measurementsPerTablePos (dict): Measurements per table position.
            measurementTablePositions (list): List of table positions for measurements.
            NEMA_PAIRS (np.ndarray): NEMA pairs for AP and LR directions.
            CENTER_POSITIONS (np.ndarray): Center positions for AP and LR directions.
            BODY_CTR_POSITIONS (np.ndarray): Body center positions for AP and LR directions.
            ALL_POSITIONS (np.ndarray): All marker positions for AP and LR directions.
            TRANSVERSE_ORIENTATION (bool): Transverse orientation configuration.
            MARKER_THRESHOLD_AFTER_FILTERING (float): Marker threshold after filtering.
            CLUSTERSIZE (int): Cluster size for marker detection.
            LIMIT_CC_SEPARATION_FROM_CC_POSITION (float): Limit for CC separation.
            GAUSSIAN_LAPLACE_SIGMA (float): Sigma value for Gaussian Laplace filter.
            FILTER_CUTOFF_PERCENTILE (float): Filter cutoff percentile.
            LIMIT_Z_SEPARATION_FROM_TABLEPOSITION (float): Limit for Z separation from table position.
            LIMIT_squaredDistanceToCurrentPoint (float): Limit for squared distance to current point.
            ELLIPSOID_FACTOR (float): Ellipsoid factor for marker detection.
            positions_CC (np.ndarray): Sorted marker positions in the CC direction.
            positions_AP_LR (np.ndarray): Marker positions in AP and LR directions.
            degLimit (float): Degree limit for fitting.
            transLimit (float): Translation limit for fitting.
            NUMBEROFINSTANCESPERSERIES (int): Number of instances per series (MRSim only).
            PHANTOM_OFFSET (float): Phantom offset (MRSim only).
            expectedMarkerPositions (np.ndarray): Expected marker positions.
            tablePosition (float): Current table position.
            detectedMarkerPositions (np.ndarray): Detected marker positions (initialized as None).
            correctedMarkerPositions (np.ndarray): Corrected marker positions (initialized as None).
            closestExpectedMarkerIndices (np.ndarray): Indices of closest expected markers (initialized as None).
            differencesCorrectedExpected (np.ndarray): Differences between corrected and expected positions (initialized as None).
        """
        
        self.phantomType = phantomType
        if self.phantomType == 'MRL':
            import GeomAccDefaultsMRL as GeomAccDefaults
            print("-GeomAcc - __init__ -- Load MRL phantom defaults.")
        elif self.phantomType == 'MRSim':
            import GeomAccDefaultsMRSim as GeomAccDefaults
            print("-GeomAcc - __init__ -- Load MRSim phantom defaults.")
        else:
            print("-GeomAcc - __init__ -- WARNING: Unknown phantom type.")

        #Define default constants
        self.LR = GeomAccDefaults.LR
        self.AP = GeomAccDefaults.AP
        self.CC = GeomAccDefaults.CC

        #Properties of the study
        self.studyDate = None
        self.studyTime = None
        self.studyScanner = None

        #results for this study
        self.rigid_transformation_setup = [0, 0, 0, 0, 0, 0]
        self.measurementsPerTablePos = {}
        self.measurementTablePositions = []

        #Phantom type dependent default values
        self.NEMA_PAIRS = np.array(GeomAccDefaults.NEMA_PAIRS_AP_LR)
        self.CENTER_POSITIONS = np.array(GeomAccDefaults.CENTER_POSITIONS_AP_LR)
        self.BODY_CTR_POSITIONS = np.array(GeomAccDefaults.BODY_CTR_POSITIONS_AP_LR)
        self.ALL_POSITIONS = np.array(GeomAccDefaults.markerPositions_AP_LR,dtype=int)

        #Constants/Config

        # check which of these are used for both MRL and MRSim
        self.TRANSVERSE_ORIENTATION = GeomAccDefaults.TRANSVERSEORIENTATION
        self.MARKER_THRESHOLD_AFTER_FILTERING = GeomAccDefaults.MARKER_THRESHOLD_AFTER_FILTERING
        self.CLUSTERSIZE = GeomAccDefaults.CLUSTERSIZE
        self.LIMIT_CC_SEPARATION_FROM_CC_POSITION=GeomAccDefaults.LIMIT_CC_SEPARATION_FROM_CC_POSITION
        self.GAUSSIAN_LAPLACE_SIGMA = GeomAccDefaults.GAUSSIAN_LAPLACE_SIGMA
        self.FILTER_CUTOFF_PERCENTILE = GeomAccDefaults.FILTER_CUTOFF_PERCENTILE
        self.LIMIT_Z_SEPARATION_FROM_TABLEPOSITION = GeomAccDefaults.LIMIT_Z_SEPARATION_FROM_TABLEPOSITION
        self.LIMIT_squaredDistanceToCurrentPoint = GeomAccDefaults.LIMIT_squaredDistanceToCurrentPoint
        self.ELLIPSOID_FACTOR = GeomAccDefaults.ELLIPSOID_FACTOR

        self.positions_CC = np.sort(np.array(GeomAccDefaults.marker_positions_CC))
        self.positions_AP_LR = np.array(GeomAccDefaults.markerPositions_AP_LR,dtype=float)

        self.degLimit = GeomAccDefaults.LIMITFITDEGREES
        self.transLimit = GeomAccDefaults.LIMITFITTRANS

        if self.phantomType == 'MRSim':
            self.NUMBEROFINSTANCESPERSERIES = GeomAccDefaults.NUMBEROFINSTANCESPERSERIES
            self.PHANTOM_OFFSET = GeomAccDefaults.PHANTOM_OFFSET
            
        self.expectedMarkerPositions = self._expected_marker_positions(self.positions_CC, self.positions_AP_LR)
        self.tablePosition = 0.0

        #Results
        self.detectedMarkerPositions  = None
        self.correctedMarkerPositions = None
        self.closestExpectedMarkerIndices = None
        self.differencesCorrectedExpected = None

    def _expected_marker_positions(self, marker_positions_cc, markerPositions_AP_LR):
        """
        Generate the expected marker positions for a phantom scan.

        This method creates a complete list of expected marker positions by 
        combining the marker positions in the anterior-posterior (AP) and 
        left-right (LR) directions with the provided cranio-caudal (CC) positions. 
        If the phantom type is 'MRSim', an additional offset is applied to the 
        calculated positions.

        Args:
            marker_positions_cc (list or array-like): A list of cranio-caudal (CC) 
                positions where markers are scanned.
            markerPositions_AP_LR (list or array-like): A list of marker positions 
                in the anterior-posterior (AP) and left-right (LR) directions.

        Returns:
            numpy.ndarray: A 2D array containing the expected marker positions 
            for all CC positions, with the optional phantom offset applied if 
            the phantom type is 'MRSim'.
        """
        expected_marker_positions= np.vstack([self._marker_positions_at_cc_pos(cc_pos,markerPositions_AP_LR) for cc_pos in marker_positions_cc])
        if self.phantomType == 'MRSim':
            expected_marker_positions += self.PHANTOM_OFFSET
            # offset is currently 0, but the setup correction finds quite large offsets
            # for the MRSim data (especially in AP), so we might need to add this offset
        return expected_marker_positions

    def _marker_positions_at_cc_pos(self, cc_pos, markerPositions_AP_LR):
        """
        Computes the 3D marker positions by appending a constant CC (cranio-caudal) 
        position to the given 2D marker positions in the AP (anterior-posterior) 
        and LR (left-right) directions.

        Args:
            cc_pos (float): The constant cranio-caudal position to be appended.
            markerPositions_AP_LR (numpy.ndarray): A 2D array of shape (N, 2) 
                containing the marker positions in the AP and LR directions, 
                where N is the number of markers.

        Returns:
            numpy.ndarray: A 2D array of shape (N, 3) containing the 3D marker 
            positions with the CC position appended to each marker.
        """
        return np.hstack((markerPositions_AP_LR,
                          np.ones((len(markerPositions_AP_LR), 1)) * cc_pos))
    
    def loadSeriesAndDetectMarkers(self, data):
        """
        Load DICOM series and detect markers based on the phantom type.
        This method processes DICOM series data to detect marker positions. It handles
        two types of phantoms: 'MRL' and 'MRSim'. The processing logic differs based on
        the phantom type.
        Args:
            data: An object providing access to DICOM series data. It must have a 
                  `getAllSeries` method that returns a list of DICOM series.
        Behavior:
            - For 'MRL' phantom type:
                - Loads a single series (typically with 400 slices).
                - Reads header data and pixel data from the series.
                - Detects marker positions in the image data.
            - For 'MRSim' phantom type:
                - Filters and selects relevant series (typically 7 series with 25 slices each).
                - Sorts the selected series.
                - Iterates through each series to:
                    - Select instances and validate the number of instances.
                    - Extract table position from the series description.
                    - Read pixel data and detect marker positions.
                    - Filters detected marker positions based on the table position.
                - Combines all detected marker positions across series.
        Attributes Modified:
            - `self.detectedMarkerPositions`: Stores the detected marker positions
              as a NumPy array.
        Warnings:
            - Logs warnings if no DICOM instances are found in a series.
            - Logs warnings if the number of instances in a series does not match
              the expected number (`self.NUMBEROFINSTANCESPERSERIES`).
        Notes:
            - The method assumes that the `phantomType` attribute is set to either
              'MRL' or 'MRSim'.
            - The method relies on several helper methods such as `readHeaderData`,
              `_readPixelData`, `detectMarkerPositions`, `selectRelevantSeriesOnly`,
              `_sortSelectedSeriesList`, `selectInstances`, and `_getTablePosFromSeriesDescr`.
        """
        print("-GeomAcc - loadSeriesAndDetectMarkers -- ")
        dcmSeries = data.getAllSeries()
        if self.phantomType == 'MRL':
            self.readHeaderData(dcmSeries[0][0])
            imageData, pixelCoordinates = self._readPixelData(dcmSeries[0])
            self.detectedMarkerPositions = self.detectMarkerPositions(imageData=imageData, pixelCoordinates=pixelCoordinates)
        elif self.phantomType == 'MRSim':            
            selectedSeries = self.selectRelevantSeriesOnly(dcmSeries)
            self.readHeaderData(selectedSeries[0][0])
            selectedSeries = self._sortSelectedSeriesList(selectedSeries)
            allDetectedMarkerPositions = np.empty((0,3))
            for series in selectedSeries:
                print("-GeomAcc - loadSeriesAndDetectMarkers -- Loading series with SeriesNumber {0}.".format(series[0].SeriesNumber))
                
                selectedInstances = self.selectInstances(series)
                nInstances = len(selectedInstances)
                if nInstances == 0:
                    print("-GeomAcc - loadSeriesAndDetectMarkers -- WARNING: No DICOM instances in series, skipping this series!")
                elif not nInstances == self.NUMBEROFINSTANCESPERSERIES:
                    print("-GeomAcc - loadSeriesAndDetectMarkers -- WARNING: Missing a number of instances --> Expected: {0}; Found: {1}!".format(
                            str(self.NUMBEROFINSTANCESPERSERIES), str(nInstances)))
                else:
                    cc_pos = self._getTablePosFromSeriesDescr(series[0].SeriesDescription) # get the table position from the first instance
                    print("-GeomAcc - loadSeriesAndDetectMarkers -- Processing position: {0}".format(str(cc_pos)))
                    imageData, pixelCoordinates = self._readPixelData(series)
                    detectedMarkerPositions = self.detectMarkerPositions(imageData=imageData, pixelCoordinates=pixelCoordinates)
                    detectedMarkerPositions = detectedMarkerPositions[abs(detectedMarkerPositions[:,self.CC] - cc_pos) < self.LIMIT_CC_SEPARATION_FROM_CC_POSITION]
                    allDetectedMarkerPositions = np.vstack((allDetectedMarkerPositions,detectedMarkerPositions))
                    
            self.detectedMarkerPositions = allDetectedMarkerPositions
    
    def _readPixelData(self, dcmSeries):
        """
        Reads pixel data and associated metadata from a series of DICOM instances.

        This method processes a series of DICOM instances to extract pixel data, 
        image positions, and pixel spacing. It sorts the image data and calculates 
        pixel coordinates based on the extracted metadata.

        Args:
            dcmSeries (list): A list of DICOM instances, where each instance contains 
                              pixel data and associated metadata such as 
                              ImagePositionPatient and PixelSpacing.

        Returns:
            tuple: A tuple containing:
                - imageData (numpy.ndarray): A 3D array of pixel data from the DICOM series.
                - pixelCoordinates (numpy.ndarray): A 3D array of calculated pixel coordinates 
                                                    based on image positions and pixel spacing.
        """
        print("--GeomAcc - _readPixelData -- ")
        imageData = []
        image_positions = []
        for dcmInstance in dcmSeries:
            imageData.append(dcmInstance.pixel_array.astype('float32'))
            image_positions.append(dcmInstance.ImagePositionPatient)
            pixelSpacing = dcmInstance.PixelSpacing
        imageData = np.array(imageData)
        image_positions = np.array(image_positions, dtype=np.float32)
        imageData, image_positions = self._sortImageData(imageData, image_positions)
        pixelCoordinates = self._calculatePixelCoordinates(image_positions, imageData.shape, pixelSpacing)
        return imageData, pixelCoordinates
    
    def _sortImageData(self, imageData, imagePositions):
        """
        Sorts image data and corresponding image positions based on the z-coordinates of the image positions.

        Args:
            imageData (numpy.ndarray): A 3D array representing the image data.
            imagePositions (list of tuples): A list of tuples where each tuple contains the (x, y, z) 
                coordinates of the image positions.

        Returns:
            tuple: A tuple containing:
                - numpy.ndarray: The sorted image data array.
                - list of tuples: The sorted list of image positions.
        """
        correctOrder = np.argsort([z for x, y, z in imagePositions])
        return imageData[correctOrder], imagePositions[correctOrder]

    def _calculatePixelCoordinates(self, image_positions, pixelDataShape, pixelSpacing):
        """
        Calculate the pixel coordinates in a 3D space based on image positions, pixel data shape, 
        and pixel spacing.

        Args:
            image_positions (numpy.ndarray): A 2D array where each row represents the (x, y, z) 
                coordinates of a pixel in the image.
            pixelDataShape (tuple): The shape of the pixel data array, typically in the form 
                (number_of_slices, height, width).
            pixelSpacing (tuple): The spacing between pixels in the y and x dimensions, 
                typically in the form (spacing_y, spacing_x).

        Returns:
            list: A list containing three elements:
                - pixelCoordinates_X (numpy.ndarray): The x-coordinates of the pixels.
                - pixelCoordinates_Y (numpy.ndarray): The y-coordinates of the pixels.
                - image_positions[:, 2] (numpy.ndarray): The z-coordinates of the pixels.
        """
        pixelCoordinates_X = image_positions[0][1] + np.arange(pixelDataShape[1]) * pixelSpacing[1]
        pixelCoordinates_Y = image_positions[0][0] + np.arange(pixelDataShape[2]) * pixelSpacing[0]
        pixelCoordinates = [pixelCoordinates_X, pixelCoordinates_Y, image_positions[:, 2]]
        return pixelCoordinates
    
    def readHeaderData(self,dcmfile):
        """
        Extracts and processes header data from a DICOM file.

        Args:
            dcmfile: A DICOM file object containing metadata attributes such as 
                     AcquisitionDate, AcquisitionTime, StationName, and SeriesDescription.

        Attributes Set:
            studyDate (datetime.date): The acquisition date of the study, parsed from the DICOM file.
            studyTime (datetime.time): The acquisition time of the study, parsed from the DICOM file.
            studyScanner (str): The name of the scanner/station used for the study.
            seriesDescription (str): A description of the series from the DICOM file.
        """
        self.studyDate=datetime.strptime(dcmfile.AcquisitionDate,"%Y%m%d").date()
        self.studyTime=datetime.strptime(dcmfile.AcquisitionTime,"%H%M%S.%f").time()
        self.studyScanner=dcmfile.StationName
        self.seriesDescription = dcmfile.SeriesDescription

    def detectMarkerPositions(self, imageData, pixelCoordinates):
        """
        Detects marker positions in the given image data.

        This method identifies high-contrast voxels in the provided image data,
        converts their indices to coordinates, and clusters them to determine
        marker positions.

        Args:
            imageData (numpy.ndarray): The image data to process for marker detection.
            pixelCoordinates (numpy.ndarray): The pixel coordinates corresponding to the image data.

        Returns:
            numpy.ndarray: An array of detected marker positions, where each position
            is represented as a coordinate in the image space.

        Prints:
            Logs the number of high-contrast voxels detected and the number of markers found.
        """
        print("--GeomAcc - detectMarkerPositions -- ")
        highContrastVoxels = self._getHighContrastVoxelsFromImageData(imageData)
        highContrastPoints = self._convertIndicesToCoords(highContrastVoxels, pixelCoordinates)
        print("--GeomAcc - detectMarkerPositions -- Detected {0} high contrast voxels".format(str(highContrastPoints.shape[0])))
        detectedMarkerPositions = self._createClusters(highContrastPoints)
        print("--GeomAcc - detectMarkerPositions -- Found {0} markers".format(str(detectedMarkerPositions.shape[0])))
        return detectedMarkerPositions
    
    def _getHighContrastVoxelsFromImageData(self, imageData):
        """
        Identifies high-contrast voxels from the given image data using a Gaussian filter.

        This method applies a Gaussian filter to the input image data and identifies
        voxels that exceed a specified intensity percentile. The resulting indices
        of these high-contrast voxels are returned.

        Args:
            imageData (numpy.ndarray): The input 3D image data array.

        Returns:
            numpy.ndarray: A 2D array of indices (shape: 3 x N) where N is the number
                           of high-contrast voxels. Each column represents the
                           coordinates of a voxel in the format [z, y, x].

        Notes:
            - The Gaussian filter is applied with a standard deviation defined by
              `self.GAUSSIAN_LAPLACE_SIGMA`.
            - The cutoff intensity is determined by the percentile specified in
              `self.FILTER_CUTOFF_PERCENTILE`.
            - The filtered data is normalized by dividing by the square root of
              2π times the square of the standard deviation.
        """
        print("---GeomAcc - _getHighContrastVoxelsFromImageData -- ")
        sigma = self.GAUSSIAN_LAPLACE_SIGMA
        cutoff = self.FILTER_CUTOFF_PERCENTILE
        dataCube = gaussian_filter(imageData, sigma, truncate=5)/ np.sqrt(2*np.pi * sigma**2)
        idx = np.argwhere(dataCube > np.percentile(dataCube,cutoff)).T

        # difference between gaussian_filter and gaussian_laplace?
        #dataCube = ndimage.gaussian_laplace(imageData, self.GAUSSIAN_LAPLACE_SIGMA)
        #idx = np.argwhere(dataCube < self.MARKER_THRESHOLD_AFTER_FILTERING).T
        return idx
    
    def _convertIndicesToCoords(self, indexList, pixelCoordinates):
        """
        Converts a list of indices to corresponding coordinates based on the provided pixel coordinates.

        Args:
            indexList (list): A list of indices representing positions in the 3D space.
                              The indices are expected in the order [z, x, y].
            pixelCoordinates (list of arrays): A list containing three arrays representing
                                                the pixel coordinates along each axis (x, y, z).

        Returns:
            numpy.ndarray: A 2D array where each row corresponds to the (x, y, z) coordinates
                           derived from the input indices.

        Notes:
            - The order of indices in `indexList` is assumed to be [z, x, y].
            - Ensure that the dimensions of `pixelCoordinates` align with the indices in `indexList`.
        """
        print("---GeomAcc - _convertIndicesToCoords -- ")
        xCoord = pixelCoordinates[0][indexList[1]] #should these be switched??
        yCoord = pixelCoordinates[1][indexList[2]]
        zCoord = pixelCoordinates[2][indexList[0]]
        return np.array(list(zip(xCoord, yCoord, zCoord)))
    
    def _createClusters(self, points):
        """
        Creates clusters of points based on the phantom type.
        For the 'MRL' phantom type, the method uses k-means clustering to separate 
        distinct marker planes based on the expected CC positions of phantom slabs. 
        It then processes the points in parallel to create clusters.
        For the 'MRSim' phantom type, the method assumes the data is already from 
        a single marker plane and directly processes the points to create clusters.
        Args:
            points (numpy.ndarray): A 2D array of points to be clustered.
        Returns:
            numpy.ndarray: An array of clustered points.
        """
        
        print("---GeomAcc - _createClusters -- ")
        if self.phantomType == 'MRL':
            centroids,cluster_id = kmeans2(points.T[self.CC], self.positions_CC)
            points_per_cc = [points[cluster_id == n_cluster] for n_cluster in np.arange(len(self.positions_CC))]
            workpool = Pool(6)
            clusters = np.concatenate(workpool.map(self._parallel_cluster,points_per_cc))
        elif self.phantomType == 'MRSim':
            # MRSim data arriving here is already from a single marker plane
            points_per_cc = points
            clusters = self._parallel_cluster(points_per_cc)

        return clusters
    
    def _parallel_cluster(self,points):
        """
        Groups points into clusters based on their proximity using an ellipsoid distance metric.

        This method iteratively identifies clusters of points that are close to each other 
        based on a predefined distance threshold (`LIMIT_squaredDistanceToCurrentPoint`). 
        For each cluster, the mean position of the points in the cluster is calculated and 
        added to the result.

        Args:
            points (numpy.ndarray): A 2D array of shape (n, m) where `n` is the number of points 
                                    and `m` is the dimensionality of each point.

        Returns:
            numpy.ndarray: A 2D array of shape (k, m) where `k` is the number of clusters and 
                            `m` is the dimensionality of each cluster's mean position.
        """
        result = []
        cur_points=np.copy(points)
        while len(cur_points) > 0:
            curPoint = cur_points[0]
            #squaredDistanceToCurrentPoint = cdist([curPoint], cur_points, 'sqeuclidean')[0]
            # ellipsoid distance? difference between x/y and z voxel size is quite large for MRSIM data...
            # check iff ellipsoid still works with MRL data
            squaredDistanceToCurrentPoint = self._ellipsoidDistance(cur_points, curPoint)
            pointsCloseToCurrent = squaredDistanceToCurrentPoint < self.LIMIT_squaredDistanceToCurrentPoint
            result.append(np.mean(cur_points[pointsCloseToCurrent], axis=0))
            cur_points = cur_points[np.logical_not(pointsCloseToCurrent)]
        return np.array(result)
    
    def _ellipsoidDistance(self, points, centerOfCylinder):
        """
        Calculate the squared distances of points from the center of a cylinder, 
        scaled by an ellipsoid factor.

        This method computes the squared distances of a set of points from the 
        center of a cylinder, applying a scaling factor to account for ellipsoidal 
        distortion.

        Args:
            points (numpy.ndarray): A 2D array of shape (n, 3) representing the 
                coordinates of the points in 3D space.
            centerOfCylinder (numpy.ndarray): A 1D array of shape (3,) representing 
                the coordinates of the center of the cylinder in 3D space.

        Returns:
            numpy.ndarray: A 1D array of shape (n,) containing the squared distances 
            of each point from the center of the cylinder, scaled by the ellipsoid 
            factor.
        """
        distances = (points - centerOfCylinder) * self.ELLIPSOID_FACTOR
        return np.sum(np.power(np.array(distances), 2), axis=1)
    
    def selectRelevantSeriesOnly(self, seriesList):
        """
        Filters the input list of series to include only those whose series number ends with '1'.
        Args:
            seriesList (list): A list of series, where each series is a list of instances. 
                               Each instance is expected to have a `SeriesNumber` attribute.
        Returns:
            list: A list of series that have a `SeriesNumber` ending with '1'.
        """
        
        print("--GeomAcc - selectRelevantSeriesOnly -- ")
        selectedSeries = []
        for series in seriesList:
            instance = series[0]
            seriesNumber = str(instance.SeriesNumber)
            if seriesNumber.endswith('1'):
                selectedSeries.append(series)
        return selectedSeries
    
    def selectInstances(self, dcmSeries):
        """
        Filters and selects instances from a given DICOM series.

        This method iterates through the provided DICOM series and attempts to access
        the `AcquisitionTime` attribute of each instance. Instances with a valid
        `AcquisitionTime` are added to the selection. Instances that do not have this
        attribute or raise an exception are ignored, and a warning message is printed.

        Args:
            dcmSeries (list): A list of DICOM instances to be filtered.

        Returns:
            list: A list of selected DICOM instances that have a valid `AcquisitionTime`.
        """
        selectedInstances = []
        for instance in dcmSeries:
            try:
                acqTime = instance.AcquisitionTime
                selectedInstances.append(instance)
            except:
                print("--GeomAcc - selectInstances -- WARNING: Ignore this instance.")
        return selectedInstances
    
    def _getTablePosFromSeriesDescr(self, seriesDescription):
        """
        Extracts the table position (z-position) from the given series description.

        The series description is expected to be a string containing an underscore-separated
        format, where the third segment represents the z-position. The z-position is assumed
        to be a numeric value, with 'm' indicating a negative sign.

        Args:
            seriesDescription (str): The series description string containing the z-position.

        Returns:
            float: The extracted z-position as a floating-point number.

        Raises:
            IndexError: If the series description does not have at least three segments.
            ValueError: If the z-position segment cannot be converted to a float.
        """
        zpos = seriesDescription.split('_')[2]
        zpos = float(zpos.replace('m', '-'))
        return zpos
    
    def _sortSelectedSeriesList(self,selectedSeries):
        """
        Sorts a list of series based on their table positions extracted from the series descriptions.

        Args:
            selectedSeries (list): A list of series, where each series is expected to have a 
                `SeriesDescription` attribute accessible via `series[0].SeriesDescription`.

        Returns:
            list: A sorted list of series, ordered by their table positions in ascending order.
        """
        print("--GeomAcc - _sortSelectedSeriesList -- ")
        z_positions = []
        for series in selectedSeries:
            z_positions.append(self._getTablePosFromSeriesDescr(series[0].SeriesDescription))

        correctOrder = np.argsort(z_positions) 
        sortedSeriesList = [selectedSeries[i] for i in correctOrder] 
        return sortedSeriesList
        
    def correctDetectedMarkerPositionsForSetup(self):
        """
        Corrects the detected marker positions for setup deviations.

        This function calculates the corrected cluster positions for the detected markers
        based on the setup rotation and translation determined at the table position 0. 
        It identifies markers within a 100 mm radius of the isocenter plane, computes the 
        rigid transformation between the detected and expected marker positions, and applies 
        this transformation to update the corrected marker positions.

        Steps:
        1. Filters detected markers at the isocenter plane (cc_pos = 0.0) within 100 mm of the isocenter.
        2. Retrieves the expected marker positions at the isocenter plane.
        3. Computes the rigid transformation (rotation and translation) between the detected 
           and expected marker positions.
        4. Updates the corrected marker positions using the computed rigid transformation.

        Attributes:
            detectedMarkerPositions (numpy.ndarray): Array of detected marker positions.
            expectedMarkerPositions (numpy.ndarray): Array of expected marker positions.
            rigid_transformation_setup (dict): Dictionary containing the computed rigid 
                                               transformation parameters.

        Dependencies:
            - `indices_cc_pos`: Helper function to filter markers based on the cc position.
            - `cdist`: Function from `scipy.spatial.distance` to compute pairwise distances.
            - `_findRigidTransformation`: Internal method to compute the rigid transformation.
            - `setCorrectedMarkerPositions`: Method to update the corrected marker positions.
        
        This function adds corrected clusterpositions to the measurements based on the
        calculated setup rotation and translation in the measurement at tableposition 0
        """
        print("-GeomAcc - correctDetectedClusterPositionsForSetup -- ")
        detected_markers_at_isoc_plane = self.detectedMarkerPositions[self.indices_cc_pos(self.detectedMarkerPositions,cc_pos=0.0)]
        dist_to_isoc_2d = cdist([[0.0, 0.0]], detected_markers_at_isoc_plane[:,:-1], metric='euclidean')[0]
        ix = dist_to_isoc_2d < 100 #Select only markers within 100 mm of isoc
        detected_markers_at_isoc_plane = detected_markers_at_isoc_plane[ix]

        #markers_at_isoc_plane = self._marker_positions_at_cc_pos(0.0, self.positions_AP_LR)
        markers_at_isoc_plane = self.expectedMarkerPositions[self.indices_cc_pos(self.expectedMarkerPositions,cc_pos=0.0)]
        self.rigid_transformation_setup = self._findRigidTransformation(detected_markers_at_isoc_plane,markers_at_isoc_plane)
        self.setCorrectedMarkerPositions(self.rigid_transformation_setup)

    
    def indices_cc_pos(self, positions, cc_pos):
        """
        Determines the indices of positions that are within a specified limit 
        from a given cranio-caudal (CC) position.

        Args:
            positions (numpy.ndarray): A 2D array where each row represents a 
                position in a multi-dimensional space. The CC dimension is 
                accessed using the `self.CC` index.
            cc_pos (float): The cranio-caudal position to compare against.

        Returns:
            numpy.ndarray: A boolean array where each element indicates whether 
            the corresponding position in `positions` is within the specified 
            limit (`self.LIMIT_CC_SEPARATION_FROM_CC_POSITION`) from `cc_pos`.
        """
        return np.abs(positions.T[self.CC] - cc_pos) < self.LIMIT_CC_SEPARATION_FROM_CC_POSITION
    
    def _findRigidTransformation(self, detectedMarkerPositions, expectedMarkerPositions):
        """
        Finds the rigid transformation (translation and rotation) that best aligns the detected marker positions
        with the expected marker positions using optimization.
        Args:
            detectedMarkerPositions (numpy.ndarray): Array of detected marker positions in 3D space.
            expectedMarkerPositions (numpy.ndarray): Array of expected marker positions in 3D space.
        Returns:
            numpy.ndarray: Optimized transformation parameters as a 1D array of size 6.
                           The first three elements represent translations [AP, LR, CC],
                           and the last three elements represent rotations [AP, LR, CC] in degrees.
        Notes:
            - The optimization minimizes the sum of squared differences between the transformed detected
              marker positions and the expected marker positions.
            - The optimization is bounded by `self.transLimit` for translations and `self.degLimit` for rotations.
            - The initial guess for the optimization includes the average detected CC position.
        """
        print("--GeomAcc - _findRigidTransformation -- ")
        # average detected cc position
        init_CC = np.mean(detectedMarkerPositions, axis=0)[self.CC]

        # optimization init
        optimization_initial_guess = np.zeros(6)
        optimization_initial_guess[self.CC] = init_CC

        # optimization bounds
        optimization_bounds = [(-self.transLimit, self.transLimit),
                               (-self.transLimit, self.transLimit),
                               (-self.transLimit, self.transLimit),
                               (-self.degLimit, self.degLimit),
                               (-self.degLimit, self.degLimit),
                               (-self.degLimit, self.degLimit)]

        self.tablePosition = 0.0
        
        def penaltyFunction(transRot):
            """
            Computes the penalty value based on the differences between detected marker positions
            and expected marker positions after applying a rigid transformation.

            Args:
                transRot (array-like): A 1D array or list containing six elements:
                    - The first three elements represent the translation vector (x, y, z).
                    - The last three elements represent the Euler angles (rotation) in radians.

            Returns:
                float: The penalty value, calculated as the sum of squared differences between
                the transformed detected marker positions and the expected marker positions.
            """
            opt_pos = self.rigidTransform(detectedMarkerPositions, translation=transRot[0:3], eulerAngles=transRot[3:6])
            differences = self.getdifferences(opt_pos, expectedMarkerPositions)
            penalty = np.sum(np.sum(np.power(differences, 2)))
            return penalty

        opt_result = optimize.minimize(fun=penaltyFunction,
                                       x0=optimization_initial_guess,
                                       bounds=optimization_bounds,
                                       tol=.0001)

        print("-GeomAcc - _findRigidTransformation -- Translations [AP LR CC]: [{0} {1} {2}]".format(str(opt_result.x[0]),str(opt_result.x[1]),str(opt_result.x[2])))
        print("-GeomAcc - _findRigidTransformation -- Rotations [AP LR CC]: [{0} {1} {2}]".format(str(opt_result.x[3]),str(opt_result.x[4]),str(opt_result.x[5])))
        return opt_result.x

    def rigidRotation(self, markerPositions, eulerAngles):
        """
        Applies a rigid rotation to a set of 3D marker positions based on given Euler angles.

        This function computes a 3D rotation matrix from the provided Euler angles and applies
        it to the input marker positions. The rotation follows the intrinsic Tait-Bryan angles
        (yaw, pitch, roll) convention.

        Parameters:
            markerPositions (numpy.ndarray): A 2D array of shape (N, 3) representing the coordinates
                of N markers in 3D space.
            eulerAngles (list or numpy.ndarray): A list or array of three Euler angles [alpha, beta, gamma]
                in radians, representing rotations around the z, y, and x axes, respectively.

        Returns:
            numpy.ndarray: A 2D array of shape (N, 3) containing the rotated coordinates of the markers.
        """
        s0 = np.sin(eulerAngles[0])
        c0 = np.cos(eulerAngles[0])
        s1 = np.sin(eulerAngles[1])
        c1 = np.cos(eulerAngles[1])
        s2 = np.sin(eulerAngles[2])
        c2 = np.cos(eulerAngles[2])

        m00 = c1 * c2
        m01 = c0 * s2 + s0 * s1 * c2
        m02 = s0 * s2 - c0 * s1 * c2
        m10 = -c1 * s2
        m11 = c0 * c2 - s0 * s1 * s2
        m12 = s0 * c2 + c0 * s1 * s2
        m20 = s1
        m21 = -s0 * c1
        m22 = c0 * c1

        rotationMatrix = np.array([
            [m00, m01, m02],
            [m10, m11, m12],
            [m20, m21, m22]])

        return np.dot(markerPositions, rotationMatrix)

    def rigidTranslation(self, markerPositions, Translation):
        """
        Applies a rigid translation to a set of marker positions.

        Parameters:
            markerPositions (numpy.ndarray): An array of marker positions, where each position is represented as a coordinate (e.g., [x, y, z]).
            Translation (numpy.ndarray): A translation vector to be applied to the marker positions.

        Returns:
            numpy.ndarray: The translated marker positions, obtained by adding the translation vector to the original positions.
        """
        return markerPositions + Translation

    def rigidTransform(self, positions, translation, eulerAngles):
        """
        Applies a rigid transformation to a set of positions based on the specified translation
        and Euler angles. The transformation process differs depending on the phantom type.

        Parameters:
            positions (list or numpy.ndarray): The initial positions to be transformed.
            translation (list or numpy.ndarray): The translation vector [x, y, z].
            eulerAngles (list or numpy.ndarray): The Euler angles [alpha, beta, gamma] for rotation.

        Returns:
            list or numpy.ndarray: The transformed positions after applying the rigid transformation.

        Notes:
            - For 'MRSim' phantom type, the method applies a translation to adjust for the table position,
              performs a rotation, and then re-applies the table position translation before applying
              the final translation.
            - For 'MRL' phantom type, the method directly applies the rotation followed by the translation.
        """
        if self.phantomType == 'MRSim':
            local_positions = self.rigidTranslation(positions, [0.0, 0.0, -self.tablePosition])
            rotated_local_positions = self.rigidRotation(local_positions, eulerAngles)
            rotated_positions = self.rigidTranslation(rotated_local_positions, [0.0, 0.0, + self.tablePosition])
            transformed_positions = self.rigidTranslation(rotated_positions, translation)
        elif self.phantomType == 'MRL':
            rotated_positions = self.rigidRotation(positions, eulerAngles)
            transformed_positions = self.rigidTranslation(rotated_positions, translation)
        return transformed_positions

    def setCorrectedMarkerPositions(self,transformation):
        """
        Apply a transformation to detected marker positions to compute corrected marker positions.
        This method adjusts the detected marker positions based on the provided transformation
        and updates the corrected marker positions. The behavior of the transformation depends
        on the phantom type (`MRL` or `MRSim`).
        Parameters:
            transformation (list or array): A 6-element list or array representing the transformation.
                The first three elements correspond to rotation parameters, and the last three elements
                correspond to translation parameters.
        Behavior:
            - For phantom type 'MRL':
                Applies a rigid transformation to all detected marker positions at once.
            - For phantom type 'MRSim':
                Applies the transformation separately for each table position (`positions_CC`),
                and combines the results.
        Updates:
            - self.correctedMarkerPositions: The transformed marker positions.
            - self.closestExpectedMarkerIndices: Indices of the closest expected markers to the corrected markers.
            - self.differencesCorrectedExpected: Differences between corrected marker positions and expected marker positions.
        Dependencies:
            - self.rigidTransform: Method to apply the rigid transformation.
            - self.indices_cc_pos: Method to filter detected markers based on table position.
            - self.closestLocations: Method to find the closest expected marker indices.
            - self.getdifferences: Method to compute differences between corrected and expected marker positions.
        """
        if self.phantomType == 'MRL':
            self.correctedMarkerPositions = self.rigidTransform(self.detectedMarkerPositions,transformation[0:3],[transformation[3], transformation[4], transformation[5]])
        elif self.phantomType == 'MRSim':
            # apply the correction on each position separately
            corrected_markers = np.empty((0,3))
            for cc_pos in self.positions_CC:
                self.tablePosition = cc_pos
                detected_markers_at_cc_pos = self.detectedMarkerPositions[self.indices_cc_pos(self.detectedMarkerPositions,cc_pos=cc_pos)]
                corrected_markers_at_cc_pos = self.rigidTransform(detected_markers_at_cc_pos,transformation[0:3],[transformation[3], transformation[4], transformation[5]])
                corrected_markers = np.vstack((corrected_markers,corrected_markers_at_cc_pos))
            self.correctedMarkerPositions = corrected_markers
            
        self.closestExpectedMarkerIndices = self.closestLocations(self.correctedMarkerPositions,self.expectedMarkerPositions)
        self.differencesCorrectedExpected = self.getdifferences(self.correctedMarkerPositions,self.expectedMarkerPositions)

    def closestLocations(self, detectedMarkerPositions, expectedMarkerPositions):
        """
        Finds the indices of the closest detected marker positions to the expected marker positions.

        This method calculates the squared Euclidean distances between each detected marker position
        and each expected marker position, and returns the indices of the detected marker positions
        that are closest to each expected marker position.

        Args:
            detectedMarkerPositions (numpy.ndarray): A 2D array of shape (n, d) representing the positions
                of detected markers, where `n` is the number of detected markers and `d` is the dimensionality
                of the positions (e.g., 2 for 2D or 3 for 3D).
            expectedMarkerPositions (numpy.ndarray): A 2D array of shape (m, d) representing the positions
                of expected markers, where `m` is the number of expected markers and `d` is the dimensionality
                of the positions.

        Returns:
            numpy.ndarray: A 1D array of shape (m,) containing the indices of the detected marker positions
            that are closest to each expected marker position.
        """
        distances = np.sum(np.power((detectedMarkerPositions - expectedMarkerPositions[:, np.newaxis]), 2), axis=2)
        return np.argmin(distances, axis=0)

    def getdifferences(self, markerPositions,expectedMarkerPositions):
        """
        Calculate the differences between the actual marker positions and the expected marker positions.

        Args:
            markerPositions (numpy.ndarray): An array of actual marker positions.
            expectedMarkerPositions (numpy.ndarray): An array of expected marker positions.

        Returns:
            numpy.ndarray: An array of differences between the actual and expected marker positions,
                           where the closest expected positions are matched to the actual positions.
        """
        return markerPositions - expectedMarkerPositions[self.closestLocations(markerPositions,expectedMarkerPositions)]

    def calculateDistanceToDifference3D(self, limit_in_mm):
        """
        Calculates the distance from the origin (0,0,0) to the closest marker 
        location that exceeds a specified deviation limit, as well as the 
        distance corresponding to the 98th percentile of markers within the limit.
        This function operates in 3D space and considers the entire volume.
        Parameters:
        ----------
        limit_in_mm : float
            The defined deviation limit in millimeters. The function identifies 
            the first marker whose detected position exceeds this deviation.
        Returns:
        -------
        marker_dist_to_origin : float
            The distance from the origin to the first marker that exceeds the 
            specified deviation limit.
        marker98_dist_to_origin : float
            The distance from the origin to the marker corresponding to the 
            98th percentile of markers within the deviation limit, as per 
            longitudinal QA standards.
        
        """
        print("-GeomAcc - calculateDistanceToDifference3D -- ")
        # calculate distance to origin and sort based on this
        detected = self.correctedMarkerPositions
        origin = [[0.0,0.0,0.0]]
        detected_dist_to_origin = cdist(detected,origin).flatten()
        detected_dist_to_origin_sortix = np.argsort(detected_dist_to_origin)
        
        # calculate the difference lengths and find the first difference length 
        # that exceeds the limit (after sorting the list using the previous indices)
        difference = self.differencesCorrectedExpected
        difference_length = np.sqrt( difference[:,0]**2 + difference[:,1]**2 + difference[:,2]**2 )
        limit_ix = np.argmax(difference_length[detected_dist_to_origin_sortix] > limit_in_mm)
        
        # take the index for the first marker to exceed limit
        # and calculate its distance from the origin
        markerposition = detected[detected_dist_to_origin_sortix][limit_ix]
        marker_dist_to_origin = np.sqrt( markerposition[0]**2 + markerposition[1]**2 + markerposition[2]**2 )

        # Also get the marker that gives the radius for which 98% of the markers are within the limit
        # (according to longitudinal QA article)
        # For this, consider the max we just found to be 98% instead of the max (100%)
        ix_98 = np.int0(np.floor(100*limit_ix/98))
        markerposition_98 = detected[detected_dist_to_origin_sortix][ix_98]
        marker98_dist_to_origin = np.sqrt( markerposition_98[0]**2 + markerposition_98[1]**2 + markerposition_98[2]**2 )
                
        return marker_dist_to_origin, marker98_dist_to_origin

    def calculateStatisticsForDSV3D(self, diameter):
        """
        Calculate statistical metrics for marker deviations within a specified spherical region (DSV).
        This method computes the maximum deviation, 98th percentile, and 95th percentile of 
        3D deviation lengths for markers detected within a spherical region of interest 
        defined by the given diameter.
        Parameters
            Diameter of the spherical region of interest (DSV) in millimeters.
        dsv_max : float
            Maximum 3D deviation length within the specified DSV.
        dsv_p98 : float
            98th percentile of 3D deviation lengths within the specified DSV.
        dsv_p95 : float
            95th percentile of 3D deviation lengths within the specified DSV.
        """
        print("-GeomAcc - calculateStatisticsForDSV3D -- ")
        detected = self.correctedMarkerPositions
        differences = self.differencesCorrectedExpected
        
        detected_dist_to_origin = np.sqrt(np.power(detected[:,0], 2)+np.power(detected[:,1], 2)+np.power(detected[:,2], 2))
        dsv_ix = detected_dist_to_origin < diameter/2
        
        # for every detected marker in the given diameter, take 3D distance length of difference with expected
        distlength = np.sqrt(np.sum(np.power(differences[dsv_ix], 2), axis=1))  
        
        dsv_max = np.max(distlength)
        dsv_p98 = np.percentile(distlength,98)
        dsv_p95 = np.percentile(distlength,95)
        
        return dsv_max, dsv_p98, dsv_p95

    def createAndSaveImages(self,results):
        """
        Generates and saves images related to geometric accuracy analysis, and adds them to the results object.

        Args:
            results (object): An object that stores the results of the analysis. 
                              It must have an `addObject` method to associate generated images with specific keys.

        Functionality:
            - Creates a deviation figure and saves it as "positions.jpg".
            - Adds the deviation figure to the results object with the key "detectedPositions".
            - Creates a histogram figure and saves it as "Histograms.jpg".
            - Adds the histogram figure to the results object with the key "Histograms".
        """
        print("-GeomAcc - createAndSaveImages -- ")
        fileName = "positions.jpg"
        self.createDeviationFigure(fileName)
        results.addObject("detectedPositions",fileName)

        fileName = "Histograms.jpg"
        self.createHistogramsFigure(fileName)
        results.addObject("Histograms", fileName)

    def createDeviationFigure(self,fileName=None):
        """
        Creates and saves a deviation figure with subplots.

        This method generates a figure with 8 subplots arranged in a 2x4 grid. The subplots
        display deviation data for specific positions along the CC axis. The figure includes
        a title based on the study date, time, and scanner information. A legend is also
        included in one of the subplots.

        Args:
            fileName (str, optional): The file path where the figure will be saved. If not 
                                      provided, the figure will not be saved.

        Subplots:
            - Row 0, Columns 0-3: Deviation data for positions_CC[3] to positions_CC[6].
            - Row 1, Column 0: Deviation legend.
            - Row 1, Columns 1-3: Deviation data for positions_CC[2] to positions_CC[0].

        Notes:
            - The method uses `self.studyDate`, `self.studyTime`, and `self.studyScanner` 
              to generate the figure title.
            - The method assumes `self.positions_CC` is a list or array-like object with 
              at least 7 elements.
            - The figure is saved with a resolution of 150 DPI.
        """
        fig, axs = plt.subplots(ncols=4, nrows=2, sharey=True, sharex=True,figsize=(12, 6),constrained_layout=True)
        title = self.studyDate.strftime("%Y-%m-%d ") +self.studyTime.strftime("%H:%M:%S ") +" "+str(self.studyScanner)
        fig.suptitle(title,fontsize=20,x=0.5)

        self._createDeviationSubplot(ax=axs[0, 0], cc_position=self.positions_CC[3])
        self._createDeviationSubplot(ax=axs[0, 1], cc_position=self.positions_CC[4])
        self._createDeviationSubplot(ax=axs[0, 2], cc_position=self.positions_CC[5])
        self._createDeviationSubplot(ax=axs[0, 3], cc_position=self.positions_CC[6])

        self._createDeviationLegend (ax=axs[1, 0])
        self._createDeviationSubplot(ax=axs[1, 1], cc_position=self.positions_CC[2])
        self._createDeviationSubplot(ax=axs[1, 2], cc_position=self.positions_CC[1])
        self._createDeviationSubplot(ax=axs[1, 3], cc_position=self.positions_CC[0])

        fig.tight_layout()
        fig.savefig(fileName,dpi=150)

    def _createDeviationSubplot(self, ax, cc_position):
        """
        Creates a deviation subplot for visualizing marker positions and deviations.
        This method generates a subplot on the provided axis (`ax`) to display the 
        detected marker positions, their deviations from expected positions, and 
        additional visual elements such as circles representing deviation thresholds.
        Args:
            ax (matplotlib.axes.Axes): The axis on which the subplot will be drawn.
            cc_position (float): The current position along the cranio-caudal (CC) axis.
        Attributes:
            self.correctedMarkerPositions (numpy.ndarray): Array of corrected marker positions.
            self.differencesCorrectedExpected (numpy.ndarray): Array of differences between 
                corrected and expected marker positions.
            self.positions_AP_LR (numpy.ndarray): Array of marker positions in the AP-LR plane.
            self.LR (int): Index for the left-right (LR) coordinate in marker positions.
            self.AP (int): Index for the anterior-posterior (AP) coordinate in marker positions.
            self.phantomType (str): Type of phantom being used (e.g., 'MRL').
        Visual Elements:
            - Blue 'x': Expected marker positions.
            - Red 'o': Detected positions with deviations > 2 mm.
            - Yellow 'o': Detected positions with deviations < 2 mm.
            - Green 'o': Detected positions with deviations < 1 mm.
            - White 'x': Origin marker at (0, 0).
            - Yellow-green circle: 1 mm deviation threshold (if applicable).
            - Orange circle: 2 mm deviation threshold (if applicable).
        Notes:
            - The method uses 3D deviation sphere calculations to determine the radius 
              of circles at the current CC position, if the phantom type is 'MRL'.
            - The subplot is configured with a black background, equal aspect ratio, 
              and specific axis limits and ticks.
        """
        ix = self.indices_cc_pos(self.correctedMarkerPositions,cc_position)
        detectedPositions = self.correctedMarkerPositions[ix]
        differences = self.differencesCorrectedExpected[ix]
        distlength = np.sqrt(np.sum(np.power(differences, 2), axis=1))
        markerPositions = self.positions_AP_LR

        ax.set_facecolor('black')
        ax.set_aspect('equal')
        ax.set_title('Position: {0:.1f}'.format(cc_position))
        # ax.text(.02,.9,cc_position, ha='left', va='top', transform=ax.transAxes,color='white')
        #ax.text(.98, .9, '{0:.1f} $\pm$ {1:.1f} mm'.format(np.mean(differences),np.std(differences)), ha='right', va='top', transform=ax.transAxes, color='white')

        ax.set_xlim(-275, 275)
        ax.set_ylim(-150, 275)
        ax.set_xticks([],[])
        ax.set_yticks([],[])

        ax.scatter(markerPositions[:, self.LR], - markerPositions[:,  self.AP], marker='x', c='blue')
        ax.scatter(detectedPositions[distlength > 2, self.LR], - detectedPositions[distlength > 2, self.AP], marker='o', c='red')
        ax.scatter(detectedPositions[distlength < 2, self.LR], - detectedPositions[distlength < 2, self.AP], marker='o', c='yellow')
        ax.scatter(detectedPositions[distlength < 1, self.LR], - detectedPositions[distlength < 1, self.AP], marker='o', c='green')
        
        #add origin
        ax.scatter(0,0,marker='x', c='white')
        
        if self.phantomType == 'MRL':
            # draw circles with radius with distance where closest marker with deviation larger than 1mm (white) & 2mm (orange) 
            #dist_origin, markerpos_ix = self.calculateDistanceToDifference(ix, 1)
            #draw_circle = plt.Circle((0,0),dist_origin,fill=False,color='yellowgreen',lw=3)
            #ax.add_artist(draw_circle)
            
            #dist_origin, markerpos_ix = self.calculateDistanceToDifference(ix, 2)
            #draw_circle = plt.Circle((0,0),dist_origin,fill=False,color='orange',lw=3)
            #ax.add_artist(draw_circle)
            
            # Alternative: draw circles based on the 3D DSV
            # the radius of the spheres at the intersections with the phantom slices are given by:
            # radius(z) = sqrt(R^2 - z^2), with R the radius of the current sphere
            radius_1mm, radius_98p_1mm = self.calculateDistanceToDifference3D(1.0)
            radius_2mm, radius_98p_2mm = self.calculateDistanceToDifference3D(2.0)
            
            # first check if the current cc_pos intersects with sphere
            if radius_1mm**2 - cc_position**2 > 0:
                radius_1mm_this_cc_pos = np.sqrt(radius_1mm**2 - cc_position**2)
                draw_circle = plt.Circle((0,0),radius_1mm_this_cc_pos,fill=False,color='yellowgreen',lw=3)
                ax.add_artist(draw_circle)
                
            if radius_2mm**2 - cc_position**2 > 0:
                radius_2mm_this_cc_pos = np.sqrt(radius_2mm**2 - cc_position**2)
                draw_circle = plt.Circle((0,0),radius_2mm_this_cc_pos,fill=False,color='orange',lw=3)
                ax.add_artist(draw_circle)
        

    def _createDeviationLegend(self,ax):
        """
        Creates and adds a legend to the given matplotlib axis to represent 
        deviations from expected marker positions.

        The legend includes markers and labels for different deviation ranges:
        - Blue cross: Expected marker position.
        - Green circle: Deviation less than 1 mm.
        - Yellow circle: Deviation between 1 mm and 2 mm.
        - Red circle: Deviation greater than 2 mm.

        If the phantom type is 'MRL', additional markers are included:
        - Yellow-green line: Deviation less than 1 mm.
        - Orange line: Deviation less than 2 mm.

        The legend is displayed with a gray background and a title indicating 
        the deviation from the expected position. The axis is turned off 
        after adding the legend.

        Args:
            ax (matplotlib.axes.Axes): The matplotlib axis to which the legend 
                                        will be added.
        """

        blue_cross = mlines.Line2D([], [], color='blue', marker='x',
                                  markersize=15,linestyle='None', label='Expected marker')
        green = mlines.Line2D([], [], color='green', marker='o',
                                   markersize=15,linestyle='None', label=r'$\delta$ < 1 mm')
        yellow = mlines.Line2D([], [], color='yellow',linestyle='None', marker='o',
                                   markersize=15, label=r'1 mm < $\delta$ < 2 mm')
        red = mlines.Line2D([], [], color='red',linestyle='None', marker='o',
                                   markersize=15, label=r'$\delta$ > 2 mm')
        items = [blue_cross,green,yellow,red]
        if self.phantomType == 'MRL':
            yellowgreen_line = mlines.Line2D([], [], color='yellowgreen', marker='_',
                                      markersize=15,linestyle='None', label='Deviation < 1 mm')
            orange_line = mlines.Line2D([], [], color='orange', marker='_',
                                      markersize=15,linestyle='None', label='Deviation < 2 mm')
            items.append(yellowgreen_line)
            items.append(orange_line)
        ax.legend(handles=items,loc=10,title=r'Deviation $\delta$ from expected position',facecolor='gray')
        ax.set_axis_off()

    def createHistogramsFigure(self, fileName):
        """
        Generates and saves a figure containing histograms for specific positions.
        This method creates a figure with 2 rows and 4 columns of subplots, where each subplot
        displays a histogram corresponding to a specific position. The figure is titled with
        the study date, time, and scanner information. The histograms are plotted in a specific
        order based on the `positions_CC` attribute, and a legend is included in one of the subplots.
        Args:
            fileName (str): The file path where the generated figure will be saved.
        Notes:
            - The `positions_CC` attribute is expected to be a list or array-like object containing
              position data for the histograms.
            - The `_createHistogramPlot` method is used to generate individual histogram plots.
            - The `_createHistogramLegend` method is used to create the legend for the histograms.
            - The figure is saved as an image file with a resolution of 150 DPI.
        """
        fig, axs = plt.subplots(ncols=4, nrows=2, sharey=True, sharex=True,figsize=(12, 6),constrained_layout=True)
        title = self.studyDate.strftime("%Y-%m-%d ") + self.studyTime.strftime("%H:%M:%S ") + " " + str(self.studyScanner)
        fig.suptitle(title, fontsize=20)
        
        self._createHistogramPlot(ax=axs[0, 0], cc_position=self.positions_CC[3])
        self._createHistogramPlot(ax=axs[0, 1], cc_position=self.positions_CC[4])
        self._createHistogramPlot(ax=axs[0, 2], cc_position=self.positions_CC[5])
        self._createHistogramPlot(ax=axs[0, 3], cc_position=self.positions_CC[6])

        self._createHistogramLegend(ax=axs[1, 0])
        self._createHistogramPlot(ax=axs[1, 1], cc_position=self.positions_CC[2])
        self._createHistogramPlot(ax=axs[1, 2], cc_position=self.positions_CC[1])
        self._createHistogramPlot(ax=axs[1, 3], cc_position=self.positions_CC[0])

        axs[0,0].tick_params(labelbottom=True)
        fig.tight_layout()
        fig.savefig(fileName,dpi=150)

    def _createHistogramPlot(self, ax, cc_position):
        """
        Creates a histogram plot on the given matplotlib axis for the specified 
        cranio-caudal (CC) position.

        Parameters:
            ax (matplotlib.axes.Axes): The matplotlib axis on which the histogram 
                will be plotted.
            cc_position (str): The cranio-caudal position label for the plot title.

        Description:
            - Computes the differences between corrected marker positions and 
              expected positions for the specified CC position.
            - Calculates the mean (μ) and standard deviation (σ) of the differences 
              for each axis (AP, LR, CC).
            - Plots histograms of the differences for the Anterior-Posterior (AP), 
              Left-Right (LR), and Cranio-Caudal (CC) axes.
            - Displays statistical information (mean and standard deviation) as 
              text on the plot.
            - Configures the plot with appropriate labels, colors, and formatting.

        Notes:
            - The method assumes that `self.indices_cc_pos`, 
              `self.correctedMarkerPositions`, `self.differencesCorrectedExpected`, 
              and axis constants (`self.AP`, `self.LR`, `self.CC`) are defined 
              within the class.
            - The histogram bins are fixed between -3 and 3 with 14 intervals.
        """
        ix = self.indices_cc_pos(self.correctedMarkerPositions, cc_position)
        differences = self.differencesCorrectedExpected[ix]
        bins = np.linspace(start=-3, stop=3, num=14)
        mu_diff = np.mean(differences, axis=0)
        sigma_diff = np.std(differences,axis=0)

        textstr = '(AP,LR,CC)\n'\
                  '$\mu$ = ({0:.1f}, {1:.1f}, {2:.1f}) mm \n' \
                  '$\sigma$ = ({3:.1f}, {4:.1f}, {5:.1f}) mm'.format(mu_diff[self.AP],mu_diff[self.LR],mu_diff[self.CC],
                                                                   sigma_diff[self.AP],sigma_diff[self.LR],sigma_diff[self.CC])
        textstrprops = dict(boxstyle='round', facecolor='white', alpha=0.5)

        ax.set_facecolor('white')
        ax.set_title(cc_position)
        ax.hist((differences[:, self.AP],differences[:, self.LR],differences[:, self.CC]), bins=bins, density=False,
                label=('AP', 'LR', 'CC'), color=['green', 'blue', 'red'],rwidth=0.66)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=textstrprops)
        ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])

    def _createHistogramLegend(self,ax):
        """
        Creates and adds a legend to the given matplotlib axis for a histogram plot.

        The legend includes three colored patches representing different directions:
        - Green: AP (Anterior-Posterior)
        - Blue: LR (Left-Right)
        - Red: CC (Cranio-Caudal)

        The legend title indicates the deviation (δ) from the expected position.

        Parameters:
            ax (matplotlib.axes.Axes): The matplotlib axis to which the legend will be added.
        """
        green = mpatches.Patch(color='green',label='AP')
        blue = mpatches.Patch(color='blue',label='LR')
        red = mpatches.Patch(color='red', label='CC')

        ax.legend(handles=[green, blue, red], loc=10, title=r'Deviation $\delta$ from expected position')
        ax.set_axis_off()

    def collectResults3Dmetrics(self,results):
        """
        Collects and calculates 3D geometric accuracy metrics and adds the results 
        to the provided results object.
        This method computes various 3D metrics related to geometric accuracy, 
        including distances to differences, maximum values, and percentiles within 
        specific DSV (Diameter of Spherical Volume) ranges. The results are then 
        stored in the `results` object for further analysis or reporting.
        Args:
            results (Results): An object to store the calculated metrics. It is 
            expected to have an `addFloat` method for adding float values.
        Metrics Calculated:
            - Total DSV to 1 mm (3D)
            - Total DSV to 2 mm (3D)
            - Max within DSV200 (system)
            - Max within DSV300 (system)
            - Max within DSV400 (system)
            - Max within DSV500 (system)
            - p98 within DSV200 (system)
            - p98 within DSV300 (system)
            - p98 within DSV400 (system)
            - p98 within DSV500 (system)
            - Max within DSV200
            - Max within DSV400
            - Max within DSV500
            - p95 within DSV200
            - p95 within DSV400
            - p95 within DSV500
            - Detected markers (number of detected marker positions)
        """
        print("-GeomAcc - collectResults3Dmetrics -- ")
        
        # Calculate 3D metrics
        distance_to_1mm_3d, distance_to_98p_1mm_3d = self.calculateDistanceToDifference3D(1.0) 
        distance_to_2mm_3d, distance_to_98p_2mm_3d = self.calculateDistanceToDifference3D(2.0) 
        results.addFloat("Total DSV to 1 mm (3D)", round(2*distance_to_1mm_3d,2))
        results.addFloat("Total DSV to 2 mm (3D)", round(2*distance_to_2mm_3d,2))
        
        dsv200_max, dsv200_p98, dsv200_p95 = self.calculateStatisticsForDSV3D(200.0)
        dsv300_max, dsv300_p98, dsv300_p95 = self.calculateStatisticsForDSV3D(300.0)
        dsv400_max, dsv400_p98, dsv400_p95 = self.calculateStatisticsForDSV3D(400.0)
        dsv500_max, dsv500_p98, dsv500_p95 = self.calculateStatisticsForDSV3D(500.0)
        
        results.addFloat("Max within DSV200 (system)", round(dsv200_max,2))
        results.addFloat("Max within DSV300 (system)", round(dsv300_max,2))
        results.addFloat("Max within DSV400 (system)", round(dsv400_max,2))
        results.addFloat("Max within DSV500 (system)", round(dsv500_max,2))
        
        results.addFloat("p98 within DSV200 (system)", round(dsv200_p98,2))
        results.addFloat("p98 within DSV300 (system)", round(dsv300_p98,2))
        results.addFloat("p98 within DSV400 (system)", round(dsv400_p98,2))
        results.addFloat("p98 within DSV500 (system)", round(dsv500_p98,2))
        
        results.addFloat("Max within DSV200", round(dsv200_max,2))
        results.addFloat("Max within DSV400", round(dsv400_max,2))
        results.addFloat("Max within DSV500", round(dsv500_max,2))
        
        results.addFloat("p95 within DSV200", round(dsv200_p95,2))
        results.addFloat("p95 within DSV400", round(dsv400_p95,2))
        results.addFloat("p95 within DSV500", round(dsv500_p95,2))

        results.addFloat("Detected markers", round(self.detectedMarkerPositions.shape[0],0))


