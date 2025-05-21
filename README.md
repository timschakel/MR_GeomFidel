# MR_GeomFidel

Generic geometric fidelity QA analysis for both MRsim and MRlinac.

## Overview

**MR_GeomFidel** is a Python-based toolkit designed for automated geometric fidelity quality assurance (QA) of MR images acquired with geometric fidelity phantoms. It supports both MR simulators (MRsim) and MR-linac systems, enabling consistent and reproducible analysis across different platforms.

The toolkit detects phantom markers in DICOM image series, analyzes geometric distortions, and generates statistics and plots to support QA workflows in radiotherapy imaging.

## Features

- Automatic detection and localization of phantom markers in MR images
- Support for both MRsim and MR-linac phantom types
- Analysis of geometric deviations and reporting of key QA metrics
- Modular design for integration with WADQC and other QA frameworks

## Usage

1. **Prepare your DICOM data**: Organize your MR phantom scans in a directory structure compatible with the toolkit.
2. **Configure analysis**: Edit the configuration files to specify phantom type, expected marker positions, and analysis parameters.
3. **Run the analysis**: Use the provided scripts (e.g., `MR_GeometricFidelity.py`) to process your data and generate QA results.
4. **Review outputs**: Examine the generated statistics, plots, and logs for QA reporting.

## Configs

The **/configs** folder has the relevant config files for MRSim and MRL analyses. It also includes the results meta files for the current systems. 
Configuration of the phantom analysis can be done using GeomAccDefaultsMRL/MRSim.py for the respective phantom types (define expected marker positions etc).

## WAD-QC Import
The .zip file in the **/selectors** folder can be directly imported into WAD-QC. This will import all current selectors along with config and results meta files. 

## Requirements

- Python 3.x
- NumPy, SciPy, Matplotlib, pydicom
- WADQC software (for integration and data handling)

## References

- WADQC Software: [https://bitbucket.org/MedPhysNL/wadqc/wiki/Home](https://bitbucket.org/MedPhysNL/wadqc/wiki/Home)
- UMC Utrecht, Medical Physics

## Author

Tim Schakel, UMC Utrecht, 2025

---

For more details, see the code documentation and example scripts in this repository.