# EEG Classification Using ETC

This repository contains the codebase for the research paper titled "EEG Classification Using Extra Trees Classifier (ETC)." This project provides a framework for replicating the results discussed in the paper. The repository is designed to allow researchers and developers to reproduce the experiments and build upon the work.

The code in this repository extensively uses [ETCPy](https://github.com/pranaysy/ETCPy), a Python library for computing Effort To Compress (ETC) using the NSRPS algorithm. Make sure to install this dependency following the instructions below.

## How to Use This Repository

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/yojit-kumar/eeg-classification-using-etc.git
   cd eeg-classification-using-etc
   ```

2. **Set Up the Environment**  
   This project uses Python, Jupyter Notebooks, and several additional libraries. It is recommended to use `conda` to manage the environment.

   - Install Miniconda/Anaconda if you don’t have it already. You can download it from [here](https://docs.conda.io/en/latest/miniconda.html).

   - Create a new environment and install all required dependencies in one step:
     ```bash
     conda create -n eeg-etc python=3.13.2 matplotlib pandas numpy scipy scikit-learn jupyter jupyterlab joblib cython edfio
     ```

     Activate the environment:
     ```bash
     conda activate eeg-etc
     ```

3. **Install Additional Pip Packages**  
   There are additional Python packages required for this project that are not available via `conda`. Install them using `pip`:
   ```bash
   pip install mne pyedflib
   ```
   ### Explanation of Pip Dependencies:
   - **mne**: For processing and analyzing EEG data.
   - **pyedflib**: For reading and writing EDF (European Data Format) files.

4. **Install ETCPy**  
   The project depends on [ETCPy](https://github.com/pranaysy/ETCPy). Install it from its GitHub repository using the following command:
   ```bash
   pip install git+https://github.com/pranaysy/ETCPy.git
   ```

5. **Download the Data**  
   The data required for this project should be placed in the `data/` folder. Refer to the instructions in the `data/` folder for details on downloading and organizing the data.

6. **Run the Notebooks**  
   Open Jupyter Notebook and execute the notebooks in sequence to replicate the results:
   ```bash
   jupyter notebook
   ```

## Repository Structure

```
|-- data/               # Folder for storing dataset
    |-- files/            # Raw dataset files
|-- notebooks/          # Jupyter Notebooks for analysis and experiments
|-- src/                # Source code for models and utilities
    |-- models/         # Code for building and training models
    |-- utils/          # Utility functions and helpers
|-- README.md           # Project documentation (this file)
```
