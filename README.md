
# TNTAnalysis
![image](https://github.com/JoeGreiner/TNTAnalysis/assets/24453528/8e2b1467-df7f-498b-afe6-06fa31b22cca)

This repository contains software for the deep learning-based analysis of tunnelling nanotubes (TNT) in reflection microscopy time series. Therefore, TNT tips are predicted using neural networks ([nnUNet](https://github.com/MIC-DKFZ/nnUNet)), and then tracked using [TrackMate](https://www.sciencedirect.com/science/article/pii/S1046202316303346?via%3Dihub). An easy-to-use GUI (capable of batch analysis) is provided to guide the user through the process. While we have written the software with Leica's .lif files in mind, it can be easily adapted to other file formats.

## Installation

Requirements: Windows or Linux (tested on Ubuntu 22.10 and Windows 11 Pro). Modern PyTorch-compatible graphic card (tested on NVIDIA 2080TI, 3090, 4090).
<details>
<summary>Windows</summary>

Steps:
1. Clone/download this repository and navigate to the folder.
``` bash
git clone https://github.com/JoeGreiner/TNTAnalysis.git
cd TNTAnalysis
```
2. Install the conda environment.
```
 conda env create --file conda_env_windows.yml
```
3. Activate the conda environment.
```
conda activate TNTAnalysis
```
4. Install the TNT package. 
```
pip install .
```
5. (Optional) If you get an error message about missing GPU support, install the correct PyTorch version for your system [(Official Pytorch HowTo)](https://pytorch.org/get-started/locally/).
</details>


<details>
<summary>Linux</summary>

Steps:
1. Clone/download this repository and navigate to the folder.
``` bash
git clone https://github.com/JoeGreiner/TNTAnalysis.git
cd TNTAnalysis
```
2. Install the conda environment.
```
 conda env create --file conda_env_linux.yml
```
3. Activate the conda environment.
```
conda activate TNTAnalysis
```
4. Install the TNT package. 
```
pip install .
```
5. (Optional) If you get an error message about missing GPU support, install the correct PyTorch version for your system [(Official Pytorch HowTo)](https://pytorch.org/get-started/locally/).
</details>



## Running Individual Steps

<details>
<summary>Windows</summary>
 The software can be run in two ways: 

 1. Using the terminal:

``` bash
conda activate TNTAnalysis
TNT_1_Prediction.exe
TNT_2_CreateMask.exe
TNT_3_ApplyMask.exe
```
 2. Using shortcuts:

For Windows-based systems, the install script sets up shortcuts in the same folder as the repository. You can run the shortcuts from there, or copy them to a location of your choice.
</details>

<details>
<summary>Linux</summary>
 
The software can be run in two ways: 

 1. Using the terminal:

``` bash
conda activate TNTAnalysis
TNT_1_Prediction
TNT_2_CreateMask
TNT_3_ApplyMask
```
 2. Using shortcuts:

For Linux-based systems, the install script sets up .desktop files for your users. You should be able to find the shortcuts in your application menu, i.e. by using the super key (application search), and typing "TNT_".

[!](https://github.com/JoeGreiner/TNTAnalysis/assets/24453528/9fc4d145-6569-42d6-8126-fafff8c502e0)

</details>

## Overview: Steps

The Analysis process is subdivided into three main steps:

1. **TNT Tip prediction and tracking:** Run the TNT tip prediction and tracking using Trackmate. Visualising the results using Fiji/TrackMate is optional but recommended.
   
<details>
<summary>Video</summary>
 
[!](https://github.com/JoeGreiner/TNTAnalysis/assets/24453528/3b4dba65-bd22-46cb-acc4-756b346d01df)
</details>

2. **(Optional) Create Masking & Apply Masking:** Select a subset of TNT tracks to be included in the analysis using Napari. Optional, but recommended: visualise the results using Fiji/TrackMate.
<details>
<summary>Video</summary>
 
[!](https://github.com/JoeGreiner/TNTAnalysis/assets/24453528/5be41ba6-bec1-4f9b-ad27-7590491739a7)

</details>

## Step 1: TNT Tip prediction and tracking

0. Run TNT_1_Prediction as described in the "Running Individual Steps" section.
1. Set up your parameters in the GUI.
<details>
<summary>Parameters</summary> 
 
* **nnUNet dataset number**: The dataset number for the nnUNet model. We provide a pre-trained model for the TNT tip prediction. The dataset number is 301. If you want to use the pre-trained model, press the "Download Model" button. It will download the model and set the dataset number automatically.
* **nnUNet model folder**: This is the folder where the nnUNet modela are saved ($nnUNet_results). If you want to use the pre-trained model, press the "Download Model" button. It will download the model and set the path automatically.
* **Disable test-time augmentation**: nnUnet uses test-time augmentation by default. While we recommend using it, you can disable it here for faster computation.
* **Sliding window size**: The size of the sliding window used for the prediction. The default and recommended value is 0.5.
* **Fiji path**: The path to the Fiji folder (e.g. C:\Fiji.app). You can also press the "Download and link Fiji" button to download Fiji and set the path automatically.

</details>

2. Drag and drop in either a folder with .lif files or a single .lif file. Press the "Start" button to start the prediction and tracking. The software will create a folder with the same name as the .lif file in the same directory. We also provide a test_file accessible by pressing the 'download test file' button.

## Step 2: Create Masking

0. Run TNT_2_CreateMask as described in the "Running Individual Steps" section.
1. Drag and drop in the timeseries. Optional: add additional files (e.g. fluorescence of virus-positive cells). 
2. Using widget 0 'Add Labels', create a new layer by using the widget on the top right. The mask will be created in this layer.
3. Optional: Make sure the additional images are registered to the timeseries, for rotations, use widget 1 'Rotate Layer' on the right (useful to match LAS X Navigator's inherent rotation).
4. Optional: If you use photon-counting data, you may want to use widget 2 'Gaussian Blur' to aid visualisation.
5. Select the new layer and draw the mask using the brush tool.
6. Using widget 3 'Save Mask Layer,' select an output directory and press run. The mask will be saved as a .tif file with the same prefix as the time series.

## Step 3: Apply Masking

0. Run TNT_3_ApplyMask as described in the "Running Individual Steps" section.
1. Drag and drop in the created mask from Step 2 and the track files from Step 1. Press Start.
2. Run the processing. This will create two sets of TrackMate files (.xml) and .xlsx files: One with all data that is inside the mask, and one with all the data outside the mask.

## Acknowledgements

This repository is based on an MSc thesis of Gregor Stief. We are further really thankful for the fantastic software packages we build upon. These include, but are not limited to, the following packages:
* [nnUNet](https://github.com/MIC-DKFZ/nnUNet)
* [TrackMate](https://www.sciencedirect.com/science/article/pii/S1046202316303346?via%3Dihub)
* [Napari](https://napari.org/stable/)
* [ImageJ/Fiji](https://fiji.sc/)
* [PyTorch](https://pytorch.org/)
* [Numpy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [ITK](https://itk.org/)
* [SciPy](https://www.scipy.org/)
* [scikit-learn](https://scikit-learn.org/stable/)
