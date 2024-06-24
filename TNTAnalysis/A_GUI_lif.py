import glob
import logging
import subprocess
from os.path import join
from typing import List
import torch

from PyQt5.QtCore import QThread, Qt, QSettings
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QPushButton, QFileDialog, QApplication, QLabel, \
    QTextEdit, QHBoxLayout, QCheckBox, QDoubleSpinBox, QMessageBox

from TNTAnalysis.fileIO.download_files import download_fiji, download_model, download_testfiles
from TNTAnalysis.fileIO.git import check_if_git_uptodate
from TNTAnalysis.fileIO.lif_preperation import prepare_single_lif_to_nii, combine_lif_and_matching_prediction
import os

from TNTAnalysis.tracking.trackmate import track_everything_v2


def run_analysis(list_tifs: list,
                 path_to_nnunet_model: str,
                 path_to_output_folder: str,
                 fiji_path: str,
                 frame_duration: int,
                 nnu_dataset_number: int,
                 overlap: float = 0.5,
                 continue_prediction: bool = False,
                 disable_tta: bool = False,
                 multiprocessing=True,
                 number_processes=1):
    if not os.path.exists(path_to_nnunet_model):
        raise FileNotFoundError(f"Could not find the nnUNet model at {path_to_nnunet_model}")
    if not os.path.exists(path_to_output_folder):
        raise FileNotFoundError(f"Could not find the output folder at {path_to_output_folder}")
    if not os.path.exists(fiji_path):
        raise FileNotFoundError(f"Could not find the Fiji executable at {fiji_path}")
    if frame_duration:
        assert frame_duration > 0, "The frames duration need to be a positive number"
    assert nnu_dataset_number > 0, "The nnU-Net dataset number needs to be a positive number"

    for img_path in list_tifs:
        prepare_single_lif_to_nii(img_path, path_to_output_folder)

    # start nnUNet Inference
    output_folder_predictions = join(path_to_output_folder, "inference_predictions")
    if not os.path.exists(output_folder_predictions):
        os.makedirs(output_folder_predictions)

    # calls the intern function predict_entry_point that is the command line command nnUNetv2_predict
    os.environ["nnUNet_results"] = path_to_nnunet_model
    os.environ["nnUNet_raw"] = ""
    os.environ["nnUNet_preprocessed"] = ""
    # set_environment_vars()
    # set_fiji_path()
    os.environ["fiji_path"] = fiji_path

    command_str = (
        f'nnUNetv2_predict -i "{path_to_output_folder}" -o "{output_folder_predictions}" -d {nnu_dataset_number} -c 3d_fullres'
        f" -step_size {overlap}")
    if disable_tta:
        command_str += " --disable_tta"
    if continue_prediction:
        command_str += " --continue_prediction"

    logging.info(f"Running command: {command_str}")
    subprocess.run(command_str, shell=True)

    # combined the nii.gz prediction with the lif file into a single tif with correct header
    # so that the .xml can be easily visualised in trackmate, AND has already all the correct metadata
    # about time and physical resolutions
    # have xml and tifs at the same folder
    output_folder_tracking = join(path_to_output_folder, "output_tracking")
    list_of_combined_files = []
    for img_path in list_tifs:
        combined_file_path = combine_lif_and_matching_prediction(lif_path=img_path,
                                                                 prediction_folder=output_folder_predictions,
                                                                 output_folder=output_folder_tracking)
        list_of_combined_files.append(combined_file_path)

    # track result
    # joblist = get_all_predictions(output_folder_predictions)
    config = {"INTERACTIVE": False,  # Initialize ImageJ
              "PRINTING": False,  # Do you want additional information? Then turn on
              "SAVE_MASK": False,  # Do you want to save the corrected Mask? Then turn on
              "SAVE_XLSX": True,  # Do you want to save xlsx file (main output)? Then turn on
              "SAVE_XML": True,  # Do you want to save the XML? Then turn on
              "ACTIVATE_BIOFILTER": False}  # Do you want only tracks over the length of 4? Then turn on

    # track_everything(joblist, config, output_folder=output_folder_tracking, lif_files_folder=lif_folder)
    track_everything_v2(list_of_combined_files, config, output_folder=output_folder_tracking)

    # open visualisation (without ground truth as this is new data)
    # visualize_multiple_images_napari(output_folder_tracking, output_folder_dataset)


def load_files_recursively(path_to_base_folder: str, level: int = 4) -> List[str]:
    """
    This function loads all the tif files from the specified directories up to a certain depth level.

    Parameters:
    path_to_base_folder (str): The base directory where the image files are located.
    level (int): The depth of the subdirectories from which to load the image files. Default is 4.

    Returns:
    list[str]: A list of filepaths for each tif file in the specified directories.
    """
    logging.info(f'Loading files from {path_to_base_folder}')

    if not os.path.isdir(path_to_base_folder):
        if path_to_base_folder.endswith('.lif'):
            files = [path_to_base_folder]
        else:
            files = []
    else:
        files = []
        for i in range(level):
            files += glob.glob(os.path.join(path_to_base_folder, '*/' * i + '*.lif'), recursive=True)
        logging.info(f'Loaded {len(files)} files')
    return files


class AnalysisThread(QThread):
    def __init__(self, path_imgs_to_analyse: list,
                 path_to_nnunet_model: str,
                 nnunet_dataset_number: str,
                 path_to_output_folder: str,
                 fiji_path: str,
                 frame_duration: str = None,
                 overlap: float = 0.5,
                 continue_prediction: bool = False,
                 disable_tta: bool = False,
                 multiprocessing: bool = True,
                 number_processes: int = 1):
        super().__init__()
        nnunet_dataset_number = int(nnunet_dataset_number)
        if frame_duration:
            frame_duration = float(frame_duration)
            assert frame_duration > 0, "The frames per second need to be a positive number"

        if not os.path.exists(path_to_output_folder):
            os.makedirs(path_to_output_folder)

        for path_img in path_imgs_to_analyse:
            assert os.path.exists(path_img), f"Could not find the image file at {path_img}"
        assert os.path.exists(path_to_nnunet_model), f"Could not find the nnUNet model at {path_to_nnunet_model}"
        assert os.path.exists(path_to_output_folder), f"Could not find the output folder at {path_to_output_folder}"
        assert nnunet_dataset_number > 0, "The nnUNet dataset number needs to be a positive number"
        assert os.path.exists(fiji_path), f"Could not find the Fiji executable at {fiji_path}"
        assert number_processes > 0, "The number of processes needs to be a positive number"
        assert overlap > 0, "The overlap needs to be a positive number"
        assert 0 < overlap < 1, "The overlap needs to be between 0 and 1"
        assert isinstance(multiprocessing, bool), "The multiprocessing flag needs to be a boolean"
        assert isinstance(continue_prediction, bool), "The continue prediction flag needs to be a boolean"
        assert isinstance(disable_tta, bool), "The disable TTA flag needs to be a boolean"

        self.path_imgs_to_analyse = path_imgs_to_analyse
        self.path_to_nnunet_model = path_to_nnunet_model
        self.nnunet_dataset_number = nnunet_dataset_number
        self.path_to_output_folder = path_to_output_folder
        self.fiji_path = fiji_path
        self.frame_duration = frame_duration
        self.multiprocessing = multiprocessing
        self.number_processes = number_processes
        self.overlap = overlap
        self.continue_prediction = continue_prediction
        self.disable_tta = disable_tta

    def run(self):
        logging.info('Starting analysis')

        run_analysis(list_tifs=self.path_imgs_to_analyse,
                     path_to_nnunet_model=self.path_to_nnunet_model,
                     path_to_output_folder=self.path_to_output_folder,
                     fiji_path=self.fiji_path,
                     frame_duration=self.frame_duration,
                     overlap=self.overlap,
                     continue_prediction=self.continue_prediction,
                     disable_tta=self.disable_tta,
                     multiprocessing=self.multiprocessing,
                     nnu_dataset_number=self.nnunet_dataset_number,
                     number_processes=self.number_processes)

        logging.info('Analysis finished')


class GUI_TNT_Analysis(QWidget):
    def __init__(self):
        super(GUI_TNT_Analysis, self).__init__()

        if torch.cuda.is_available():
            logging.info('GPU available')
        else:
            logging.info('No GPU available')
            msg = QMessageBox()
            msg.setWindowTitle("No GPU Support Available")
            msg.setText("No GPU support available. Please make sure to install GPU-supported version of PyTorch.")
            msg.exec()
            exit()




        self.settings = QSettings("JoeGreiner", "TNT_Analysis_GUI")
        self.printSettings()
        self.initUI()
        self.printSettings()
        self.loadSettings()

    def initUI(self):
        self.setWindowTitle('Automatic TNT Analysis')
        figure_width = 800
        figure_height = 400
        self.setGeometry(300, 300, figure_width, figure_height)

        layout = QVBoxLayout()
        self.setLayout(layout)

        ## Output Folder Selection
        self.row_layout = QHBoxLayout()
        layout.addLayout(self.row_layout)

        self.output_folder_label = QLabel("Output Folder:", self)
        self.row_layout.addWidget(self.output_folder_label)
        self.output_folder = QLineEdit()
        self.output_folder.setReadOnly(True)
        self.row_layout.addWidget(self.output_folder)

        self.browseButton = QPushButton("Select Output Folder", self)
        self.browseButton.clicked.connect(self.showFileDialogOutputPath)
        self.row_layout.addWidget(self.browseButton)

        ## Framerate Selection and nnu dataset number
        self.row_layout_frameduration = QHBoxLayout()
        layout.addLayout(self.row_layout_frameduration)

        # self.frameduration_label = QLabel("Frame Duration (sec):", self)
        # self.row_layout_frameduration.addWidget(self.frameduration_label)
        # self.frame_duration = QLineEdit()
        # self.row_layout_frameduration.addWidget(self.frame_duration)
        # self.frame_duration.textChanged.connect(self.saveSettingsFrameDuration)

        self.nnudataset_label = QLabel("nnU-Net Dataset Number:", self)
        self.row_layout_frameduration.addWidget(self.nnudataset_label)
        self.nnudataset = QLineEdit()
        self.row_layout_frameduration.addWidget(self.nnudataset)
        self.nnudataset.textChanged.connect(self.saveSettingsNNUNetDataset)
        self.download_test_file_button = QPushButton("Download Test File", self)
        self.download_test_file_button.clicked.connect(self.download_test_file)
        self.row_layout_frameduration.addWidget(self.download_test_file_button)

        ## NNU-Net Path Selection
        self.row_layout_nnunet = QHBoxLayout()
        layout.addLayout(self.row_layout_nnunet)

        self.nnunet_label = QLabel("nnU-Net Model Folder", self)
        self.row_layout_nnunet.addWidget(self.nnunet_label)
        self.nnunet_folder = QLineEdit()
        self.nnunet_folder.setReadOnly(True)
        self.row_layout_nnunet.addWidget(self.nnunet_folder)

        self.browseButton_nnunet = QPushButton("Select nnU-Net Model Folder", self)
        self.browseButton_nnunet.clicked.connect(self.showFileDialogNNUNetPath)
        self.row_layout_nnunet.addWidget(self.browseButton_nnunet)

        self.download_model_button = QPushButton("Download Model", self)
        self.download_model_button.clicked.connect(self.download_model_from_web)
        self.row_layout_nnunet.addWidget(self.download_model_button)


        # hbox layout: checkbox disable tta, checkbox continue prediction, float box one decimal for step size
        self.row_layout_nnunet_options = QHBoxLayout()
        layout.addLayout(self.row_layout_nnunet_options)

        self.disable_tta = QCheckBox("Disable Test-Time Augmentation", self)
        self.row_layout_nnunet_options.addWidget(self.disable_tta)
        self.disable_tta.stateChanged.connect(self.saveSettingsTTA)

        self.continue_prediction = QCheckBox("Skip Already-Predicted Stacks", self)
        self.row_layout_nnunet_options.addWidget(self.continue_prediction)
        self.continue_prediction.stateChanged.connect(self.saveSettingsContinuePrediction)

        self.step_size = QDoubleSpinBox()
        self.row_layout_nnunet_options.addWidget(self.step_size)
        self.step_size.setRange(0.1, 1.0)
        self.step_size.setSingleStep(0.1)
        self.step_size.setDecimals(1)
        self.step_size.setValue(0.5)
        self.step_size.setPrefix("Sliding Window Overlap: ")
        self.step_size.valueChanged.connect(self.saveSettingsStepSize)

        # restore default prediction button
        self.restore_default_prediction = QPushButton("Restore Default Prediction Settings", self)
        self.restore_default_prediction.clicked.connect(self.restoreDefaultPredictionSettings)
        self.row_layout_nnunet_options.addWidget(self.restore_default_prediction)

        ## Fiji Path Selection
        self.row_layout_fiji = QHBoxLayout()
        layout.addLayout(self.row_layout_fiji)

        self.fiji_label = QLabel("Fiji Executable Path", self)
        self.row_layout_fiji.addWidget(self.fiji_label)
        self.fiji_folder = QLineEdit()
        self.fiji_folder.setReadOnly(True)
        self.row_layout_fiji.addWidget(self.fiji_folder)

        self.browseButton_fiji = QPushButton("Select Fiji Executable Path", self)
        self.browseButton_fiji.clicked.connect(self.showFileDialogFijiPath)
        self.row_layout_fiji.addWidget(self.browseButton_fiji)

        self.download_and_link_fiji_button = QPushButton("Download and Link Fiji", self)
        self.download_and_link_fiji_button.clicked.connect(self.download_and_link_fiji)
        self.row_layout_fiji.addWidget(self.download_and_link_fiji_button)

        ## Drop Folder for TIFs Selection
        self.dropLabel = QLabel("Drop folder with Lifs here", self)
        self.dropLabel.setAlignment(Qt.AlignCenter)
        font = QFont("Arial", 20, QFont.Bold)
        self.dropLabel.setFont(font)
        self.dropLabel.setStyleSheet("background-color: #f0f0f0; border: 2px dashed #888888; color: #888888;")
        self.dropLabel.setGeometry(0, 0, figure_width, 300)
        layout.addWidget(self.dropLabel)

        self.pathLineEdit = QLineEdit()
        self.pathLineEdit.hide()

        ## Start Analysis Button
        self.startAnalysisButton = QPushButton('Start Analysis', self)
        self.startAnalysisButton.hide()
        self.startAnalysisButton.clicked.connect(self.startAnalysis)
        self.startAnalysisButton.setStyleSheet("background-color: 10c46a;")

        self.progressLabel = QLabel('Analysing...')
        self.progressLabel.hide()
        self.progressLabel.setAlignment(Qt.AlignCenter)

        self.fileCountLabel = QLabel('')
        self.fileListTextEdit = QTextEdit()

        self.widgets_to_hide = [self.pathLineEdit, self.startAnalysisButton,
                                self.fileCountLabel, self.fileListTextEdit]

        self.setAcceptDrops(True)
        self.is_analysis_running = False

        self.files_to_analyse = []

    def printSettings(self):
        logging.info(f"Settings:")
        for key in self.settings.allKeys():
            logging.info(f"{key}: {self.settings.value(key)}")

    def loadSettings(self):
        output_path = self.settings.value("outputPath", "")
        self.output_folder.setText(output_path)
        # frame_duration = self.settings.value("frameDuration", "")
        # self.frame_duration.setText(frame_duration)
        nnunet_path = self.settings.value("nnunetPath", "")
        self.nnunet_folder.setText(nnunet_path)
        fiji_path = self.settings.value("fijiPath", "")
        self.fiji_folder.setText(fiji_path)
        nnu_dataset = self.settings.value("nnuDatasetNumber", "")
        self.nnudataset.setText(nnu_dataset)
        disable_tta = self.settings.value("disableTTA", "")
        if disable_tta == 'true':
            self.disable_tta.setChecked(True)
        elif disable_tta == 'false':
            self.disable_tta.setChecked(False)
        continue_prediction = self.settings.value("continuePrediction", "")
        if continue_prediction == 'true':
            self.continue_prediction.setChecked(True)
        elif continue_prediction == 'false':
            self.continue_prediction.setChecked(False)
        step_size = self.settings.value("stepSize", "")
        if step_size == '':
            step_size = 0.5
        self.step_size.setValue(float(step_size))

    def showFileDialogOutputPath(self):
        dir_path = QFileDialog.getExistingDirectory(self, 'Select directory')
        if dir_path:
            self.output_folder.setText(dir_path)
            self.settings.setValue("outputPath", dir_path)

    def showFileDialogFijiPath(self):
        folder_to_fiji = QFileDialog.getExistingDirectory(self, 'Select Fiji Executable')
        if folder_to_fiji:
            self.fiji_folder.setText(folder_to_fiji)
            self.settings.setValue("fijiPath", folder_to_fiji)

    def showFileDialogNNUNetPath(self):
        dir_path = QFileDialog.getExistingDirectory(self, 'Select directory')
        if dir_path:
            self.nnunet_folder.setText(dir_path)
            self.settings.setValue("nnunetPath", dir_path)

    def saveSettingsNNUNetDataset(self):
        self.settings.setValue("nnuDatasetNumber", self.nnudataset.text())

    def saveSettingsTTA(self):
        self.settings.setValue("disableTTA", self.disable_tta.isChecked())

    def saveSettingsContinuePrediction(self):
        self.settings.setValue("continuePrediction", self.continue_prediction.isChecked())

    def saveSettingsStepSize(self):
        self.settings.setValue("stepSize", self.step_size.value())

    # def saveSettingsFrameDuration(self):
    #     self.settings.setValue("frameDuration", self.frame_duration.text())

    def restoreDefaultPredictionSettings(self):
        self.disable_tta.setChecked(False)
        self.continue_prediction.setChecked(False)
        self.step_size.setValue(0.5)

    def widgetIsAdded(self, object):
        # Check if the button is in the layout
        index = self.layout().indexOf(object)
        return index != -1

    def showDialog(self):
        dir_path = QFileDialog.getExistingDirectory(self, 'Select directory')
        self.pathLineEdit.setText(dir_path)
        self.loadFiles(dir_path)

    def showProgressBar(self):
        if not self.widgetIsAdded(self.progressLabel):
            self.layout().addWidget(self.progressLabel)
        self.progressLabel.show()

    def hideProgressBar(self):
        self.progressLabel.hide()

    def download_and_link_fiji(self):
        tmp_directory = os.path.join(os.path.expanduser("~"), "TNTAnalysis")

        msg = QMessageBox()
        msg.setWindowTitle("Downloading Fiji")
        msg.setText("Downloading Fiji may take a while. Please be patient.")
        msg.exec()

        link_to_fiji = download_fiji(tmp_directory)
        self.fiji_folder.setText(link_to_fiji)
        if link_to_fiji:
            self.fiji_folder.setText(link_to_fiji)
            self.settings.setValue("fijiPath", link_to_fiji)

    def download_model_from_web(self):
        tmp_directory = os.path.join(os.path.expanduser("~"), "TNTAnalysis")

        msg = QMessageBox()
        msg.setWindowTitle("Downloading Model")
        msg.setText("Downloading the model may take a while. Please be patient.")
        msg.exec()

        model_folder = download_model(tmp_directory)
        self.nnunet_folder.setText(model_folder)
        if model_folder:
            self.nnunet_folder.setText(model_folder)
            self.settings.setValue("nnunetPath", model_folder)
        self.nnudataset.setText("301")
        self.settings.setValue("nnuDatasetNumber", "301")

    def download_test_file(self):
        tmp_directory = os.path.join(os.path.expanduser("~"), "TNTAnalysis", "testfiles")
        if not os.path.exists(tmp_directory):
            os.makedirs(tmp_directory)

        msg = QMessageBox()
        msg.setWindowTitle("Downloading Test File")
        msg.setText("Downloading the test file may take a while. Please be patient.")
        msg.exec()

        path_to_testfile = download_testfiles(tmp_directory)
        self.pathLineEdit.setText(path_to_testfile)
        self.loadFiles(path_to_testfile, namefilter='2023-11-16_1_GFP-ctrl_TREK1-transfection.lif')



    def startAnalysis(self):
        # Set the analysis state to True
        self.is_analysis_running = True

        # Change the background color to yellow
        self.setStyleSheet("background-color: #bce0ce;")


        # frame_duration = self.frame_duration.text()
        # if frame_duration == "-1":
        #     frame_duration = None

        self.hideWidgets()  # Hide all visible widgets
        self.showProgressBar()  # Show progress bar and label
        self.analysisThread = AnalysisThread(path_imgs_to_analyse=self.files_to_analyse,
                                             path_to_nnunet_model=self.nnunet_folder.text(),
                                             nnunet_dataset_number=self.nnudataset.text(),
                                             path_to_output_folder=self.output_folder.text(),
                                             fiji_path=self.fiji_folder.text(),
                                             overlap=self.step_size.value(),
                                             continue_prediction=self.continue_prediction.isChecked(),
                                             disable_tta=self.disable_tta.isChecked(),
                                             # frame_duration=frame_duration,
                                             multiprocessing=True)  # TODO do multiprocessing gui
        self.analysisThread.finished.connect(self.hideProgressBar)
        self.analysisThread.finished.connect(self.analysisFinished)
        self.analysisThread.finished.connect(self.resetApplication)  # Show all hidden widgets when analysis is finished
        self.analysisThread.start()




    def analysisFinished(self):
        logging.info('Analysis finished')
        # Reset the analysis state to False
        self.is_analysis_running = False

        # Remove the background color style sheet
        self.setStyleSheet("")

        # message box that it should be analysed using trackmate
        msg = QMessageBox()
        msg.setWindowTitle("TNT Analysis finished")
        msg.setText("TNT Analysis finished. Please proofread the generated .xml file using Trackmate/Fiji.")
        msg.exec()


    def hideWidgets(self):
        logging.info('Hiding widgets')
        for widget in self.widgets_to_hide:
            widget.hide()

    def resetApplication(self):
        logging.info('Resetting application')
        for widget in self.widgets_to_hide:
            widget.show()
        # reset widgets
        self.fileListTextEdit.setText('')
        self.fileListTextEdit.hide()
        self.fileCountLabel.setText('')
        self.fileCountLabel.hide()
        self.pathLineEdit.setText('')
        self.pathLineEdit.hide()
        self.startAnalysisButton.hide()
        self.dropLabel.show()

    def loadFiles(self, directory_path, namefilter=''):
        logging.info(f'Loading files from {directory_path}')
        files = load_files_recursively(directory_path)

        if len(namefilter) > 0:
            files = [f for f in files if namefilter in f]

        if len(files) > 0:
            if not self.widgetIsAdded(self.fileListTextEdit):
                self.layout().addWidget(self.fileListTextEdit)
            if not self.widgetIsAdded(self.fileCountLabel):
                self.layout().addWidget(self.fileCountLabel)
            self.fileCountLabel.setText(f'Number of files found: {len(files)}')
            self.fileListTextEdit.setText('\n'.join(files))
            self.fileListTextEdit.show()
            if not self.widgetIsAdded(self.startAnalysisButton):
                self.layout().addWidget(self.startAnalysisButton)
            if not self.widgetIsAdded(self.pathLineEdit):
                self.layout().addWidget(self.pathLineEdit)
            self.startAnalysisButton.show()

            self.files_to_analyse = files



    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.accept()
            self.dropLabel.setText("Drop Folder Here")
            self.dropLabel.setStyleSheet("background-color: #bce0ce; border: 2px dashed #888888; color: #10c46a;")
        else:
            e.ignore()

    def dragLeaveEvent(self, e):
        self.dropLabel.setText("Drop Folder Here")
        self.dropLabel.setStyleSheet("background-color: #f0f0f0; border: 2px dashed #888888; color: #888888;")

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            if url.isLocalFile():
                path = str(url.toLocalFile())
                self.pathLineEdit.setText(path)
                self.loadFiles(path)
                self.dropLabel.setStyleSheet("background-color: #f0f0f0; border: 2px dashed #888888; color: #888888;")
                self.dropLabel.hide()


def run_GUI():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')

    app = QApplication([])

    git_is_uptodate = check_if_git_uptodate()
    if not git_is_uptodate:
        logging.info("The local git repository is not up to date with the remote repository. Please update the git repository.")
        msg = QMessageBox()
        msg.setWindowTitle("Git Repository not up to date")
        msg.setText("The local git repository is not up to date with the remote repository. Please update the git repository!")
        msg.exec()

    ex = GUI_TNT_Analysis()
    ex.show()
    app.exec()

if __name__ == '__main__':
    run_GUI()
