from collections import Counter
import logging
from math import isfinite
import subprocess
from os.path import join
from tempfile import TemporaryDirectory
from typing import List
import torch

from PyQt6.QtCore import QThread, Qt, QSettings, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QFont, QPalette
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QPushButton, QFileDialog, QApplication, QLabel, \
    QHBoxLayout, QCheckBox, QDoubleSpinBox, QSpinBox, QMessageBox, QProgressDialog, QTableWidget, QTableWidgetItem, \
    QAbstractItemView, QHeaderView

from TNTAnalysis.fileIO.download_files import download_fiji, download_model, download_testfiles
from TNTAnalysis.fileIO.git import check_if_git_uptodate
from TNTAnalysis.fileIO.lif_preperation import prepare_single_lif_to_nii, combine_lif_and_matching_prediction
from TNTAnalysis.fileIO.lif import (open_lif, get_lif_series_metadata, get_frame_timing,
                                   apply_frame_duration_override, FrameTimingError)
import os

from TNTAnalysis.tracking.trackmate import track_everything_v2


def get_prediction_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def validate_prediction_processes(preprocessing_processes, export_processes):
    if preprocessing_processes < 0 or export_processes < 0:
        raise ValueError('Prediction worker counts cannot be negative.')
    if (preprocessing_processes == 0) != (export_processes == 0):
        raise ValueError('Set both preprocessing and export workers to 0 for sequential processing, '
                         'or set both to at least 1.')


def run_analysis(list_tifs: list,
                 path_to_nnunet_model: str,
                 path_to_output_folder: str,
                 fiji_path: str,
                 frame_duration: float,
                 nnu_dataset_number: int,
                 overlap: float = 0.5,
                 continue_prediction: bool = False,
                 disable_tta: bool = False,
                 perform_everything_on_device: bool = False,
                 preprocessing_processes: int = 1,
                 export_processes: int = 1):
    validate_prediction_processes(preprocessing_processes, export_processes)
    if not os.path.exists(path_to_nnunet_model):
        raise FileNotFoundError(f"Could not find the nnUNet model at {path_to_nnunet_model}")
    if not os.path.exists(path_to_output_folder):
        raise FileNotFoundError(f"Could not find the output folder at {path_to_output_folder}")
    if not os.path.exists(fiji_path):
        raise FileNotFoundError(f"Could not find the Fiji executable at {fiji_path}")
    if frame_duration is not None and (not isfinite(frame_duration) or frame_duration <= 0):
        raise ValueError('The frame interval must be a finite positive number of seconds.')
    assert nnu_dataset_number > 0, "The nnU-Net dataset number needs to be a positive number"

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

    with TemporaryDirectory(prefix='nnunet-input-', dir=path_to_output_folder) as input_folder:
        for img_path in list_tifs:
            prepare_single_lif_to_nii(img_path, input_folder)

        command_str = (
            f'nnUNetv2_predict -i "{input_folder}" -o "{output_folder_predictions}" -d {nnu_dataset_number} -c 3d_fullres'
            f" -step_size {overlap} -device {get_prediction_device()}"
            f" -npp {preprocessing_processes} -nps {export_processes}")
        if not perform_everything_on_device:
            command_str += " --not_on_device"
        if disable_tta:
            command_str += " --disable_tta"
        if continue_prediction:
            command_str += " --continue_prediction"

        logging.info(f"Running command: {command_str}")
        subprocess.run(command_str, shell=True, check=True)

    # combined the nii.gz prediction with the lif file into a single tif with correct header
    # so that the .xml can be easily visualised in trackmate, AND has already all the correct metadata
    # about time and physical resolutions
    # have xml and tifs at the same folder
    output_folder_tracking = join(path_to_output_folder, "output_tracking")
    list_of_combined_files = []
    for img_path in list_tifs:
        combined_file_path = combine_lif_and_matching_prediction(lif_path=img_path,
                                                                 prediction_folder=output_folder_predictions,
                                                                 output_folder=output_folder_tracking,
                                                                 frame_duration=frame_duration)
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


def load_files_recursively(path_to_base_folder: str, level: int = None) -> List[str]:
    """Find LIF files in a file or folder, including all subfolders unless a depth limit is given."""
    logging.info(f'Loading files from {path_to_base_folder}')
    path_to_base_folder = os.path.abspath(path_to_base_folder)
    if os.path.isfile(path_to_base_folder):
        return [path_to_base_folder] if path_to_base_folder.lower().endswith('.lif') else []
    files = []
    for directory, subdirectories, names in os.walk(path_to_base_folder):
        subdirectories.sort()
        relative_path = os.path.relpath(directory, path_to_base_folder)
        depth = 0 if relative_path == '.' else len(relative_path.split(os.sep))
        if level is not None and depth >= level:
            subdirectories[:] = []
            continue
        files.extend(join(directory, name) for name in sorted(names)
                     if name.lower().endswith('.lif') and os.path.isfile(join(directory, name)))
    logging.info(f'Loaded {len(files)} files')
    return files


class DownloadThread(QThread):
    progress = pyqtSignal(int)

    def __init__(self, download_function, directory, parent=None):
        super().__init__(parent)
        self.download_function = download_function
        self.directory = directory
        self.result = None
        self.error = None
        self.last_progress = None

    def update_progress(self, current, total, width):
        progress = min(100, current * 100 // total) if total > 0 else -1
        if progress != self.last_progress:
            self.progress.emit(progress)
            self.last_progress = progress

    def run(self):
        try:
            self.result = self.download_function(self.directory, progress_callback=self.update_progress)
        except Exception as error:
            logging.exception('Download failed')
            self.error = str(error)


class DownloadProgressDialog(QProgressDialog):
    def __init__(self, title, download_function, directory, parent=None):
        super().__init__(f'Connecting...\n{directory}', None, 0, 0, parent)
        self.setWindowTitle(title)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.CustomizeWindowHint | Qt.WindowType.WindowTitleHint)
        self.setAutoClose(False)
        self.setAutoReset(False)
        self.directory = directory
        self.download_thread = DownloadThread(download_function, directory, self)
        self.download_thread.progress.connect(self.update_progress)
        self.download_thread.finished.connect(self.accept)

    def update_progress(self, progress):
        self.setRange(0, 100 if 0 <= progress < 100 else 0)
        if 0 <= progress < 100:
            self.setValue(progress)
        status = 'Extracting...' if progress == 100 else 'Downloading...'
        self.setLabelText(f'{status}\n{self.directory}')

    def reject(self):
        if not self.download_thread.isRunning():
            super().reject()

    def closeEvent(self, event):
        if self.download_thread.isRunning():
            event.ignore()
        else:
            super().closeEvent(event)


class AnalysisThread(QThread):
    def __init__(self, path_imgs_to_analyse: list,
                 path_to_nnunet_model: str,
                 nnunet_dataset_number: str,
                 path_to_output_folder: str,
                 fiji_path: str,
                 frame_duration: float = None,
                 overlap: float = 0.5,
                 continue_prediction: bool = False,
                 disable_tta: bool = False,
                 perform_everything_on_device: bool = False,
                 preprocessing_processes: int = 1,
                 export_processes: int = 1):
        super().__init__()
        validate_prediction_processes(preprocessing_processes, export_processes)
        nnunet_dataset_number = int(nnunet_dataset_number)
        if frame_duration is not None:
            frame_duration = float(frame_duration)
            if not isfinite(frame_duration) or frame_duration <= 0:
                raise ValueError('The frame interval must be a finite positive number of seconds.')

        if not os.path.exists(path_to_output_folder):
            os.makedirs(path_to_output_folder)

        for path_img in path_imgs_to_analyse:
            assert os.path.exists(path_img), f"Could not find the image file at {path_img}"
        assert os.path.exists(path_to_nnunet_model), f"Could not find the nnUNet model at {path_to_nnunet_model}"
        assert os.path.exists(path_to_output_folder), f"Could not find the output folder at {path_to_output_folder}"
        assert nnunet_dataset_number > 0, "The nnUNet dataset number needs to be a positive number"
        assert os.path.exists(fiji_path), f"Could not find the Fiji executable at {fiji_path}"
        assert overlap > 0, "The overlap needs to be a positive number"
        assert 0 < overlap < 1, "The overlap needs to be between 0 and 1"
        assert isinstance(continue_prediction, bool), "The continue prediction flag needs to be a boolean"
        assert isinstance(disable_tta, bool), "The disable TTA flag needs to be a boolean"

        self.path_imgs_to_analyse = path_imgs_to_analyse
        self.path_to_nnunet_model = path_to_nnunet_model
        self.nnunet_dataset_number = nnunet_dataset_number
        self.path_to_output_folder = path_to_output_folder
        self.fiji_path = fiji_path
        self.frame_duration = frame_duration
        self.perform_everything_on_device = perform_everything_on_device
        self.preprocessing_processes = preprocessing_processes
        self.export_processes = export_processes
        self.overlap = overlap
        self.continue_prediction = continue_prediction
        self.disable_tta = disable_tta
        self.error = None

    def run(self):
        logging.info('Starting analysis')
        self.error = None
        try:
            run_analysis(list_tifs=self.path_imgs_to_analyse,
                         path_to_nnunet_model=self.path_to_nnunet_model,
                         path_to_output_folder=self.path_to_output_folder,
                         fiji_path=self.fiji_path,
                         frame_duration=self.frame_duration,
                         overlap=self.overlap,
                         continue_prediction=self.continue_prediction,
                         disable_tta=self.disable_tta,
                         nnu_dataset_number=self.nnunet_dataset_number,
                         perform_everything_on_device=self.perform_everything_on_device,
                         preprocessing_processes=self.preprocessing_processes,
                         export_processes=self.export_processes)
        except Exception as error:
            self.error = f'{type(error).__name__}: {error}'
            logging.exception('Analysis failed')
        else:
            logging.info('Analysis finished')


class GUI_TNT_Analysis(QWidget):
    def __init__(self):
        super(GUI_TNT_Analysis, self).__init__()

        logging.info(f'Prediction device: {get_prediction_device()}')
        self.settings = QSettings("JoeGreiner", "TNT_Analysis_GUI")
        self.printSettings()
        self.initUI()
        self.printSettings()
        self.loadSettings()

    def initUI(self):
        self.setWindowTitle('Automatic TNT Analysis')
        figure_width = 1300
        figure_height = 650
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

        self.open_results_button = QPushButton("Open Results Folder", self)
        self.open_results_button.setEnabled(False)
        self.open_results_button.clicked.connect(self.openResultsFolder)
        self.row_layout.addWidget(self.open_results_button)
        self.output_folder.textChanged.connect(self.updateResultsFolderButton)

        ## NNU dataset number
        self.row_layout_frameduration = QHBoxLayout()
        layout.addLayout(self.row_layout_frameduration)

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
        self.step_size.setRange(0.1, 0.9)
        self.step_size.setSingleStep(0.1)
        self.step_size.setDecimals(1)
        self.step_size.setValue(0.5)
        self.step_size.setPrefix("Sliding Window Step Size: ")
        self.step_size.setToolTip(
            "Step size as a fraction of the window size: 0.1 means about 90% overlap, 0.5 about 50%.")
        self.step_size.valueChanged.connect(self.saveSettingsStepSize)

        # restore default prediction button
        self.restore_default_prediction = QPushButton("Restore Default Prediction Settings", self)
        self.restore_default_prediction.clicked.connect(self.restoreDefaultPredictionSettings)
        self.row_layout_nnunet_options.addWidget(self.restore_default_prediction)

        prediction_memory_layout = QHBoxLayout()
        layout.addLayout(prediction_memory_layout)

        self.perform_everything_on_device = QCheckBox('perform_everything_on_device', self)
        self.perform_everything_on_device.setToolTip(
            'Keep the full input and prediction arrays on the CUDA GPU. '
            'When disabled, these arrays stay in CPU RAM while the network still runs on the selected device.')
        self.perform_everything_on_device.toggled.connect(self.saveSettingsPerformEverythingOnDevice)
        prediction_memory_layout.addWidget(self.perform_everything_on_device)

        self.preprocessing_processes = QSpinBox(self)
        self.preprocessing_processes.setRange(0, 64)
        self.preprocessing_processes.setValue(1)
        self.preprocessing_processes.setPrefix('Preprocessing workers (npp): ')
        self.preprocessing_processes.setToolTip(
            'Number of background preprocessing workers. More workers use more RAM. '
            'Set both worker counts to 0 for sequential processing.')
        self.preprocessing_processes.valueChanged.connect(self.saveSettingsPreprocessingProcesses)
        prediction_memory_layout.addWidget(self.preprocessing_processes)

        self.export_processes = QSpinBox(self)
        self.export_processes.setRange(0, 64)
        self.export_processes.setValue(1)
        self.export_processes.setPrefix('Export workers (nps): ')
        self.export_processes.setToolTip(
            'Number of background segmentation export workers. More workers use more RAM. '
            'Set both worker counts to 0 for sequential processing.')
        self.export_processes.valueChanged.connect(self.saveSettingsExportProcesses)
        prediction_memory_layout.addWidget(self.export_processes)
        prediction_memory_layout.addWidget(QLabel('0/0: sequential processing', self))

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

        ## Input files and folders
        self.dropLabel = QLabel("Drop LIF files or folders here", self)
        self.dropLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont("Arial", 20, QFont.Weight.Bold)
        self.dropLabel.setFont(font)
        self.dropLabel.setStyleSheet("background-color: #f0f0f0; border: 2px dashed #888888; color: #888888;")
        self.dropLabel.setGeometry(0, 0, figure_width, 300)
        layout.addWidget(self.dropLabel, 1)

        self.pathLineEdit = QLineEdit()
        self.pathLineEdit.hide()

        ## Start Analysis Button
        self.startAnalysisButton = QPushButton('Start Analysis', self)
        self.startAnalysisButton.hide()
        self.startAnalysisButton.clicked.connect(self.startAnalysis)
        self.startAnalysisButton.setStyleSheet("background-color: 10c46a;")

        self.progressLabel = QLabel('Analysing...')
        self.progressLabel.hide()
        self.progressLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.fileCountLabel = QLabel('')
        self.frameTimingPanel = QWidget()
        timing_layout = QVBoxLayout(self.frameTimingPanel)
        timing_layout.setContentsMargins(0, 0, 0, 0)
        timing_controls = QHBoxLayout()
        self.override_frame_duration = QCheckBox('Override frame interval')
        timing_controls.addWidget(self.override_frame_duration)
        self.frame_duration = QDoubleSpinBox()
        self.frame_duration.setDecimals(6)
        self.frame_duration.setRange(0.000001, 1_000_000)
        self.frame_duration.setValue(1.0)
        self.frame_duration.setSuffix(' s')
        self.frame_duration.setEnabled(False)
        timing_controls.addWidget(self.frame_duration)
        timing_controls.addStretch()
        timing_layout.addLayout(timing_controls)
        self.frameTimingTable = QTableWidget(0, 10)
        self.frameTimingTable.setHorizontalHeaderLabels([
            'File', '# Channels', '# Timesteps', 'Automatic\ninterval (s)', 'Min Δt (s)', 'Max Δt (s)',
            'CycleTime (s)', 'Sequence\nmaster (s)', 'Automatic source', 'Used\ninterval (s)'])
        self.frameTimingTable.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.frameTimingTable.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.frameTimingTable.verticalHeader().hide()
        self.frameTimingTable.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.frameTimingTable.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.frameTimingTable.setMinimumHeight(180)
        timing_layout.addWidget(self.frameTimingTable)
        self.override_frame_duration.toggled.connect(self.updateFrameTimingPreview)
        self.frame_duration.valueChanged.connect(self.updateFrameTimingPreview)

        self.widgets_to_hide = [self.pathLineEdit, self.startAnalysisButton,
                                self.fileCountLabel, self.frameTimingPanel]

        self.setAcceptDrops(True)
        for field in self.findChildren(QLineEdit):
            field.setAcceptDrops(False)
        self.frameTimingTable.setAcceptDrops(False)
        self.frameTimingTable.viewport().setAcceptDrops(False)
        self.is_analysis_running = False

        self.files_to_analyse = []
        self.frame_timings = []
        self.duplicate_file_names = set()

    def printSettings(self):
        logging.debug("Settings:")
        for key in self.settings.allKeys():
            logging.debug("%s: %s", key, self.settings.value(key))

    def loadSettings(self):
        output_path = self.settings.value("outputPath", "")
        self.output_folder.setText(output_path)
        nnunet_path = self.settings.value("nnunetPath", "")
        self.nnunet_folder.setText(nnunet_path)
        fiji_path = self.settings.value("fijiPath", "")
        self.fiji_folder.setText(fiji_path)
        nnu_dataset = self.settings.value("nnuDatasetNumber", "")
        self.nnudataset.setText(nnu_dataset)
        self.disable_tta.setChecked(self.settings.value("disableTTA", False, type=bool))
        self.continue_prediction.setChecked(self.settings.value("continuePrediction", False, type=bool))
        step_size = self.settings.value("stepSize", "")
        if step_size == '':
            step_size = 0.5
        self.step_size.setValue(float(step_size))
        self.perform_everything_on_device.setChecked(
            self.settings.value('performEverythingOnDevice', False, type=bool))
        self.preprocessing_processes.setValue(self.settings.value('preprocessingProcesses', 1, type=int))
        self.export_processes.setValue(self.settings.value('exportProcesses', 1, type=int))

    def showFileDialogOutputPath(self):
        dir_path = QFileDialog.getExistingDirectory(self, 'Select directory')
        if dir_path:
            self.output_folder.setText(dir_path)
            self.settings.setValue("outputPath", dir_path)

    def updateResultsFolderButton(self):
        output_folder = self.output_folder.text()
        self.open_results_button.setEnabled(
            bool(output_folder) and os.path.isdir(join(output_folder, 'output_tracking')))

    def openResultsFolder(self):
        self.updateResultsFolderButton()
        if not self.open_results_button.isEnabled():
            QMessageBox.warning(self, 'Results folder unavailable', 'The results folder does not exist yet.')
            return
        results_folder = os.path.abspath(join(self.output_folder.text(), 'output_tracking'))
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(results_folder)):
            QMessageBox.warning(self, 'Cannot open results folder', f'Could not open:\n{results_folder}')

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

    def saveSettingsPerformEverythingOnDevice(self):
        self.settings.setValue('performEverythingOnDevice', self.perform_everything_on_device.isChecked())

    def saveSettingsPreprocessingProcesses(self):
        self.settings.setValue('preprocessingProcesses', self.preprocessing_processes.value())

    def saveSettingsExportProcesses(self):
        self.settings.setValue('exportProcesses', self.export_processes.value())

    def restoreDefaultPredictionSettings(self):
        self.disable_tta.setChecked(False)
        self.continue_prediction.setChecked(False)
        self.step_size.setValue(0.5)
        self.perform_everything_on_device.setChecked(False)
        self.preprocessing_processes.setValue(1)
        self.export_processes.setValue(1)

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
            self.layout().addWidget(self.progressLabel, 1)
        self.progressLabel.show()

    def hideProgressBar(self):
        self.progressLabel.hide()

    def download_with_progress(self, download_function, subdirectory, archive_name, title):
        repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if not os.path.exists(join(repo_path, '.git')):
            repo_path = self.settings.value('git_repo_path', '')
        if not repo_path or not os.path.exists(join(repo_path, '.git')):
            repo_path = os.path.expanduser('~/TNTAnalysis')
        directory = join(repo_path, 'downloads', subdirectory)
        archive_path = join(directory, archive_name)
        if os.path.isfile(archive_path):
            QMessageBox.information(
                self, 'Download skipped',
                f'This file already exists:\n{archive_path}\n\n'
                'The download will be skipped. The existing archive will be extracted.')

        dialog = DownloadProgressDialog(title, download_function, directory, self)
        dialog.download_thread.start()
        dialog.exec()
        dialog.download_thread.wait()
        result, error = dialog.download_thread.result, dialog.download_thread.error
        dialog.deleteLater()
        if error is not None:
            QMessageBox.warning(self, 'Download failed', error)
        return result

    def download_and_link_fiji(self):
        link_to_fiji = self.download_with_progress(download_fiji, 'fiji', 'fiji.zip', 'Downloading Fiji')
        if link_to_fiji:
            self.fiji_folder.setText(link_to_fiji)
            self.settings.setValue("fijiPath", link_to_fiji)

    def download_model_from_web(self):
        model_folder = self.download_with_progress(download_model, 'models', 'models.zip', 'Downloading Model')
        if model_folder:
            self.nnunet_folder.setText(model_folder)
            self.settings.setValue("nnunetPath", model_folder)
            self.nnudataset.setText("301")
            self.settings.setValue("nnuDatasetNumber", "301")

    def download_test_file(self):
        path_to_testfile = self.download_with_progress(
            download_testfiles, 'testfiles', 'test_data.zip', 'Downloading Test File')
        if path_to_testfile:
            self.pathLineEdit.setText(path_to_testfile)
            self.loadFiles(path_to_testfile, namefilter='2023-11-16_1_GFP-ctrl_TREK1-transfection.lif')



    def startAnalysis(self):
        override = self.override_frame_duration.isChecked()
        if (self.is_analysis_running or not self.files_to_analyse or self.duplicate_file_names
                or any(error and not (override and isinstance(error, FrameTimingError))
                       for timing, error in self.frame_timings)):
            return
        frame_duration = self.frame_duration.value() if override else None
        try:
            self.analysisThread = AnalysisThread(path_imgs_to_analyse=self.files_to_analyse,
                                                 path_to_nnunet_model=self.nnunet_folder.text(),
                                                 nnunet_dataset_number=self.nnudataset.text(),
                                                 path_to_output_folder=self.output_folder.text(),
                                                 fiji_path=self.fiji_folder.text(),
                                                 overlap=self.step_size.value(),
                                                 continue_prediction=self.continue_prediction.isChecked(),
                                                 disable_tta=self.disable_tta.isChecked(),
                                                 frame_duration=frame_duration,
                                                 perform_everything_on_device=self.perform_everything_on_device.isChecked(),
                                                 preprocessing_processes=self.preprocessing_processes.value(),
                                                 export_processes=self.export_processes.value())
        except Exception as error:
            logging.exception('Could not start analysis')
            QMessageBox.critical(self, 'Cannot start analysis', f'{type(error).__name__}: {error}')
            return

        self.analysisThread.finished.connect(self.analysisFinished)
        self.is_analysis_running = True
        self.hideWidgets()
        self.showProgressBar()
        self.analysisThread.start()

    def analysisFinished(self):
        self.is_analysis_running = False
        self.hideProgressBar()
        self.updateResultsFolderButton()
        if self.analysisThread.error is not None:
            for widget in self.widgets_to_hide:
                widget.show()
            self.pathLineEdit.hide()
            self.updateFrameTimingPreview()
            QMessageBox.critical(self, 'Analysis failed', self.analysisThread.error)
            return

        self.resetApplication()
        QMessageBox.information(self, 'TNT Analysis finished',
                                'TNT Analysis finished. Please proofread the generated .xml file using Trackmate/Fiji.')

    def closeEvent(self, event):
        if self.is_analysis_running:
            QMessageBox.information(self, 'Analysis running', 'Wait for the analysis to finish before closing.')
            event.ignore()
        else:
            super().closeEvent(event)

    def hideWidgets(self):
        logging.info('Hiding widgets')
        for widget in self.widgets_to_hide:
            widget.hide()

    def resetApplication(self):
        logging.info('Resetting application')
        for widget in self.widgets_to_hide:
            widget.show()
        # reset widgets
        self.files_to_analyse = []
        self.frame_timings = []
        self.duplicate_file_names = set()
        self.override_frame_duration.setChecked(False)
        self.frameTimingTable.setRowCount(0)
        self.frameTimingPanel.hide()
        self.fileCountLabel.setText('')
        self.fileCountLabel.hide()
        self.pathLineEdit.setText('')
        self.pathLineEdit.hide()
        self.startAnalysisButton.hide()
        self.dropLabel.show()

    def loadFiles(self, paths, namefilter=''):
        if self.is_analysis_running:
            return
        if isinstance(paths, (str, os.PathLike)):
            paths = [paths]
        files = []
        seen = set()
        for path in paths:
            for file_path in load_files_recursively(path):
                identity = os.path.normcase(os.path.realpath(file_path))
                if identity not in seen and namefilter in file_path:
                    seen.add(identity)
                    files.append(file_path)

        name_counts = Counter(os.path.splitext(os.path.basename(path))[0].casefold() for path in files)
        self.duplicate_file_names = {name for name, count in name_counts.items() if count > 1}
        self.pathLineEdit.setText('; '.join(os.fspath(path) for path in paths))

        self.files_to_analyse = files
        self.frame_timings = []
        self.override_frame_duration.setChecked(False)
        for path in files:
            timing = {'channel_count': None, 'timestep_count': None}
            try:
                lif_file = open_lif(path)
                timing.update(get_lif_series_metadata(lif_file))
                timing.update(get_frame_timing(lif_file))
                self.frame_timings.append((timing, None))
            except FrameTimingError as error:
                logging.warning('Could not read frame timing for %s: %s', path, error)
                timing.update(error.timing)
                self.frame_timings.append((timing, error))
            except Exception as error:
                logging.exception('Could not read frame timing for %s', path)
                self.frame_timings.append((timing, error))
        if self.frame_timings and not self.frame_timings[0][1]:
            self.frame_duration.setValue(self.frame_timings[0][0]['frame_interval_s'])
        self.updateFrameTimingPreview()

        if len(files) > 0:
            if not self.widgetIsAdded(self.fileCountLabel):
                self.layout().addWidget(self.fileCountLabel)
            if not self.widgetIsAdded(self.frameTimingPanel):
                self.layout().addWidget(self.frameTimingPanel, 1)
            self.fileCountLabel.show()
            self.frameTimingPanel.show()
            self.dropLabel.hide()
            if not self.widgetIsAdded(self.startAnalysisButton):
                self.layout().addWidget(self.startAnalysisButton)
            if not self.widgetIsAdded(self.pathLineEdit):
                self.layout().addWidget(self.pathLineEdit)
            self.startAnalysisButton.show()

        else:
            self.frameTimingPanel.hide()
            self.startAnalysisButton.hide()
            self.fileCountLabel.setText('No LIF files found.')
            if not self.widgetIsAdded(self.fileCountLabel):
                self.layout().addWidget(self.fileCountLabel)
            self.fileCountLabel.show()
            self.dropLabel.show()

    def updateFrameTimingPreview(self):
        override = self.override_frame_duration.isChecked()
        self.frame_duration.setEnabled(override)
        frame_duration = self.frame_duration.value() if override else None
        self.frameTimingTable.setRowCount(len(self.files_to_analyse))
        errors = 0
        timing_errors = 0
        multichannel_files = 0
        for row, (path, (timing, error)) in enumerate(zip(self.files_to_analyse, self.frame_timings)):
            duplicate_name = os.path.splitext(os.path.basename(path))[0].casefold() in self.duplicate_file_names
            values = [os.path.basename(path), timing['channel_count'], timing['timestep_count']]
            multichannel_files += (timing['channel_count'] or 0) > 1
            if isinstance(error, FrameTimingError):
                timing_errors += 1
            if error and not isinstance(error, FrameTimingError):
                values += [None, None, None, None, None, 'Unavailable', None]
                errors += 1
            else:
                effective_timing = apply_frame_duration_override(timing, frame_duration)
                source = {'timestamps': 'Timestamps', 'sequence_master': 'Sequence master'}.get(
                    timing['frame_interval_source'], 'Timing unavailable')
                values += [timing['frame_interval_s'], timing['delta_t_min_s'], timing['delta_t_max_s'],
                           timing['cycle_time_s'], timing['sequence_cycle_time_s'],
                           source, effective_timing['frame_interval_s']]
            for column, value in enumerate(values):
                text = '—' if value is None else (f'{value:.6f}' if isinstance(value, float) else str(value))
                item = QTableWidgetItem(text)
                tooltip = f'{path}\n{error}' if error else f'{path}\n{text}'
                if duplicate_name:
                    tooltip += '\nAnother input file has the same name. Use unique filenames to avoid overwriting results.'
                item.setToolTip(tooltip)
                if column not in (0, 8):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.frameTimingTable.setItem(row, column, item)
        status = f'Number of files found: {len(self.files_to_analyse)}'
        if multichannel_files:
            warning_color = '#f6c344' if self.palette().color(QPalette.ColorRole.Window).lightness() < 128 else '#8a4f00'
            status += (f' — <span style="color: {warning_color};">'
                       f'First channel only for {multichannel_files} multichannel file(s).</span>')
        if timing_errors:
            status += (f' — timing unavailable for {timing_errors} file(s); '
                       + ('using manual interval.' if override else 'set a manual interval to continue.'))
        if errors:
            status += f' — cannot read {errors} file(s); see row tooltips.'
        if self.duplicate_file_names:
            status += ' — duplicate filenames; use unique names before starting.'
        self.fileCountLabel.setText(status)
        self.fileCountLabel.setWordWrap(True)
        self.startAnalysisButton.setEnabled(bool(self.files_to_analyse) and errors == 0
                                            and (timing_errors == 0 or override) and not self.duplicate_file_names)

    def dragEnterEvent(self, e):
        if not self.is_analysis_running and any(
                url.isLocalFile() and (os.path.isdir(url.toLocalFile()) or
                                      (os.path.isfile(url.toLocalFile()) and url.toLocalFile().lower().endswith('.lif')))
                for url in e.mimeData().urls()):
            e.acceptProposedAction()
            self.dropLabel.setText("Drop LIF files or folders here")
            self.dropLabel.setStyleSheet("background-color: #bce0ce; border: 2px dashed #888888; color: #10c46a;")
        else:
            e.ignore()

    def dragLeaveEvent(self, e):
        self.dropLabel.setText("Drop LIF files or folders here")
        self.dropLabel.setStyleSheet("background-color: #f0f0f0; border: 2px dashed #888888; color: #888888;")

    def dropEvent(self, e):
        paths = [url.toLocalFile() for url in e.mimeData().urls() if url.isLocalFile()]
        if self.is_analysis_running or not paths:
            e.ignore()
            return
        self.loadFiles(paths)
        self.dropLabel.setStyleSheet("background-color: #f0f0f0; border: 2px dashed #888888; color: #888888;")
        e.acceptProposedAction()


def run_GUI():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    app = QApplication([])

    git_is_uptodate = check_if_git_uptodate()
    if git_is_uptodate is False:
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
