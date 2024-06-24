from skimage import io as skio
import numpy as np
import os
import pandas as pd

from TNTAnalysis.fileIO.git import check_if_git_uptodate
from TNTAnalysis.fileIO.xlsx import fix_df_lengths
import logging


from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLineEdit, QPushButton, QApplication, QLabel, QTextEdit,
                             QHBoxLayout,
                             QMessageBox)


def post_process_tracking_data(path_to_label_mask, path_to_xlsx):
    """
    Post-process tracking data from a xlsx file and a label mask.
    The label mask is used to check if the track coordinates fall within the label mask.
    :param path_to_label_mask: Path to a label mask (tif file)
    :param path_to_xlsx: Path to a xlsx file with tracking data
    :return: None

    write xlsx file with the data that falls within the label mask and another one with the data that falls outside
    write a tiffile with pixel=2 where the data falls within the label mask and pixel=1 where it falls outside
    paths are the same as the input paths but with suffixes (e.g. _inside_label_mask, _outside_label_mask, ...)
    also write trackmate-readable xml files where the tracks appear as 'filtered tracks'
    as long as track filtering is not repeated, the xml files can be used to visualize the tracks in trackmate

    """
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # path_to_label_mask = '/Users/greiner/label_example.tif'
    # path_to_xlsx = '/Users/greiner/tmp/output_tracking/2023-10-20_1_GFP-ctrl_TREK1-transfection.xlsx'

    assert path_to_label_mask is not None, "Path to label mask is None"
    assert os.path.exists(path_to_label_mask), f"Path to label mask {path_to_label_mask} does not exist"
    assert path_to_xlsx is not None, "Path to xlsx is None"
    assert os.path.exists(path_to_xlsx), f"Path to xlsx {path_to_xlsx} does not exist"

    path_to_xml = path_to_xlsx.replace('.xlsx', '.xml')
    assert os.path.exists(path_to_xml), f"Path to xml {path_to_xml} does not exist"

    logging.info(f"Reading label mask from {path_to_label_mask}")
    label_mask = skio.imread(path_to_label_mask)

    if len(label_mask.shape) == 3:
        logging.info("Label mask has shape (t, y, x), reducing to (y, x)")
        label_mask = np.max(label_mask, axis=0)

    df = pd.read_excel(path_to_xlsx, sheet_name='tracks')
    logging.info(f"Read {len(df)} rows from {path_to_xlsx}")

    # check if coordinates fall within the label mask > 0
    logging.info("Checking if coordinates fall within the label mask")

    def check_if_coordinate_falls_within_label_mask(row):
        x = row['TRACK_X_LOCATION_INDEX']
        y = row['TRACK_Y_LOCATION_INDEX']
        dimY, dimX = label_mask.shape
        if x < 0 or x >= dimX or y < 0 or y >= dimY:
            return False
        return label_mask[int(y), int(x)] > 0

    df['FALLS_WITHIN_LABEL_MASK'] = df.apply(check_if_coordinate_falls_within_label_mask, axis=1)

    df_falls_within_label_mask = df[df['FALLS_WITHIN_LABEL_MASK']]
    df_falls_outside_label_mask = df[~df['FALLS_WITHIN_LABEL_MASK']]

    track_ids_falls_within_label_mask = df_falls_within_label_mask['TRACK_ID'].unique()
    track_ids_falls_outside_label_mask = df_falls_outside_label_mask['TRACK_ID'].unique()

    logging.info(
        f"Found {len(df_falls_within_label_mask)} tracks that fall within the label mask (percentage: {len(df_falls_within_label_mask) / len(df) * 100:.2f}%)")
    logging.info(
        f"Found {len(df_falls_outside_label_mask)} tracks that fall outside the label mask (percentage: {len(df_falls_outside_label_mask) / len(df) * 100:.2f}%)")

    outpath_inside = path_to_xlsx.replace('.xlsx', '_inside_label_mask.xlsx')
    outpath_outside = path_to_xlsx.replace('.xlsx', '_outside_label_mask.xlsx')

    logging.info(
        f"Writing {len(df_falls_within_label_mask)} tracks that fall within the label mask to {outpath_inside}")
    writer = pd.ExcelWriter(outpath_inside)
    df_falls_within_label_mask.to_excel(writer, sheet_name='tracks', index=False, freeze_panes=(1, 1))
    fix_df_lengths(df_falls_within_label_mask, writer, 'tracks')
    writer.close()

    logging.info(
        f"Writing {len(df_falls_outside_label_mask)} tracks that fall outside the label mask to {outpath_outside}")
    writer = pd.ExcelWriter(outpath_outside)
    df_falls_outside_label_mask.to_excel(writer, sheet_name='tracks', index=False, freeze_panes=(1, 1))
    fix_df_lengths(df_falls_outside_label_mask, writer, 'tracks')
    writer.close()

    logging.info("Writing xml files with tracks that fall within and outside the label mask")
    # create two xml files, one for the tracks that fall within the label mask and one for the tracks that fall outside
    # read the xml until '<FilteredTracks>'
    xml_before_filtered_tracks = []
    xml_after_filtered_tracks_end = []
    is_before_filtered_tracks = True
    is_after_filtered_tracks = False
    with open(path_to_xml, 'r') as f:
        for line in f:
            if '    </FilteredTracks>' in line:
                is_after_filtered_tracks = True
            if '    <FilteredTracks>' in line:
                is_before_filtered_tracks = False

            if is_before_filtered_tracks:
                xml_before_filtered_tracks.append(line)
            if is_after_filtered_tracks:
                xml_after_filtered_tracks_end.append(line)

    # add '      <TrackID TRACK_ID="0" /> for every trackid that falls within the label mask
    xml_inside = xml_before_filtered_tracks.copy()
    xml_outside = xml_before_filtered_tracks.copy()
    xml_inside.append('    <FilteredTracks>\n')
    for track_id in track_ids_falls_within_label_mask:
        xml_inside.append(f'      <TrackID TRACK_ID="{track_id}" />\n')

    xml_outside.append('    <FilteredTracks>\n')
    for track_id in track_ids_falls_outside_label_mask:
        xml_outside.append(f'      <TrackID TRACK_ID="{track_id}" />\n')

    xml_inside.extend(xml_after_filtered_tracks_end)
    xml_outside.extend(xml_after_filtered_tracks_end)

    outpath_inside_xml = path_to_xml.replace('.xml', '_inside_label_mask.xml')
    outpath_outside_xml = path_to_xml.replace('.xml', '_outside_label_mask.xml')

    logging.info(f"Writing xml file with tracks that fall within the label mask to {outpath_inside_xml}")
    with open(outpath_inside_xml, 'w') as f:
        f.writelines(xml_inside)

    logging.info(f"Writing xml file with tracks that fall outside the label mask to {outpath_outside_xml}")
    with open(outpath_outside_xml, 'w') as f:
        f.writelines(xml_outside)

    logging.info("Done")


class GUI_TNT_Analysis(QWidget):
    def __init__(self):
        super(GUI_TNT_Analysis, self).__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Postprocessing: Automatic TNT Analysis')
        figure_width = 800
        figure_height = 400
        self.setGeometry(300, 300, figure_width, figure_height)

        layout = QVBoxLayout()
        self.setLayout(layout)

        ## Drop Folder for TIFs Selection
        self.dropLabelField = QLabel("Drop one label mask (.tif) and tracking results (.xlsx) here", self)
        self.dropLabelField.setAlignment(Qt.AlignCenter)
        font = QFont("Arial", 20, QFont.Bold)
        self.dropLabelField.setFont(font)
        self.dropLabelField.setStyleSheet("background-color: #f0f0f0; border: 2px dashed #888888; color: #888888;")
        self.dropLabelField.setGeometry(0, 0, figure_width, 300)
        layout.addWidget(self.dropLabelField)

        self.imageInputLayout = QHBoxLayout()
        layout.addLayout(self.imageInputLayout)

        self.imageInputLabel = QLabel("Image Path: ", self)
        self.imageInputLayout.addWidget(self.imageInputLabel)
        self.lineWidgetImage = QLineEdit()
        self.imageInputLayout.addWidget(self.lineWidgetImage)

        self.xlsxInputLayout = QHBoxLayout()
        layout.addLayout(self.xlsxInputLayout)

        self.xlsxInputLabel = QLabel("XLSX Path: ", self)
        self.xlsxInputLayout.addWidget(self.xlsxInputLabel)
        self.lineWidgetXLSX = QLineEdit()
        self.xlsxInputLayout.addWidget(self.lineWidgetXLSX)

        self.xml_added = False
        self.img_added = False

        ## Start Analysis Button
        self.startVisButton = QPushButton('Start Analysis', self)
        layout.addWidget(self.startVisButton)
        self.startVisButton.hide()
        self.startVisButton.clicked.connect(self.startProcessing)
        # make it green
        self.startVisButton.setStyleSheet("background-color: #10c46a; color: white; font-size: 20px;")

        self.setAcceptDrops(True)

    def widgetIsAdded(self, object):
        # Check if the button is in the layout
        index = self.layout().indexOf(object)
        return index != -1

    def startProcessing(self):
        post_process_tracking_data(path_to_label_mask=self.lineWidgetImage.text(),
                                   path_to_xlsx=self.lineWidgetXLSX.text())

        msg = QMessageBox()
        msg.setWindowTitle("Masking finished")
        msg.setText("Masking done. Please proofread the generated .xml file using Trackmate/Fiji.")
        msg.exec()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.accept()
            print("Drag Enter Image")
            self.dropLabelField.setStyleSheet("background-color: #bce0ce; border: 2px dashed #888888; color: #10c46a;")
        else:
            e.ignore()

    def dragLeaveEvent(self, e):
        print("Drag Leave")
        self.dropLabelField.setStyleSheet("background-color: #f0f0f0; border: 2px dashed #888888; color: #888888;")

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            if url.isLocalFile():
                path = str(url.toLocalFile())
                # if ending is xml
                if path.endswith('.xlsx'):
                    self.lineWidgetXLSX.setText(path)
                    self.xml_added = True
                else:
                    self.lineWidgetImage.setText(path)
                    self.img_added = True

                self.dropLabelField.setStyleSheet(
                    "background-color: #f0f0f0; border: 2px dashed #888888; color: #888888;")

                if self.xml_added and self.img_added:
                    self.startVisButton.show()


def run_GUI():
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
