import logging
import os.path
from os import makedirs
from os.path import basename, join, exists

import itk
import numpy as np
from tifffile import tifffile

from TNTAnalysis.fileIO.lif import read_lif_series

logger = logging.getLogger(__name__)


def prepare_single_lif_to_nii(path_to_file, path_to_output_folder):
    """
    Read a LIF series and write the existing model input convention [t, x, y] as nii.gz.

    :param path_to_file:
    :param path_to_output_folder:
    :return:
    """
    name_of_file = basename(path_to_file)
    output_name = os.path.splitext(name_of_file)[0] + 'lif_0000.nii.gz'
    output_path = join(path_to_output_folder, output_name)
    image_tyx = read_lif_series(path_to_file)['image_data']
    # Existing nnUNet models were trained with TXY input; retain that convention.
    image_txy = np.ascontiguousarray(image_tyx.swapaxes(1, 2))
    itk.imwrite(itk.GetImageFromArray(image_txy), output_path)
    logger.info(f"File {output_name} written successfully!")



def combine_lif_and_matching_prediction(lif_path, prediction_folder, output_folder, frame_duration=None):
    logging.info(f"Loading the lif file {lif_path}")
    lif_data = read_lif_series(lif_path, include_frame_timing=True, frame_duration=frame_duration)
    image_tyx = lif_data['image_data']
    scale_y_in_um, scale_x_in_um = lif_data['pixel_size_yx_um']
    frame_timing = lif_data['frame_timing']
    frame_interval_s = frame_timing['frame_interval_s']
    scale_x_in_um = round(scale_x_in_um, 3)
    scale_y_in_um = round(scale_y_in_um, 3)
    assert scale_x_in_um == scale_y_in_um, f"The scales are different {scale_x_in_um} != {scale_y_in_um}"
    logging.info(f"Scale in x and y is {scale_x_in_um} um, frame interval is {frame_interval_s} s")

    image_path_no_ext = os.path.splitext(lif_path)[0]
    expected_prediction_path = join(prediction_folder, basename(image_path_no_ext) + "lif.nii.gz")
    assert exists(expected_prediction_path), f"Expected prediction file {expected_prediction_path} does not exist"
    prediction_data = itk.GetArrayFromImage(itk.imread(expected_prediction_path))
    expected_shape = (image_tyx.shape[0], image_tyx.shape[2], image_tyx.shape[1])
    if prediction_data.shape != expected_shape:
        raise ValueError(f'Prediction shape {prediction_data.shape} does not match LIF model-input shape {expected_shape}.')
    if np.max(prediction_data) == 1:
        prediction_data = prediction_data * 255
    if prediction_data.dtype != np.uint8:
        prediction_data = np.clip(prediction_data, 0, 255)
        prediction_data = prediction_data.astype(np.uint8)

    # swap x y to keep compatibility with imagej/drag and drop lif
    combined_data = np.stack((np.swapaxes(prediction_data, 1, 2),
                              image_tyx), axis=1)
    output_path_combined = join(output_folder, os.path.splitext(basename(lif_path))[0] + '_combined.tif')


    if not exists(output_folder):
        makedirs(output_folder)

    gray_lut = np.tile(np.arange(256, dtype=np.uint8), (3, 1))
    red_lut = gray_lut.copy()
    red_lut[1:] = 0
    tifffile.imwrite(
        output_path_combined,
        combined_data,
        imagej=True,
        resolution=(1.0 / scale_x_in_um, 1.0 / scale_y_in_um),
        metadata={
            'unit': 'um',
            'finterval': frame_interval_s,
            'fps': 1 / frame_interval_s,
            'axes': 'TCYX',
            'mode': 'composite',
            'LUTs': [red_lut, gray_lut],
            **{key: value for key, value in frame_timing.items() if value is not None},
        },
    )

    return output_path_combined
