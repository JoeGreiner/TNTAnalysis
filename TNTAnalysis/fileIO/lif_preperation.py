import logging
import os.path
from os import makedirs
from os.path import basename, join, exists

import itk
import numpy as np
from readlif.reader import LifFile
from tifffile import tifffile

# Configure the root logger
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
# Create a logger
logger = logging.getLogger(__name__)

def load_part_of_lif_file(lif_obj, squeeze=True, swap_axes=True):
    if lif_obj.bit_depth[0] == 8:
        data_type = np.uint8
    elif lif_obj.bit_depth[0] == 12:
        data_type = np.uint16
    elif lif_obj.bit_depth[0] == 16:
        data_type = np.uint32
    else:
        data_type = np.float32
        print(f'bit_depth {lif_obj.bit_depth} not implemented')
        # return
    dim_x, dim_y, dim_z, dim_t, dim_c = [lif_obj.dims[i] for i in range(len(lif_obj.dims))]
    img = np.zeros(shape=(dim_x, dim_y, dim_z, dim_t, dim_c), dtype=data_type)
    for z in range(dim_z):
        for t in range(dim_t):
            for c in range(dim_c):
                img[:, :, z, t, c] = np.array(lif_obj.get_frame(z=z, t=t, c=c)).T

    is_timeseries = False
    if dim_t > 1:
        is_timeseries = True

    is_2d_xy_image = False
    if dim_x > 1:
        if dim_y > 1:
            if dim_z == 1:
                if dim_t == 1:
                    if dim_c == 1:
                        is_2d_xy_image = True

    if squeeze:
        img = np.squeeze(img)
    if swap_axes:
        img = np.swapaxes(img, 0, 1)

    series_name = lif_obj.name

    if 'PhaseX' in lif_obj.settings.keys():
        phase = float(lif_obj.settings['PhaseX'])
    else:
        phase = -1

    scanspeed = lif_obj.settings['ScanSpeed']
    zoom = lif_obj.settings['Zoom']

    if 'CycleTime' in lif_obj.settings.keys():
        cycle_time = lif_obj.settings['CycleTime']
    else:
        cycle_time = -1

    is_bidirectional = lif_obj.settings['ScanDirectionXName'] == 'Bidirectional'
    is_unidirectional = not is_bidirectional
    all_settings = lif_obj.settings

    frame_average = lif_obj.settings['FrameAverage']
    line_average = lif_obj.settings['LineAverage']
    frame_accumulation = lif_obj.settings['FrameAccumulation']
    line_accumulation = lif_obj.settings['Line_Accumulation']

    magnification = lif_obj.settings['Magnification']
    objective_name = lif_obj.settings['ObjectiveName']

    # scale is given in px/µm
    scale_x_in_m = (1 / lif_obj.scale[0] * 1e-6)
    scale_y_in_m = (1 / lif_obj.scale[1] * 1e-6)
    # scaleZ_in_M = (1/lif_obj.scale[2]*1e-6)
    if lif_obj.scale[3] is not None:
        scale_t_in_s = (1 / lif_obj.scale[3])
    else:
        scale_t_in_s = None

    print(f'Name: {series_name} PhaseX: {phase}')
    return {'image_data': img, 'PhaseX': phase, 'series_name': series_name,
            'isBidirectional': is_bidirectional,
            'isUnidirectional': is_unidirectional,
            'frameAverage': frame_average,
            'lineAverage': line_average,
            'frameAccumulation': frame_accumulation,
            'lineAccumulation': line_accumulation,
            'magnification': magnification,
            'objectiveName': objective_name,
            'dimX': dim_x,
            'dimY': dim_y,
            'dimZ': dim_z,
            'dimT': dim_t,
            'dimC': dim_c,
            "scaleX_in_M": scale_x_in_m,
            "scaleY_in_M": scale_y_in_m,
            "scaleT_in_s": scale_t_in_s,
            'is2DXYImage': is_2d_xy_image,
            'scanspeed': scanspeed,
            'zoom': zoom,
            'isTimeseries': is_timeseries,
            'cycleTime': cycle_time,
            'all_settings': all_settings}


def read_series_with_read_lif(path, series_index=0, return_only_image_data=True):
    opened_lif_file = LifFile(path)
    number_images_in_lif = len(opened_lif_file.image_list)

    assert series_index < number_images_in_lif, "requested series is larger than series present in lif file"
    lif_data = load_part_of_lif_file(lif_obj=opened_lif_file.get_image(series_index))

    if return_only_image_data:
        return lif_data['image_data']
    else:
        return lif_data


skip_if_file_exists = True


def prepare_single_lif_to_nii(path_to_file, path_to_output_folder, skip_if_file_exists=True):
    """
    Load up a lif, file, transpose it to [t, y, x] and write it in the nii.gz format.

    :param path_to_file:
    :param path_to_output_folder:
    :return:
    """
    name_of_file = basename(path_to_file)
    output_name = name_of_file.replace(".lif", "lif_0000.nii.gz")
    output_path = join(path_to_output_folder, output_name)
    if skip_if_file_exists:
        if os.path.exists(output_path):
            print(f'{output_path} already exists, skipping')
            return

    first_item_v2 = read_series_with_read_lif(path_to_file, series_index=0)
    logger.info(f"Shape before transposing: {first_item_v2.shape}")
    first_item_v2 = np.transpose(first_item_v2, (2, 1, 0))
    logger.info(f"Shape after transposing: {first_item_v2.shape}")
    itk.imwrite(itk.GetImageFromArray(first_item_v2), )
    logger.info(f"File {output_name} written successfully!")



def combine_lif_and_matching_prediction(lif_path, prediction_folder, output_folder):
    logging.info(f"Loading the lif file {lif_path}")
    lif_data = read_series_with_read_lif(lif_path, series_index=0, return_only_image_data=False)
    logger.info(f"Shape before transposing: {lif_data['image_data'].shape}")
    lif_data['image_data'] = np.transpose(lif_data['image_data'], (2, 1, 0))
    logger.info(f"Shape after transposing: {lif_data['image_data'].shape}")

    scale_x_in_m, scale_y_in_m, _ = [lif_data[key] for key in ['scaleX_in_M', 'scaleY_in_M', 'scaleT_in_s']]
    cycle_time_in_s = lif_data['cycleTime']
    scale_x_in_m = round(scale_x_in_m * 1000000, 3)
    scale_y_in_m = round(scale_y_in_m * 1000000, 3)
    assert scale_x_in_m == scale_y_in_m, f"The scales are different {scale_x_in_m} != {scale_y_in_m}"
    logging.info(f"Scale in x and y is {scale_x_in_m} um, cycle time is {cycle_time_in_s} s")

    image_path_no_ext = lif_path.replace(".lif", "")
    expected_prediction_path = join(prediction_folder, basename(image_path_no_ext) + "lif.nii.gz")
    assert exists(expected_prediction_path), f"Expected prediction file {expected_prediction_path} does not exist"
    prediction_data = itk.GetArrayFromImage(itk.imread(expected_prediction_path))
    assert prediction_data.shape == lif_data['image_data'].shape, f"Prediction shape {prediction_data.shape} does not match lif shape {lif_data['image_data'].shape}"
    if np.max(prediction_data) == 1:
        prediction_data = prediction_data * 255
    if prediction_data.dtype != np.uint8:
        prediction_data = np.clip(prediction_data, 0, 255)
        prediction_data = prediction_data.astype(np.uint8)

    timeseries_shape = lif_data['image_data'].shape
    combined_data = np.zeros((timeseries_shape[0], 2,  timeseries_shape[1], timeseries_shape[2]),
                             dtype=np.uint8)
    # swap x y to keep compatibility with imagej/drag and drop lif
    combined_data[:, 1, :, :] = np.swapaxes(lif_data['image_data'], 1, 2)
    combined_data[:, 0, :, :] = np.swapaxes(prediction_data, 1, 2)
    output_path_combined = join(output_folder, basename(lif_path).replace(".lif", "_combined.tif"))


    if not exists(output_folder):
        makedirs(output_folder)

    tifffile.imwrite(
        output_path_combined,
        combined_data,
        imagej=True,
        resolution=(1.0 / scale_x_in_m, 1.0 / scale_y_in_m),
        metadata={
            'unit': 'um',
            'finterval': cycle_time_in_s,
            'fps': 1 / float(cycle_time_in_s),
            'axes': 'TCYX',
            'mode': 'composite'
        },
    )

    return output_path_combined
