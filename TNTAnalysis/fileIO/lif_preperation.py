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


def get_frame_timing(lif_file, series_index=0, channel_index=0):
    """Read timing for a 2D time series without loading pixels; all returned times are in seconds."""
    def iter_images(element):
        children = element.findall('./Children/Element') or element.findall('./Element')
        for child in children:
            image = child.find('./Data/Image')
            if image is not None:
                yield image
            yield from iter_images(child)

    images = list(iter_images(lif_file.xml_root))
    if len(images) != len(lif_file.image_list) or not 0 <= series_index < len(images):
        raise ValueError('Cannot match the selected LIF series to its timing metadata.')
    image = images[series_index]
    series_name = lif_file.image_list[series_index]['name']
    dimensions = {int(node.get('DimID')): node for node in image.findall('./ImageDescription/Dimensions/*')}
    time_axis = dimensions.get(4)
    frame_count = int(time_axis.get('NumberOfElements')) if time_axis is not None else 1
    if frame_count < 2:
        raise ValueError(f'{series_name}: at least two time points are needed to determine a frame interval.')
    if any(int(node.get('NumberOfElements')) > 1 for axis, node in dimensions.items() if axis not in (1, 2, 4)):
        raise ValueError(f'{series_name}: timing is only supported for 2D time series, without Z stacks or tiles.')
    channels = image.findall('./ImageDescription/Channels/ChannelDescription')
    if not 0 <= channel_index < len(channels):
        raise ValueError(f'{series_name}: invalid channel index {channel_index}.')

    timing = {'delta_t_min_s': None, 'delta_t_max_s': None}
    for key, path in (
        ('cycle_time_s', './Attachment/ATLConfocalSettingDefinition'),
        ('sequence_cycle_time_s', './Attachment/LDM_Block_Sequential/LDM_Block_Sequential_Master/ATLConfocalSettingDefinition'),
    ):
        values = set()
        for settings in image.findall(path):
            try:
                value = float(settings.get('CycleTime', 'nan'))
            except ValueError:
                continue
            if np.isfinite(value) and value > 0:
                values.add(value)
        timing[key] = values.pop() if len(values) == 1 else None

    timestamp_lists = image.findall('./TimeStampList')
    if len(timestamp_lists) > 1:
        raise ValueError(f'{series_name}: multiple timestamp lists are ambiguous.')
    ticks = []
    if timestamp_lists:
        timestamps = timestamp_lists[0]
        try:
            if len(timestamps):
                if any(stamp.tag != 'TimeStamp' for stamp in timestamps):
                    raise ValueError('Unsupported timestamp format')
                ticks = [(int(stamp.attrib['HighInteger']) << 32) | (int(stamp.attrib['LowInteger']) & 0xffffffff)
                         for stamp in timestamps]
            else:
                ticks = [int(value, 16) for value in (timestamps.text or '').split()]
            declared_count = timestamps.get('NumberOfTimeStamps')
            if declared_count is not None and int(declared_count) != len(ticks):
                raise ValueError('Timestamp count does not match NumberOfTimeStamps')
            ticks = np.asarray(ticks, dtype=np.int64)
            if np.any(ticks < 0):
                raise ValueError('Negative timestamp')
        except (KeyError, ValueError, OverflowError) as error:
            raise ValueError(f'{series_name}: invalid LIF timestamps ({error}).') from error

    if len(ticks) == 0:
        if timing['sequence_cycle_time_s'] is None:
            raise ValueError(f'{series_name}: no timestamps or unambiguous positive sequence-master CycleTime available.')
        timing['frame_interval_s'] = timing['sequence_cycle_time_s']
        timing['frame_interval_source'] = 'sequence_master'
        logger.warning('%s: no timestamps; using sequence-master CycleTime %.6f s. Delta-T extrema are unknown.',
                       series_name, timing['frame_interval_s'])
        return timing

    if len(ticks) == frame_count:
        frame_ticks = ticks
    elif len(ticks) == frame_count * len(channels):
        try:
            offsets = [int(channel.attrib['BytesInc']) for channel in channels]
            time_stride = int(time_axis.attrib['BytesInc'])
        except (KeyError, ValueError) as error:
            raise ValueError(f'{series_name}: cannot determine timestamp order from BytesInc.') from error
        if min(offsets) != 0 or len(set(offsets)) != len(offsets) or time_stride <= 0:
            raise ValueError(f'{series_name}: ambiguous channel/time ordering.')
        channel_position = sorted(offsets).index(offsets[channel_index])
        # LIF plane order follows byte offsets; channels may be stored before or after time.
        if time_stride > max(offsets):
            frame_ticks = ticks.reshape(frame_count, len(channels))[:, channel_position]
        elif time_stride * frame_count <= sorted(offsets)[1]:
            frame_ticks = ticks.reshape(len(channels), frame_count)[channel_position]
        else:
            raise ValueError(f'{series_name}: unsupported channel/time ordering.')
    else:
        raise ValueError(f'{series_name}: {len(ticks)} timestamps cannot be assigned to '
                         f'{frame_count} time points and {len(channels)} channels.')

    # Subtract integer 100 ns ticks before converting to seconds to preserve precision.
    delta_t_s = np.diff(frame_ticks) / 10_000_000
    if np.any(delta_t_s <= 0):
        raise ValueError(f'{series_name}: timestamps must increase strictly within the selected channel.')
    timing.update(frame_interval_s=float(delta_t_s.mean()), frame_interval_source='timestamps',
                  delta_t_min_s=float(delta_t_s.min()), delta_t_max_s=float(delta_t_s.max()))
    logger.info('%s: timestamp interval mean %.6f s, min %.6f s, max %.6f s (channel %d).',
                series_name, timing['frame_interval_s'], timing['delta_t_min_s'], timing['delta_t_max_s'], channel_index)
    return timing


def apply_frame_duration_override(timing, frame_duration=None):
    """Keep the automatic timing for comparison when an absolute interval is supplied."""
    timing = dict(timing)
    timing['automatic_frame_interval_s'] = timing['frame_interval_s']
    timing['automatic_frame_interval_source'] = timing['frame_interval_source']
    if frame_duration is not None:
        if not np.isfinite(frame_duration) or frame_duration <= 0:
            raise ValueError('The frame interval must be a finite positive number of seconds.')
        timing['frame_interval_s'] = float(frame_duration)
        timing['frame_interval_source'] = 'manual_override'
    return timing


def load_part_of_lif_file(lif_obj, squeeze=True, swap_axes=True):
    if lif_obj.dims.m > 1:
        raise ValueError(f'{lif_obj.name}: mosaic images are not supported.')
    if lif_obj.channels > 1:
        logger.warning('%s: using only the first of %d channels.', lif_obj.name, lif_obj.channels)
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
    dim_x, dim_y, dim_z, dim_t = lif_obj.dims[:4]
    dim_c = 1
    img = np.zeros(shape=(dim_x, dim_y, dim_z, dim_t, dim_c), dtype=data_type)
    for z in range(dim_z):
        for t in range(dim_t):
            img[:, :, z, t, 0] = np.array(lif_obj.get_frame(z=z, t=t, c=0)).T

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


def read_series_with_read_lif(path, series_index=0, return_only_image_data=True, include_frame_timing=False):
    opened_lif_file = LifFile(path)
    number_images_in_lif = len(opened_lif_file.image_list)

    assert series_index < number_images_in_lif, "requested series is larger than series present in lif file"
    frame_timing = get_frame_timing(opened_lif_file, series_index) if include_frame_timing else None
    lif_data = load_part_of_lif_file(lif_obj=opened_lif_file.get_image(series_index))

    if return_only_image_data:
        return lif_data['image_data']
    else:
        if include_frame_timing:
            lif_data['frame_timing'] = frame_timing
        return lif_data


def prepare_single_lif_to_nii(path_to_file, path_to_output_folder):
    """
    Load up a lif, file, transpose it to [t, y, x] and write it in the nii.gz format.

    :param path_to_file:
    :param path_to_output_folder:
    :return:
    """
    name_of_file = basename(path_to_file)
    output_name = os.path.splitext(name_of_file)[0] + 'lif_0000.nii.gz'
    output_path = join(path_to_output_folder, output_name)
    first_item_v2 = read_series_with_read_lif(path_to_file, series_index=0)
    logger.info(f"Shape before transposing: {first_item_v2.shape}")
    first_item_v2 = np.transpose(first_item_v2, (2, 1, 0))
    logger.info(f"Shape after transposing: {first_item_v2.shape}")
    itk.imwrite(itk.GetImageFromArray(first_item_v2), output_path)
    logger.info(f"File {output_name} written successfully!")



def combine_lif_and_matching_prediction(lif_path, prediction_folder, output_folder, frame_duration=None):
    logging.info(f"Loading the lif file {lif_path}")
    lif_data = read_series_with_read_lif(lif_path, series_index=0, return_only_image_data=False, include_frame_timing=True)
    logger.info(f"Shape before transposing: {lif_data['image_data'].shape}")
    lif_data['image_data'] = np.transpose(lif_data['image_data'], (2, 1, 0))
    logger.info(f"Shape after transposing: {lif_data['image_data'].shape}")

    scale_x_in_m, scale_y_in_m = [lif_data[key] for key in ['scaleX_in_M', 'scaleY_in_M']]
    frame_timing = apply_frame_duration_override(lif_data['frame_timing'], frame_duration)
    frame_interval_s = frame_timing['frame_interval_s']
    scale_x_in_m = round(scale_x_in_m * 1000000, 3)
    scale_y_in_m = round(scale_y_in_m * 1000000, 3)
    assert scale_x_in_m == scale_y_in_m, f"The scales are different {scale_x_in_m} != {scale_y_in_m}"
    logging.info(f"Scale in x and y is {scale_x_in_m} um, frame interval is {frame_interval_s} s")

    image_path_no_ext = os.path.splitext(lif_path)[0]
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
    output_path_combined = join(output_folder, os.path.splitext(basename(lif_path))[0] + '_combined.tif')


    if not exists(output_folder):
        makedirs(output_folder)

    tifffile.imwrite(
        output_path_combined,
        combined_data,
        imagej=True,
        resolution=(1.0 / scale_x_in_m, 1.0 / scale_y_in_m),
        metadata={
            'unit': 'um',
            'finterval': frame_interval_s,
            'fps': 1 / frame_interval_s,
            'axes': 'TCYX',
            'mode': 'composite',
            **{key: value for key, value in frame_timing.items() if value is not None},
        },
    )

    return output_path_combined
