import logging
import os

from bioio import BioImage
from bioio_lif import Reader
import numpy as np

logger = logging.getLogger(__name__)


def open_lif(path):
    return BioImage(os.fspath(path), reader=Reader, reconstruct_mosaic=False)


def get_lif_series_xml(lif_image, series_index=0):
    def iter_images(element):
        children = element.findall('./Children/Element') or element.findall('./Element')
        for child in children:
            image = child.find('./Data/Image')
            if image is not None:
                yield image
            yield from iter_images(child)

    images = list(iter_images(lif_image.metadata))
    if len(images) != len(lif_image.scenes) or not 0 <= series_index < len(images):
        raise ValueError('Cannot match the selected LIF series to its metadata.')
    return images[series_index]


def get_lif_series_metadata(lif_image, series_index=0):
    image = get_lif_series_xml(lif_image, series_index)
    dimensions = {int(node.get('DimID')): node for node in image.findall('./ImageDescription/Dimensions/*')}
    if any(int(node.get('NumberOfElements')) > 1 for axis, node in dimensions.items() if axis not in (1, 2, 4)):
        raise ValueError(f'{lif_image.scenes[series_index]}: only 2D images and time series are supported, '
                         'without Z stacks or tiles.')
    time_axis = dimensions.get(4)
    return {
        'scene_name': lif_image.scenes[series_index],
        'channel_count': len(image.findall('./ImageDescription/Channels/ChannelDescription')),
        'timestep_count': int(time_axis.get('NumberOfElements')) if time_axis is not None else 1,
    }


class FrameTimingError(ValueError):
    def __init__(self, message, timing):
        super().__init__(message)
        self.timing = dict(timing, frame_interval_s=None, frame_interval_source=None)


def get_frame_timing(lif_file, series_index=0, channel_index=0):
    """Read timing for a 2D time series without loading pixels; all returned times are in seconds."""
    image = get_lif_series_xml(lif_file, series_index)
    series_name = lif_file.scenes[series_index]
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
        raise FrameTimingError(f'{series_name}: multiple timestamp lists are ambiguous.', timing)
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
            raise FrameTimingError(f'{series_name}: invalid LIF timestamps ({error}).', timing) from error

    if len(ticks) == 0:
        if timing['sequence_cycle_time_s'] is None:
            raise FrameTimingError(
                f'{series_name}: no timestamps or unambiguous positive sequence-master CycleTime available.', timing)
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
            raise FrameTimingError(f'{series_name}: cannot determine timestamp order from BytesInc.', timing) from error
        if min(offsets) != 0 or len(set(offsets)) != len(offsets) or time_stride <= 0:
            raise FrameTimingError(f'{series_name}: ambiguous channel/time ordering.', timing)
        channel_position = sorted(offsets).index(offsets[channel_index])
        # LIF plane order follows byte offsets; channels may be stored before or after time.
        if time_stride > max(offsets):
            frame_ticks = ticks.reshape(frame_count, len(channels))[:, channel_position]
        elif time_stride * frame_count <= sorted(offsets)[1]:
            frame_ticks = ticks.reshape(len(channels), frame_count)[channel_position]
        else:
            raise FrameTimingError(f'{series_name}: unsupported channel/time ordering.', timing)
    else:
        raise FrameTimingError(f'{series_name}: {len(ticks)} timestamps cannot be assigned to '
                               f'{frame_count} time points and {len(channels)} channels.', timing)

    # Subtract integer 100 ns ticks before converting to seconds to preserve precision.
    delta_t_s = np.diff(frame_ticks) / 10_000_000
    if np.any(delta_t_s <= 0):
        raise FrameTimingError(f'{series_name}: timestamps must increase strictly within the selected channel.', timing)
    timing.update(frame_interval_s=float(delta_t_s.mean()), frame_interval_source='timestamps',
                  delta_t_min_s=float(delta_t_s.min()), delta_t_max_s=float(delta_t_s.max()))
    logger.info('%s: timestamp interval mean %.6f s, min %.6f s, max %.6f s (channel %d).',
                series_name, timing['frame_interval_s'], timing['delta_t_min_s'], timing['delta_t_max_s'], channel_index)
    return timing


def apply_frame_duration_override(timing, frame_duration=None):
    """Keep the automatic timing for comparison when an absolute interval is supplied."""
    timing = dict(timing)
    timing['automatic_frame_interval_s'] = timing.get('frame_interval_s')
    timing['automatic_frame_interval_source'] = timing.get('frame_interval_source')
    if frame_duration is not None:
        if not np.isfinite(frame_duration) or frame_duration <= 0:
            raise ValueError('The frame interval must be a finite positive number of seconds.')
        timing['frame_interval_s'] = float(frame_duration)
        timing['frame_interval_source'] = 'manual_override'
    return timing


def read_lif_series(path, series_index=0, include_frame_timing=False, frame_duration=None):
    """Read the first channel as TYX, preserving singleton time and spatial axes."""
    image = open_lif(path)
    metadata = get_lif_series_metadata(image, series_index)
    image.set_scene(series_index)
    if metadata['channel_count'] > 1:
        logger.warning('%s: using only the first of %d channels.', metadata['scene_name'], metadata['channel_count'])
    if include_frame_timing:
        try:
            timing = get_frame_timing(image, series_index)
        except FrameTimingError as error:
            if frame_duration is None:
                raise
            logger.warning('Using manual frame interval: %s', error)
            timing = error.timing
        metadata['frame_timing'] = apply_frame_duration_override(timing, frame_duration)
    pixel_sizes = image.physical_pixel_sizes
    if any(value is None or not np.isfinite(value) or value <= 0 for value in (pixel_sizes.Y, pixel_sizes.X)):
        raise ValueError(f"{metadata['scene_name']}: missing or invalid spatial pixel sizes.")
    metadata['pixel_size_yx_um'] = (pixel_sizes.Y, pixel_sizes.X)
    metadata['image_data'] = image.get_image_dask_data('TYX', C=0, Z=0).compute()
    return metadata
