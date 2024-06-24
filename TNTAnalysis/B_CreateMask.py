from os.path import join
import napari
import numpy as np
from magicgui import magicgui
from napari.layers import Image, Labels
import skimage

from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QSettings


def run_GUI():
    try:
        viewer = napari.Viewer()
    except Exception as e:
        print(f'Error: {e}')
        print('If it is a Qt error, try changing the import settings to match the installed Qt version.')
        return

    # load settings
    settings = QSettings("JoeGreiner", "TNT_Analysis_GUI")
    save_mask_layer_directory = settings.value('napari_create_mask_output_directory')
    if isinstance(save_mask_layer_directory, type(None)):
        save_mask_layer_directory = ''



    @magicgui(auto_call=False)
    def add_shape_layer(layer: Image):
        layer_shape = layer.data.shape
        layer_scale = layer.scale
        if len(layer_shape) == 3:
            # make 2D mask layer
            layer_shape = layer_shape[1:]
            layer_scale = layer_scale[1:]
        elif len(layer_shape) == 2:
            pass
        else:
            raise ValueError('Layer shape is not 2D or 3D')

        dummy_data = np.zeros(shape=layer_shape, dtype=np.uint8)
        mask_layer_to_save = viewer.add_labels(dummy_data, name=layer.name + '_mask', scale=layer_scale)
        mask_layer_to_save.brush_size = 150

    # rotate layer by 90 degrees in anticlockwise direction
    @magicgui(auto_call=False)
    def rotate_layer(layer: Image):
        layer.data = np.rot90(layer.data, k=1)

    # add gausian blurr option, with slider from 0 to 5
    @magicgui(auto_call=False,
              sigma={'widget_type': 'IntSlider', 'min': 0, 'max': 10})
    def add_gaussian_blur(layer: Image, sigma: float = 5) -> Image:
        # if 3D return too, its probably the reflection image you don't want to blur
        if len(layer.data.shape) == 3:
            print('3D image, not blurring!')
            return Image(layer.data, scale=layer.scale, name=layer.name + '_blurred')
        # global original_img
        # if isinstance(original_img, type(None)):
        #     original_img = layer.data.copy()
        if sigma > 0:
            # layer.data = skimage.filters.gaussian(original_img, sigma=sigma)
            blurred_img = skimage.filters.gaussian(layer.data, sigma=sigma)
            # if max < 1, multiply by 255, so that the int slider in napari for contrast works
            # if np.max(layer.data) < 1:
            #     layer.data = layer.data * 255
            # else:
            #     layer.data = original_img.copy()
            min_contrast = np.min(blurred_img) * 30
            max_contrast = np.max(blurred_img)
            # layer.contrast_limits = [min_contrast, max_contrast]

            return Image(blurred_img, scale=layer.scale, name=layer.name + '_blurred',
                         contrast_limits=[min_contrast, max_contrast], blending='additive', colormap='green')
        else:
            return Image(layer.data, scale=layer.scale, name=layer.name + '_blurred')

    # save mask layer with the same name as the image
    @magicgui(auto_call=False,
              directory={'widget_type': 'FileEdit', 'mode': 'd'}
              )
    def save_mask_layer(layer: Labels, directory=save_mask_layer_directory):
        series_name = layer.name
        series_name = series_name.replace(' :: ', '_')
        dock_widgets = viewer.window._dock_widgets
        dock_widget_keys = dock_widgets.keys()
        dock_widgets_with_lif_in_name = [key for key in dock_widget_keys if '.lif' in key]
        if len(dock_widgets_with_lif_in_name) == 1:
            lif_name = dock_widgets_with_lif_in_name[0].replace(' :: Scenes', '').replace('.lif', '_lif_')
        elif len(dock_widgets_with_lif_in_name) == 0:
            lif_name = ""
        else:
            lif_name = dock_widgets_with_lif_in_name[-1].replace(' :: Scenes', '').replace('.lif', '_lif_')

        if isinstance(layer, type(None)):
            print('No mask layer to save')
            return

        # save directory to qt settings
        settings = QSettings("JoeGreiner", "TNT_Analysis_GUI")
        settings.setValue('napari_create_mask_output_directory', directory)

        output_path = join(directory, f'{lif_name}{series_name}.tif')
        layer.save(join(directory, output_path))

        msg = QMessageBox()
        msg.setWindowTitle("Save Mask Layer")
        msg.setText(f"Saved {series_name} to {output_path}")
        msg.exec_()
        print(f'Saved "{series_name}" to {output_path}')
        # set status
        viewer.status = f'Saved "{series_name}" to {output_path}'

    viewer.window.add_dock_widget(add_shape_layer, name='0: Add Labels', add_vertical_stretch=True)
    viewer.window.add_dock_widget(rotate_layer, name='1: Rotate Layer', add_vertical_stretch=True)
    viewer.window.add_dock_widget(add_gaussian_blur, name='2: Gaussian Blur', add_vertical_stretch=True)
    viewer.window.add_dock_widget(save_mask_layer, name='3: Save Mask Layer', add_vertical_stretch=True)
    napari.run()

if __name__ == '__main__':
    run_GUI()