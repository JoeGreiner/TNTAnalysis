from pathlib import Path
import re
import napari
import numpy as np
from magicgui import magicgui
from napari.layers import Image, Labels
import skimage

from PyQt6.QtWidgets import QMessageBox, QApplication
from PyQt6.QtCore import QSettings

from TNTAnalysis.fileIO.git import check_if_git_uptodate


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
        # The reader may retain singleton T/C/Z axes; the mask is always spatial YX.
        layer_shape = layer.data.shape[-3:-1] if layer.rgb else layer.data.shape[-2:]
        layer_scale = layer.scale[-2:]
        dummy_data = np.zeros(shape=layer_shape, dtype=np.uint8)
        mask_layer_to_save = viewer.add_labels(dummy_data, name=layer.name + '_mask', scale=layer_scale)
        mask_layer_to_save.brush_size = 150

    # rotate layer by 90 degrees in anticlockwise direction
    @magicgui(auto_call=False)
    def rotate_layer(layer: Image):
        layer.data = np.rot90(layer.data, k=1, axes=(-3, -2) if layer.rgb else (-2, -1))
        layer.scale = (*layer.scale[:-2], layer.scale[-1], layer.scale[-2])

    # add gausian blurr option, with slider from 0 to 5
    @magicgui(auto_call=False,
              sigma={'widget_type': 'IntSlider', 'min': 0, 'max': 10})
    def add_gaussian_blur(layer: Image, sigma: float = 5) -> Image:
        # if 3D return too, its probably the reflection image you don't want to blur
        if layer.data.ndim != 2:
            print('Select a 2D image for Gaussian blur.')
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
            min_contrast = float(np.min(blurred_img)) * 30
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
        if layer is None:
            print('No mask layer to save')
            return
        series_name = re.sub(r'[\\/:*?"<>|]', '_', layer.name.replace(' :: ', '_'))

        # save directory to qt settings
        settings = QSettings("JoeGreiner", "TNT_Analysis_GUI")
        settings.setValue('napari_create_mask_output_directory', directory)

        output_path = Path(directory) / f'{series_name}.tif'
        layer.save(str(output_path))

        msg = QMessageBox()
        msg.setWindowTitle("Save Mask Layer")
        msg.setText(f"Saved {series_name} to {output_path}")
        msg.exec()
        print(f'Saved "{series_name}" to {output_path}')
        # set status
        viewer.status = f'Saved "{series_name}" to {output_path}'

    viewer.window.add_dock_widget(add_shape_layer, name='0: Add Labels')
    viewer.window.add_dock_widget(rotate_layer, name='1: Rotate Layer')
    viewer.window.add_dock_widget(add_gaussian_blur, name='2: Gaussian Blur')
    viewer.window.add_dock_widget(save_mask_layer, name='3: Save Mask Layer')
    napari.run()

if __name__ == '__main__':

    git_is_uptodate = check_if_git_uptodate()
    if git_is_uptodate is False:
        print("The local git repository is not up to date with the remote repository. Please update the git repository.")
        app = QApplication([])
        msg = QMessageBox()
        msg.setWindowTitle("Git Repository not up to date")
        msg.setText("The local git repository is not up to date with the remote repository. Please update the git repository!")
        msg.exec()


    run_GUI()
