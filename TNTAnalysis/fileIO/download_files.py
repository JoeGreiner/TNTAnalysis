import logging
import os
import wget
import zipfile
import platform


def download(link, outpath, overwrite=False):
    if not os.path.exists(outpath) or overwrite:
        logging.info(f'downloading: {link}')
        wget.download(link, out=outpath)
        logging.info('done')
    else:
        logging.info(f'path {outpath} already exists, not downloading')


def unzip_file(path_to_zip_file: str, path_to_extract_directory: str, delete_zip_afterwards: bool = False):
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(path_to_extract_directory)
    if delete_zip_afterwards:
        os.remove(path_to_zip_file)


def download_model(path_to_download_directory: str, overwrite: bool = False):
    logging.info(f'Downloading model to {path_to_download_directory}')
    if not os.path.exists(path_to_download_directory):
        logging.info(f'Creating directory {path_to_download_directory}')
        os.makedirs(path_to_download_directory)
    url_models = 'https://www.iekm.uniklinik-freiburg.de/storage/s/qkcfEALY2jtExbJ/download'
    download(url_models, os.path.join(path_to_download_directory, 'models.zip'), overwrite)
    unzip_file(os.path.join(path_to_download_directory, 'models.zip'), path_to_download_directory)

    return os.path.join(path_to_download_directory, 'nnu_best_model_files')


def download_testfiles(path_to_download_directory: str, overwrite: bool = False):
    logging.info(f'Downloading testfiles to {path_to_download_directory}')
    if not os.path.exists(path_to_download_directory):
        logging.info(f'Creating directory {path_to_download_directory}')
        os.makedirs(path_to_download_directory)
    url_testfiles = 'https://www.iekm.uniklinik-freiburg.de/storage/s/RyoWaiBQsA4AaMt/download'
    download(url_testfiles, os.path.join(path_to_download_directory, 'test_data.zip'), overwrite)
    unzip_file(os.path.join(path_to_download_directory, 'test_data.zip'), path_to_download_directory)

    return os.path.join(path_to_download_directory, 'test_data')

def download_fiji(path_to_download_directory: str, overwrite: bool = False):
    logging.info(f'Downloading Fiji to {path_to_download_directory}')
    if not os.path.exists(path_to_download_directory):
        logging.info(f'Creating directory {path_to_download_directory}')
        os.makedirs(path_to_download_directory)
    if platform.system() == 'Linux':
        logging.info('Downloading Fiji for Linux')
        url_fiji = 'https://downloads.imagej.net/fiji/latest/fiji-linux64.zip'
    elif platform.system() == 'Darwin':
        logging.info('Downloading Fiji for MacOS')
        url_fiji = 'https://downloads.imagej.net/fiji/latest/fiji-macosx.zip'
    elif platform.system() == 'Windows':
        logging.info('Downloading Fiji for Windows')
        url_fiji = 'https://downloads.imagej.net/fiji/latest/fiji-win64.zip'
    else:
        raise NotImplementedError(f'Platform {platform.system()} not supported')
    download(url_fiji, os.path.join(path_to_download_directory, 'fiji.zip'), overwrite)
    unzip_file(os.path.join(path_to_download_directory, 'fiji.zip'), path_to_download_directory)

    return os.path.join(path_to_download_directory, 'Fiji.app')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    path_to_model = download_model('models', overwrite=True)
    path_to_testfiles = download_testfiles('test_data', overwrite=False)
    path_to_fiji = download_fiji('fiji', overwrite=False)
    logging.info(f'Model downloaded to {path_to_model}')
    logging.info(f'Testfiles downloaded to {path_to_testfiles}')
    logging.info(f'Fiji downloaded to {path_to_fiji}')
    logging.info('Finished downloading model and testfiles')
