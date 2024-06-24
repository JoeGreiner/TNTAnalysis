import logging
from os import chdir, getcwd, makedirs, environ
from os.path import join, exists, basename
import pandas as pd
from tqdm import tqdm
import scyjava as sj
import imagej

from TNTAnalysis.fileIO.xlsx import write_trackmate_xlsx


# global variable
IMAGEJ_INITIALIZED = False
ij = None


def track_everything_v2(pathlist, variable_config, output_folder="output_tracking"):
    """
    The main function of this python file. Takes in a list of nii.gz files and does the fiji and trackmate pipeline. Takes in segmentation maps,
    associates the spots with a Spot Tracker and then tracks all spots together into tracks with LAP Tracking.
    In the end, creates an output folder and stores all the created tracks in single xlsx files for metrics later.

    :param pathlist: list of all the nii.gz files (predictions of TNT)
    :param variable_config: config dict of variables that decide what you do
    :param output_folder: folder to save the output files
    :return:
    """
    FIJI_PATH = environ["fiji_path"]
    global IMAGEJ_INITIALIZED, ij
    if not IMAGEJ_INITIALIZED:
        if variable_config["INTERACTIVE"]:
            ij = imagej.init(FIJI_PATH, mode='interactive')
        else:
            ij = imagej.init(FIJI_PATH, mode='headless')
        logging.info(f"ImageJ initialized: {ij}")
        IMAGEJ_INITIALIZED = True
    else:
        logging.info("ImageJ already initialized")

    # Import necessary Java classes
    model_class = sj.jimport('fiji.plugin.trackmate.Model')
    settings_class = sj.jimport('fiji.plugin.trackmate.Settings')
    trackmate_class = sj.jimport('fiji.plugin.trackmate.TrackMate')
    logger_class = sj.jimport('fiji.plugin.trackmate.Logger')
    mask_detector_factory_class = sj.jimport('fiji.plugin.trackmate.detection.MaskDetectorFactory')  # Hier ist der Maskendetektor gecalled
    sparse_lap_tracker_factory_class = sj.jimport('fiji.plugin.trackmate.tracking.jaqaman.SparseLAPTrackerFactory')
    selection_model_class = sj.jimport('fiji.plugin.trackmate.SelectionModel')
    hyperstack_displayer_class = sj.jimport('fiji.plugin.trackmate.visualization.hyperstack.HyperStackDisplayer')
    display_settings_io_class = sj.jimport('fiji.plugin.trackmate.gui.displaysettings.DisplaySettingsIO')
    Integer = sj.jimport('java.lang.Integer')
    File = sj.jimport('java.io.File')
    TmXmlWriter = sj.jimport('fiji.plugin.trackmate.io.TmXmlWriter')

    # this will be changed to fiji's working directory, therefore we need to save the current working directory
    old_cwd = getcwd()
    output_directory = output_folder
    if not exists(output_directory):
        makedirs(output_directory)

    for image_path in pathlist:
        # prepare the path of output file
        image_name = basename(image_path).split('/')[-1]
        output_path_combined_xlsx = join(output_directory, image_name[:-7] + ".xlsx")
        logging.info(f"Output will be written to: {output_path_combined_xlsx}")

        # Load image
        image = ij.IJ.openImage(image_path)
        if variable_config["INTERACTIVE"]:
            ij.ui().show(image)

        # get calibration
        cal = image.getCalibration()
        res_width = cal.pixelWidth
        res_height = cal.pixelHeight
        res_time = cal.frameInterval

        # Initialize TrackMate components
        track_model = model_class()
        track_model.setLogger(logger_class.IJ_LOGGER)

        track_settings = settings_class(image)

        # detector settings - change to your needs -> changed to MASK Detector
        track_settings.detectorFactory = mask_detector_factory_class()

        # Set detector settings
        track_settings.detectorSettings = track_settings.detectorFactory.getDefaultSettings()

        # Set tracker factory and settings
        track_settings.trackerFactory = sparse_lap_tracker_factory_class()
        track_settings.trackerSettings = track_settings.trackerFactory.getDefaultSettings()
        track_settings.trackerSettings['LINKING_MAX_DISTANCE'] = 3.0
        track_settings.trackerSettings['GAP_CLOSING_MAX_DISTANCE'] = 3.0
        track_settings.trackerSettings['MAX_FRAME_GAP'] = Integer(1)

        if variable_config["PRINTING"]:
            print(track_settings.detectorFactory.getDefaultSettings())

        track_settings.addAllAnalyzers()              #  VERY IMPORTANT DO NOT DELETE !!!

        trackmate = trackmate_class(track_model, track_settings)

        # Check input and process
        if trackmate.checkInput():
            if not trackmate.process():
                logging.error(str(trackmate.getErrorMessage()))
        else:
            logging.error(str(trackmate.getErrorMessage()))
        logging.info('TrackMate processing completed successfully!')

        if variable_config["SAVE_XML"]:
            out_file = File(output_directory, image_name[:-7] + ".xml")  # this will write the full trackmate xml
            writer = TmXmlWriter(out_file)
            writer.appendModel(track_model)
            writer.appendSettings(track_settings)
            writer.writeToFile()

        # Display results
        if variable_config["INTERACTIVE"]:
            ij.ui().showUI()
            selection_model = selection_model_class(track_model)
            display_settings = display_settings_io_class.readUserDefault()
            displayer = hyperstack_displayer_class(track_model, selection_model, image, display_settings)
            displayer.render()
            displayer.refresh()

        if variable_config["PRINTING"]:
            logging.info(str(track_model)) # prints to standard output
            track_model.getLogger().log(str(track_model)) # prints to imagej console log

        if variable_config["SAVE_XLSX"]:
            # print features
            list_track_ids = []
            list_track_ids_str = []
            list_spot_ids = []
            list_spot_ids_str = []
            list_edge_ids = []
            list_edge_id_str = []

            feature_model = track_model.getFeatureModel()

            feature_names = feature_model.getTrackFeatures()
            spot_feature_names = feature_model.getSpotFeatures()
            edge_feature_names = feature_model.getEdgeFeatures()

            if variable_config["PRINTING"]:
                logging.info(f"Track features: {feature_names}")
                logging.info(f"Spot features: {spot_feature_names}")
                logging.info(f"Edge features: {edge_feature_names}")

            for track_id in tqdm(track_model.getTrackModel().trackIDs(True), desc="Extracting features"):

                dict_tract_features = {}
                for feature_name in feature_names:
                    dict_tract_features[feature_name] = feature_model.getTrackFeature(track_id, feature_name)
                if variable_config["ACTIVATE_BIOFILTER"]:
                    # this is the Biofilter. When active, every track with less than 4 spots gets eliminated
                    if int(dict_tract_features["NUMBER_SPOTS"]) < 4:
                        continue
                list_track_ids = write_list_instead_of_df(list_track_ids, dict_tract_features)
                list_track_ids_str.append(track_id)
                # df_track_ids = pd.concat([df_track_ids, pd.DataFrame(dict_tract_features, index=[track_id])])


                for edge in track_model.getTrackModel().trackEdges(track_id):
                    dict_edge_features = {}
                    for feature_name in edge_feature_names:
                        dict_edge_features[feature_name] = feature_model.getEdgeFeature(edge, feature_name)
                    id_str = f"ID{int(feature_model.getEdgeFeature(edge, 'SPOT_SOURCE_ID'))} → ID{int(feature_model.getEdgeFeature(edge, 'SPOT_TARGET_ID'))}"
                    list_edge_ids = write_list_instead_of_df(list_edge_ids, dict_edge_features)
                    list_edge_id_str.append(id_str)
                    # df_edge_ids = pd.concat([df_edge_ids, pd.DataFrame(dict_edge_features, index=[id_str])])


                for spot in track_model.getTrackModel().trackSpots(track_id):
                    dict_spot_features = {}
                    dict_spot_features["TRACK_ID"] = track_id
                    for feature_name in spot_feature_names:
                        dict_spot_features[feature_name] = spot.getFeature(feature_name)
                    dict_spot_features["INDEX_X"] = dict_spot_features["POSITION_X"] / res_width
                    dict_spot_features["INDEX_Y"] = dict_spot_features["POSITION_Y"] / res_height
                    list_spot_ids = write_list_instead_of_df(list_spot_ids, dict_spot_features)
                    list_spot_ids_str.append(spot.ID())
                    # df_spot_ids = pd.concat([df_spot_ids, pd.DataFrame(dict_spot_features, index=[spot.ID()])])

            df_edge_ids_via_list = pd.DataFrame(list_edge_ids, columns=list(dict_edge_features.keys()), index=list_edge_id_str)
            df_spot_ids_via_list = pd.DataFrame(list_spot_ids, columns=list(dict_spot_features.keys()), index=list_spot_ids_str)
            df_track_ids_via_list = pd.DataFrame(list_track_ids, columns=list(dict_tract_features.keys()), index=list_track_ids_str)

            # print(f"The spots are the same? : {df_spot_ids.equals(df_spot_ids_via_list)}")
            # print(f"The edges are the same? : {df_edge_ids.equals(df_edge_ids_via_list)}")
            # print(f"The tracks are the same? : {df_track_ids.equals(df_track_ids_via_list)}")

            # add frame duration and resolution to the track features
            df_track_ids_via_list["FRAME_DURATION"] = res_time
            df_track_ids_via_list["RESOLUTION_X"] = res_width
            df_track_ids_via_list["RESOLUTION_Y"] = res_height
            df_track_ids_via_list['TRACK_Y_LOCATION_INDEX'] = df_track_ids_via_list['TRACK_Y_LOCATION'] / res_height
            df_track_ids_via_list['TRACK_X_LOCATION_INDEX'] = df_track_ids_via_list['TRACK_X_LOCATION'] / res_width

            # do date + replicate addition for easier analysis in in graphpad
            # image_path = /home/greinerj/tmp/vanessa_demo/output_folder/inference_predictions/2023-11-16_1_GFP-ctrl_TREK1-transfectionlif.nii.gz (e.g.)
            # date = 2023-11-16
            # replicate = 1
            image_path_basename = basename(image_path)
            image_path_basename_split = image_path_basename.split('_')
            if len(image_path_basename_split) == 0:
                date = "unknown"
                replicate = "unknown"
            elif len(image_path_basename_split) == 1:
                date = image_path_basename_split[0]
                replicate = "unknown"
            else:
                date = image_path_basename.split('_')[0]
                replicate = image_path_basename.split('_')[1]

            df_track_ids_via_list['DATE'] = date
            df_track_ids_via_list['REPLICATE'] = replicate
            df_track_ids_via_list['IMAGE_PATH'] = image_path_basename

            df_spot_ids_via_list['DATE'] = date
            df_spot_ids_via_list['REPLICATE'] = replicate
            df_spot_ids_via_list['IMAGE_PATH'] = image_path_basename

            df_edge_ids_via_list['DATE'] = date
            df_edge_ids_via_list['REPLICATE'] = replicate
            df_edge_ids_via_list['IMAGE_PATH'] = image_path_basename


            logging.info(f"Writing to xlsx file: {output_path_combined_xlsx}")
            # write_trackmate_xlsx(df_spot=df_spot_ids, df_edge=df_edge_ids, df_track=df_track_ids,
            #                      path_xlsx_out=output_path_combined_xlsx)
            write_trackmate_xlsx(df_spot=df_spot_ids_via_list,
                                 df_edge=df_edge_ids_via_list,
                                 df_track=df_track_ids_via_list,
                                 path_xlsx_out=output_path_combined_xlsx)


        # set working directory
        logging.info(f"Switching to working directory from {getcwd()} to {old_cwd}")
        chdir(old_cwd)

        if variable_config["SAVE_MASK"]:
            image_corrected_path = join(output_directory, image_name[:-7] + "_corrected")
            logging.info(f"Saving Image with correct dimensions and channels back to: {image_corrected_path}")
            ij.IJ.saveAs(image, "TIFF", image_corrected_path)

        logging.info('Iteration Done')
    logging.info('It is finally Done')

def write_list_instead_of_df(combined_data_list, new_data, id_str=None):
    if id_str is not None:
        listo = [id_str] + new_data.values()
        combined_data_list.append(listo)
    else:
        combined_data_list.append(list(new_data.values()))
    return combined_data_list