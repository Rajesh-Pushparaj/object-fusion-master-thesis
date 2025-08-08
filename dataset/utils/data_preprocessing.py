import os
import numpy as np
from utils import SDFReader

from operator import itemgetter
from sensor_topics import *
from data_utils import *
from environment_model.ctrv_movement_calculator import CTRVMovementCalculator

import math

# Create a constant turn rate and velocity model object and register follower
ctrv_movement_calculator = CTRVMovementCalculator()
ctrv_movement_calculator.register_follower("fusion")

# Path to the sdf files
absolute_path = os.path.dirname(__file__)
relaive_path = "sdf_files/KISSaF_2023_06_09_064504_extended.sdf"
relaive_path_gt = "sdf_files/KISSaF_2023_01_24_114638_extended_GT.sdf"
HDF_file_path = os.path.join(absolute_path, relaive_path)
HDF_file_path_gt = os.path.join(absolute_path, relaive_path_gt)
# path to store datasets
dataset_path = "data"


def next_array_with_dim(iter):
    elmn = next(iter)[np.newaxis]
    return elmn


# Define the data types and field names for each column in the structured array
obj_dtype = [
    ("class_id", "int32"),
    ("x", "float64"),
    ("y", "float64"),
    ("v_x", "float64"),
    ("v_y", "float64"),
    ("length", "float64"),
    ("width", "float64"),
    ("heading", "float64"),
]

dtype = np.dtype([("sensor", "int32"), ("time", "int64"), ("obj", "object")])


def get_object_velocity(Sensor_obj):
    obj = Sensor_obj
    StTyp = obj["St"]["StTyp"]
    if StTyp == 2:
        ego_x_vel = ctrv_movement_calculator.velocity
        psi = obj["St"]["Mean"][2]
        sin_psi = np.sin(psi)
        cos_psi = np.cos(psi)
        return {
            "v_x_rel": cos_psi * obj["St"]["Mean"][3] - ego_x_vel,
            "v_y_rel": sin_psi * obj["St"]["Mean"][3],
        }
    elif StTyp == 3:
        ego_x_vel = ctrv_movement_calculator.velocity
        psi = obj["St"]["Mean"][2]
        sin_psi = np.sin(psi)
        cos_psi = np.cos(psi)
        return {
            "v_x_rel": cos_psi * obj["St"]["Mean"][3] - ego_x_vel,
            "v_y_rel": sin_psi * obj["St"]["Mean"][3],
        }
    elif StTyp == 4:
        ego_x_vel = ctrv_movement_calculator.velocity
        return {
            "v_x_rel": obj["St"]["Mean"][2] - ego_x_vel,
            "v_y_rel": obj["St"]["Mean"][3],
        }
    elif StTyp == 12:
        return {
            "v_x_rel": obj["St"]["Mean"][2],
            "v_y_rel": 0.0,
        }
    elif StTyp == 15:
        return {
            "v_x_rel": obj["St"]["Mean"][2],
            "v_y_rel": obj["St"]["Mean"][3],
        }
    elif StTyp == 18:
        return {
            "v_x_rel": obj["St"]["Mean"][2],
            "v_y_rel": obj["St"]["Mean"][4],
        }
    else:
        return {
            "v_x_rel": 0.0,
            "v_y_rel": 0.0,
        }


def extract_object_parameters(sdf_reader, topics, dataTyp):

    objectTimed = []

    # Get the iterators
    (
        sim_objs_time_iter,
        sim_objs_data_iter,
        sim_objs_sens,
        max_time,
        sim_objs_time_nditer,
        sim_objs_data_nditer,
    ) = sdf_reader.data2iterator(topics, 0)

    sensor_name = []
    reading_type = []
    for sensor_info in sim_objs_sens:
        sensor_name.append(sensor_info[0])
        reading_type.append(sensor_info[1])

    # next timestamps from the iterator
    next_times = np.round(
        np.fromiter(map(next, sim_objs_time_nditer), dtype=np.uint64) * 1e-3
    ).astype(np.uint64)
    # next data from the iterator
    next_data = list(map(next_array_with_dim, sim_objs_data_nditer))

    while next_data:
        try:
            # Iteratae over the objects in each timestamp
            for t in range(len(next_times)):
                objects = []
                curr_time = itemgetter(t)(next_times).astype("int64")
                curr_objs = itemgetter(t)(next_data)
                curr_sensors = itemgetter(t)(sensor_name)

                # Check if the sensor topic is Vehicle Internal
                if curr_sensors.value == 10:
                    # vehicle_vel  = curr_objs['Data']['HstVehSpd'][0]
                    vhl_speed_mps = curr_objs["Data"]["HstVehSpd"]
                    vhl_yaw_rate = curr_objs["Data"]["HstVehYawRateFild"]
                    ctrv_movement_calculator.new_measurement(
                        vhl_speed_mps[0], vhl_yaw_rate[0], curr_time
                    )
                else:
                    # Access object data
                    for n in range(curr_objs[0]["NumObjs"]):
                        Sensor_obj = curr_objs[0]["Obj"][n]

                        # filter objects with existence probability less than the threshold
                        if dataTyp == "gt" and Sensor_obj["Cmn"]["ExistenceProb"] < 0.9:
                            continue

                        # Get object velocity
                        # objVelocity = get_object_velocity(Sensor_obj, vehicle_vel)
                        objVelocity = get_object_velocity(Sensor_obj)
                        # extract object parameteres from the list
                        class_id = Sensor_obj["Cmn"]["Class"]
                        x = Sensor_obj["St"]["Mean"][0]
                        y = Sensor_obj["St"]["Mean"][1]
                        length = Sensor_obj["Cmn"]["Extsn"]["Length"]
                        width = Sensor_obj["Cmn"]["Extsn"]["Width"]
                        heading = Sensor_obj["Cmn"]["Orientation"]
                        v_x = objVelocity["v_x_rel"]
                        v_y = objVelocity["v_y_rel"]
                        exstProb = Sensor_obj["Cmn"]["ExistenceProb"]

                        # filter data within 100m range with origin being (0,0)
                        radius = 100
                        distance = math.sqrt((x) ** 2 + (y) ** 2)
                        if distance > radius:
                            continue

                        if dataTyp == "gt":
                            data = [x, y, length, width, v_x, v_y, heading, class_id]
                            data = normalizeData(data)
                        else:
                            ObjPresent = 1  # parameter to differentiate between object and padding
                            # Create the structured array
                            data = [x, y, length, width, v_x, v_y, heading, class_id]
                            data = normalizeData(data)
                            data.append(exstProb)
                            # data.append(ObjPresent)
                            # data = [curr_sensors.value, curr_time, class_id, x, y, v_x, v_y, length, width, heading]
                        objects.append(data)

                if curr_sensors.value != 10:
                    if objects:
                        # objectTimed.append(objects)
                        objectTimed.append((curr_sensors.value, curr_time, objects))
            # next timestamps from the iterator
            next_times = np.round(
                np.fromiter(map(next, sim_objs_time_nditer), dtype=np.uint64) * 1e-3
            ).astype(np.uint64)
            # next data from the iterator
            next_data = list(map(next_array_with_dim, sim_objs_data_nditer))

        except StopIteration:
            # If the iterator is exhausted, break the loop
            break
    max_object_size = 10
    padding_array = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    padded_data = []
    if dataTyp == "gt":
        padded_data = objectTimed
    else:
        for sensor_id, timestamp, objects_list in objectTimed:
            padded_objects_list = objects_list[:max_object_size] + [padding_array] * (
                max_object_size - len(objects_list)
            )
            padded_data.append((sensor_id, timestamp, padded_objects_list))

    # Make an array of all objects of type dtype
    objectList = np.array(padded_data, dtype=dtype)

    return padded_data, objectList


def find_closest_timestamp(ground_truth_data, timestamp, threshold=1000):
    for _, gt_timestamp, _ in ground_truth_data:
        if (
            abs(timestamp - gt_timestamp) <= threshold
        ):  # take data +-1000 from the timestamp
            return gt_timestamp
    return None


def find_closest_previous_timestamp(sensor_data, timestamp, threshold=100):
    closest_timestamps = {}
    delta_map = {5: threshold, 6: threshold, 3: threshold, 8: threshold, 7: threshold}
    for sensor_id, sensor_timestamp, sensor_object in sensor_data:
        delta_ts = timestamp - sensor_timestamp
        if (delta_ts < threshold) and (
            delta_ts >= 0
        ):  # take only data -threshold from the timestamp
            objList = np.array(sensor_object)
            if np.any(objList != 0):
                if delta_map[sensor_id] > delta_ts:
                    delta_map.update({sensor_id: delta_ts})
                    if sensor_id in closest_timestamps:
                        closest_timestamps.update({sensor_id: sensor_timestamp})
                    else:
                        closest_timestamps.setdefault(
                            sensor_id, sensor_timestamp
                        )  # .append(sensor_timestamp)
    return closest_timestamps


if __name__ == "__main__":

    # Initialze SDFReader object
    # sdf_reader = SDFReader(HDF_file_path)
    # sdf_reader.log_sdf_file_info()
    print("starting..")
    sdf_reader_gt = SDFReader(HDF_file_path_gt)
    sdf_reader_gt.log_sdf_file_info()
    print(f"Reading file from {relaive_path_gt}")
    # get cam and radar object list
    camRadar_objectList, _ = extract_object_parameters(
        sdf_reader_gt, asw_topics_cam_radar, "sd"
    )
    # get lidar grounhd truth object list
    lidarGt_objectList, _ = extract_object_parameters(
        sdf_reader_gt, asw_topics_lidar_gt, "gt"
    )

    # generate sensor data and ground truth pairs and save it in .npy files
    combined_data = []

    # Dictionary to store the mapping between ground truth timestamp and closest sensor timestamps
    timestamp_mapping = {}
    print("finding closest ts")
    for _, ground_truth_timestamp, _ in lidarGt_objectList:
        closest_timestamps = find_closest_previous_timestamp(
            camRadar_objectList, ground_truth_timestamp
        )
        timestamp_mapping[ground_truth_timestamp] = closest_timestamps
    print("organising objects from closest ts")
    # generate a dict for sensor objects and ground truth objects for mapped timestamp
    for ground_truth_timestamp, closest_timestamps in timestamp_mapping.items():
        dataDict = {
            "cam_FC_objects": None,
            "radar_FR_objects": None,
            "radar_FL_objects": None,
            "radar_RR_objects": None,
            "radar_RL_objects": None,
            "cam_FC_timestamp": None,
            "radar_FR_timestamp": None,
            "radar_FL_timestamp": None,
            "radar_RR_timestamp": None,
            "radar_RL_timestamp": None,
            "ground_truth_timestamp": ground_truth_timestamp,
            "ground_truth_object": None,
        }
        # get ground truth object
        index = None
        for i, (_, timestamp, _) in enumerate(lidarGt_objectList):
            if timestamp == ground_truth_timestamp:
                index = i
                break
        dataDict["ground_truth_object"] = lidarGt_objectList[index][2]
        # get sensor data objects
        for sensor_id, sensor_timestamps in closest_timestamps.items():
            index = None
            for i, (id, timestamp, _) in enumerate(camRadar_objectList):
                if id == sensor_id and timestamp == sensor_timestamps:
                    index = i
                    break
            if sensor_id == 3:
                dataDict["cam_FC_objects"] = camRadar_objectList[index][2]
                dataDict["cam_FC_timestamp"] = sensor_timestamps
            elif sensor_id == 5:
                dataDict["radar_FL_objects"] = camRadar_objectList[index][2]
                dataDict["radar_FL_timestamp"] = sensor_timestamps
            elif sensor_id == 6:
                dataDict["radar_FR_objects"] = camRadar_objectList[index][2]
                dataDict["radar_FR_timestamp"] = sensor_timestamps
            elif sensor_id == 7:
                dataDict["radar_RL_objects"] = camRadar_objectList[index][2]
                dataDict["radar_RL_timestamp"] = sensor_timestamps
            elif sensor_id == 8:
                dataDict["radar_RR_objects"] = camRadar_objectList[index][2]
                dataDict["radar_RR_timestamp"] = sensor_timestamps

        # remove ground truth that doesnt have any sensor data
        ground_truth_objects = dataDict["ground_truth_object"]
        sensor_objects = [
            dataDict["cam_FC_objects"],
            dataDict["radar_FR_objects"],
            dataDict["radar_FL_objects"],
            dataDict["radar_RR_objects"],
            dataDict["radar_RL_objects"],
        ]
        fixed_radius = 5
        min_area = 0.5
        gt = []
        for gt_object in ground_truth_objects:
            # Initialize a flag to check if a match is found
            match_found = False

            l = (gt_object[2] * (lRange[1] - lRange[0])) + lRange[0]  # Denormalize 'l'
            w = (gt_object[3] * (wRange[1] - wRange[0])) + wRange[0]  # Denormalize 'w'
            area = l * w
            if area < min_area:
                continue

            # Iterate through sensor objects from all sensors
            for sensor_object in sensor_objects:
                if match_found:
                    break
                if sensor_object is None:
                    continue  # Skip None values (if sensor data is missing)

                # Iterate through the sensor objects to check for matches
                for sensor_data in sensor_object:
                    x1 = (gt_object[0] * (xRange[1] - xRange[0])) + xRange[
                        0
                    ]  # Denormalize 'x'
                    y1 = (gt_object[1] * (yRange[1] - yRange[0])) + yRange[
                        0
                    ]  # Denormalize 'y'
                    x2 = (sensor_data[0] * (xRange[1] - xRange[0])) + xRange[
                        0
                    ]  # Denormalize 'x'
                    y2 = (sensor_data[1] * (yRange[1] - yRange[0])) + yRange[
                        0
                    ]  # Denormalize 'y'
                    # Calculate the distance between the ground truth object and sensor object
                    distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

                    # Check if the distance is within the fixed radius
                    if distance <= fixed_radius:
                        match_found = True
                        break  # Stop searching for matches for this ground truth object

            if match_found:
                gt.append(gt_object)

        # Update the 'ground_truth_object' field in your dataDict
        # if gt:
        #     dataDict['ground_truth_object'] = gt
        #     combined_data.append(dataDict)
        if gt:
            dataDict["ground_truth_object"] = gt
            padding_object = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            sensorObjPrs = False
            # Iterate through each sensor type
            for i, sensor_type in enumerate(sensor_objects):
                valid_sensor_objects_type = []
                if sensor_type:
                    # Iterate through the sensor objects of the current sensor type
                    for sensor_object in sensor_type:
                        # Initialize a flag to check if a match is found
                        match_found = False

                        # Iterate through ground truth objects
                        for gt_object in gt:
                            # Calculate the distance between the sensor object and ground truth object
                            x1 = (gt_object[0] * (xRange[1] - xRange[0])) + xRange[
                                0
                            ]  # Denormalize 'x'
                            y1 = (gt_object[1] * (yRange[1] - yRange[0])) + yRange[
                                0
                            ]  # Denormalize 'y'
                            x2 = (sensor_object[0] * (xRange[1] - xRange[0])) + xRange[
                                0
                            ]  # Denormalize 'x'
                            y2 = (sensor_object[1] * (yRange[1] - yRange[0])) + yRange[
                                0
                            ]  # Denormalize 'y'
                            # Calculate the distance between the ground truth object and sensor object
                            distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

                            # Check if the distance is within the fixed radius
                            if distance <= fixed_radius:
                                match_found = True
                                break  # Stop searching for matches for this sensor object

                        if match_found:
                            valid_sensor_objects_type.append(sensor_object)
                            sensorObjPrs = True
                        else:
                            # Add padding object if no match is found
                            valid_sensor_objects_type.append(padding_object)

                # Update the sensor objects for the current sensor type
                if i == 0:
                    dataDict["cam_FC_objects"] = valid_sensor_objects_type
                elif i == 1:
                    dataDict["radar_FR_objects"] = valid_sensor_objects_type
                elif i == 2:
                    dataDict["radar_FL_objects"] = valid_sensor_objects_type
                elif i == 3:
                    dataDict["radar_RR_objects"] = valid_sensor_objects_type
                elif i == 4:
                    dataDict["radar_RL_objects"] = valid_sensor_objects_type
                else:
                    raise ValueError("Invalid sensor type")

            # Update the 'ground_truth_object' field in your dataDict
            if sensorObjPrs:
                combined_data.append(dataDict)

    print("writing to files")
    # Save the object lists as a numpy file
    for i, data_pair in enumerate(combined_data):
        data_pair_filename = f"sensor_data_{i}.npy"
        np.save(
            os.path.join(absolute_path, dataset_path, data_pair_filename),
            data_pair,
            allow_pickle=True,
        )

    print("Finished")
