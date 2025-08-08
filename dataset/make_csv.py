import os
import numpy as np
import csv

absolute_path = os.path.dirname(__file__)
folder_path = "data"
file_list = os.listdir(os.path.join(absolute_path, folder_path))

# Create a CSV file with the same name as the .npy file
csv_file_path = "data.csv"

data_dict = {}
for file_name in file_list:
    if file_name.endswith(".npy"):
        data = np.load(
            os.path.join(absolute_path, folder_path, file_name), allow_pickle=True
        ).item()
        data_dict[file_name] = data


with open(csv_file_path, "w", newline="") as csv_file:
    writer = csv.writer(csv_file, delimiter=",")

    # Write the header row with column names (assuming you have 5 sensor data)
    header = [
        "File Name",
        "Sensor 1",
        "Sensor 2",
        "Sensor 3",
        "Sensor 4",
        "Sensor 5",
        "Ground Truth",
    ]
    writer.writerow(header)

    # Write data from each .npy file into a row
    for file_name, data in data_dict.items():
        sensor_data1 = data["cam_FC_objects"]
        sensor_data2 = data["radar_FR_objects"]
        sensor_data3 = data["radar_FL_objects"]
        sensor_data4 = data["radar_RR_objects"]
        sensor_data5 = data["radar_RL_objects"]
        ground_truth = data["ground_truth_object"]
        sensor_1_ts = data["cam_FC_timestamp"]
        sensor_2_ts = data["radar_FR_timestamp"]
        sensor_3_ts = data["radar_FL_timestamp"]
        sensor_4_ts = data["radar_RR_timestamp"]
        sensor_5_ts = data["radar_RL_timestamp"]

        # Write the data row, including the file name
        row = (
            [file_name]
            + [sensor_data1]
            + [sensor_data2]
            + [sensor_data3]
            + [sensor_data4]
            + [sensor_data5]
            + [ground_truth]
        )
        writer.writerow(row)
        # row = {'File Name':[file_name],
        #        'Sensor 1': sensor_data1,
        #        'Sensor 2': sensor_data2,
        #        'Sensor 3': sensor_data3,
        #        'Sensor 3': sensor_data3,
        #        'Sensor 3': sensor_data3,
        #        'Ground Truth': ground_truth}
        #
        # writer.writerow(row)
