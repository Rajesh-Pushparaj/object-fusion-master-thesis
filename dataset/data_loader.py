import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import pickle
import tqdm
from numpy.lib.recfunctions import structured_to_unstructured
from dataset.utils.data_utils import normalize2dData, denormalize2dData


class FusionAndTrackingDataset(Dataset):
    def __init__(self, sensor_data_folder_path, Noise=False):
        # get the files list
        # sensorDataFiles = os.listdir(sensor_data_folder_path)
        # sensorDataFiles.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        data_paths = []
        if os.path.exists(sensor_data_folder_path):
            with open(sensor_data_folder_path, "rb") as f:
                data_paths.extend(pickle.load(f))

        indx = sensor_data_folder_path.find("fusion_filtered")
        self.sensor_data_folder_path = sensor_data_folder_path[:indx]
        self.sensorDataFiles = data_paths
        self.no_objs = 20
        self.Noise = Noise

    def __len__(self):
        return len(self.sensorDataFiles)

    def get_filtered_data(self, data):
        a = []
        for x in data:
            b = x[:4]  # get x, y, length, width
            b.append(x[7])  # get class_id
            b.append(x[8])  # get existence prob
            a.append(b)
        return a

    def __getitem__(self, idx):
        file_path = os.path.join(
            self.sensor_data_folder_path, self.sensorDataFiles[idx]
        )
        file_path = file_path.replace("\\", "/")
        # dataDict = np.load(file_path, allow_pickle=True).item()
        with open(file_path, "rb") as f:
            dataDict = pickle.load(f)  # , allow_pickle=True).item()
        pad = torch.zeros(
            (self.no_objs, 16)
        )  # make it 6 when removing vx, vy, heading params

        data_in = dataDict["data_in"]

        objs_CAMERA_FC0 = np.zeros((self.no_objs, 16))
        objs_RADAR_FR0 = np.zeros((self.no_objs, 16))
        objs_RADAR_FL0 = np.zeros((self.no_objs, 16))
        objs_RADAR_RR0 = np.zeros((self.no_objs, 16))
        objs_RADAR_RL0 = np.zeros((self.no_objs, 16))

        d_ts1 = 0
        d_ts2 = 0
        d_ts3 = 0
        d_ts4 = 0
        d_ts5 = 0

        for sensor_readings, sensor_id in data_in:

            if sensor_id[0] == "CAMERA_FC0":
                objs_CAMERA_FC0 = structured_to_unstructured(
                    sensor_readings[
                        [
                            "x",
                            "y",
                            "length",
                            "width",
                            "v_x",
                            "v_y",
                            "heading",
                            "class_id",
                            "p_exist",
                            "x_std_dev",
                            "y_std_dev",
                            "length_std_dev",
                            "width_std_dev",
                            "v_x_std_dev",
                            "v_y_std_dev",
                            "heading_std_dev",
                        ]
                    ]
                )
                objs_CAMERA_FC0 = normalize2dData(objs_CAMERA_FC0)
                objs_CAMERA_FC0[:, 7] = self.map_class_labels(objs_CAMERA_FC0[:, 7])
                n_boxes = len(objs_CAMERA_FC0)
                d_ts1 = max(
                    structured_to_unstructured(sensor_readings[["time"]])
                ).item()
                objs_CAMERA_FC0 = np.pad(
                    objs_CAMERA_FC0,
                    ((0, self.no_objs - n_boxes), (0, 0)),
                    constant_values=0,
                )
            if sensor_id[0] == "RADAR_FR0":
                objs_RADAR_FR0 = structured_to_unstructured(
                    sensor_readings[
                        [
                            "x",
                            "y",
                            "length",
                            "width",
                            "v_x",
                            "v_y",
                            "heading",
                            "class_id",
                            "p_exist",
                            "x_std_dev",
                            "y_std_dev",
                            "length_std_dev",
                            "width_std_dev",
                            "v_x_std_dev",
                            "v_y_std_dev",
                            "heading_std_dev",
                        ]
                    ]
                )
                objs_RADAR_FR0 = normalize2dData(objs_RADAR_FR0)
                objs_RADAR_FR0[:, 7] = self.map_class_labels(objs_RADAR_FR0[:, 7])
                d_ts2 = max(
                    structured_to_unstructured(sensor_readings[["time"]])
                ).item()
                n_boxes = len(objs_RADAR_FR0)
                objs_RADAR_FR0 = np.pad(
                    objs_RADAR_FR0,
                    ((0, self.no_objs - n_boxes), (0, 0)),
                    constant_values=0,
                )
            if sensor_id[0] == "RADAR_FL0":
                objs_RADAR_FL0 = structured_to_unstructured(
                    sensor_readings[
                        [
                            "x",
                            "y",
                            "length",
                            "width",
                            "v_x",
                            "v_y",
                            "heading",
                            "class_id",
                            "p_exist",
                            "x_std_dev",
                            "y_std_dev",
                            "length_std_dev",
                            "width_std_dev",
                            "v_x_std_dev",
                            "v_y_std_dev",
                            "heading_std_dev",
                        ]
                    ]
                )
                objs_RADAR_FL0 = normalize2dData(objs_RADAR_FL0)
                objs_RADAR_FL0[:, 7] = self.map_class_labels(objs_RADAR_FL0[:, 7])
                d_ts3 = max(
                    structured_to_unstructured(sensor_readings[["time"]])
                ).item()
                n_boxes = len(objs_RADAR_FL0)
                objs_RADAR_FL0 = np.pad(
                    objs_RADAR_FL0,
                    ((0, self.no_objs - n_boxes), (0, 0)),
                    constant_values=0,
                )
            if sensor_id[0] == "RADAR_RR0":
                objs_RADAR_RR0 = structured_to_unstructured(
                    sensor_readings[
                        [
                            "x",
                            "y",
                            "length",
                            "width",
                            "v_x",
                            "v_y",
                            "heading",
                            "class_id",
                            "p_exist",
                            "x_std_dev",
                            "y_std_dev",
                            "length_std_dev",
                            "width_std_dev",
                            "v_x_std_dev",
                            "v_y_std_dev",
                            "heading_std_dev",
                        ]
                    ]
                )
                objs_RADAR_RR0 = normalize2dData(objs_RADAR_RR0)
                objs_RADAR_RR0[:, 7] = self.map_class_labels(objs_RADAR_RR0[:, 7])
                d_ts4 = max(
                    structured_to_unstructured(sensor_readings[["time"]])
                ).item()
                n_boxes = len(objs_RADAR_RR0)
                objs_RADAR_RR0 = np.pad(
                    objs_RADAR_RR0,
                    ((0, self.no_objs - n_boxes), (0, 0)),
                    constant_values=0,
                )
            if sensor_id[0] == "RADAR_RL0":
                objs_RADAR_RL0 = structured_to_unstructured(
                    sensor_readings[
                        [
                            "x",
                            "y",
                            "length",
                            "width",
                            "v_x",
                            "v_y",
                            "heading",
                            "class_id",
                            "p_exist",
                            "x_std_dev",
                            "y_std_dev",
                            "length_std_dev",
                            "width_std_dev",
                            "v_x_std_dev",
                            "v_y_std_dev",
                            "heading_std_dev",
                        ]
                    ]
                )
                objs_RADAR_RL0 = normalize2dData(objs_RADAR_RL0)
                objs_RADAR_RL0[:, 7] = self.map_class_labels(objs_RADAR_RL0[:, 7])
                d_ts5 = max(
                    structured_to_unstructured(sensor_readings[["time"]])
                ).item()
                n_boxes = len(objs_RADAR_RL0)
                objs_RADAR_RL0 = np.pad(
                    objs_RADAR_RL0,
                    ((0, self.no_objs - n_boxes), (0, 0)),
                    constant_values=0,
                )

        sensor_data1 = torch.tensor(objs_CAMERA_FC0)
        sensor_data2 = torch.tensor(objs_RADAR_FR0)
        sensor_data3 = torch.tensor(objs_RADAR_FL0)
        sensor_data4 = torch.tensor(objs_RADAR_RR0)
        sensor_data5 = torch.tensor(objs_RADAR_RL0)

        # # cam_FC_objects
        # if not dataDict['cam_FC_objects']:
        #     sensor_data1 = pad
        #     d_ts1 = 100
        # else:
        #     # sensor_data1 = torch.tensor(dataDict['cam_FC_objects'])
        #     # dat = self.get_filtered_data(dataDict['cam_FC_objects'])
        #     dat = dataDict['cam_FC_objects']
        #     sensor_data1 = torch.tensor(dat)
        #     d_ts1 = dataDict['ground_truth_timestamp'] - dataDict['cam_FC_timestamp']
        # # radar_FR_objects
        # if not dataDict['radar_FR_objects']:
        #     sensor_data2 = pad
        #     d_ts2 = 100
        # else:
        #     # sensor_data2 = torch.tensor(dataDict['radar_FR_objects'])
        #     # dat = self.get_filtered_data(dataDict['radar_FR_objects'])
        #     dat = dataDict['radar_FR_objects']
        #     sensor_data2 = torch.tensor(dat)
        #     d_ts2 = dataDict['ground_truth_timestamp'] - dataDict['radar_FR_timestamp']
        # # radar_FL_objects
        # if not dataDict['radar_FL_objects']:
        #     sensor_data3 = pad
        #     d_ts3 = 100
        # else:
        #     # sensor_data3 = torch.tensor(dataDict['radar_FL_objects'])
        #     # dat = self.get_filtered_data(dataDict['radar_FL_objects'])
        #     dat = dataDict['radar_FL_objects']
        #     sensor_data3 = torch.tensor(dat)
        #     d_ts3 = dataDict['ground_truth_timestamp'] - dataDict['radar_FL_timestamp']
        # # radar_RR_objects
        # if not dataDict['radar_RR_objects']:
        #     sensor_data4 = pad
        #     d_ts4 = 100
        # else:
        #     # sensor_data4 = torch.tensor(dataDict['radar_RR_objects'])
        #     # dat = self.get_filtered_data(dataDict['radar_RR_objects'])
        #     dat = dataDict['radar_RR_objects']
        #     sensor_data4 = torch.tensor(dat)
        #     d_ts4 = dataDict['ground_truth_timestamp'] - dataDict['radar_RR_timestamp']
        # # radar_RL_objects
        # if not dataDict['radar_RL_objects']:
        #     sensor_data5 = pad
        #     d_ts5 = 100
        # else:
        #     # sensor_data5 = torch.tensor(dataDict['radar_RL_objects'])
        #     # dat = self.get_filtered_data(dataDict['radar_RL_objects'])
        #     dat = dataDict['radar_RL_objects']
        #     sensor_data5 = torch.tensor(dat)
        #     d_ts5 = dataDict['ground_truth_timestamp'] - dataDict['radar_RL_timestamp']

        # ground_truth = torch.tensor(dataDict['ground_truth_object'])

        # if self.Noise:
        #     # add white noise to inputs
        #     transform = AddWhiteNoise(noise_std=0.05)
        #     sensor_data1 = transform(sensor_data1)
        #     sensor_data2 = transform(sensor_data2)
        #     sensor_data3 = transform(sensor_data3)
        #     sensor_data4 = transform(sensor_data4)
        #     sensor_data5 = transform(sensor_data5)

        # Create a mask to identify padded elements (assuming padded elements are zero)
        mask1 = torch.any(sensor_data1, dim=1).unsqueeze(-1).float()
        mask2 = torch.any(sensor_data2, dim=1).unsqueeze(-1).float()
        mask3 = torch.any(sensor_data3, dim=1).unsqueeze(-1).float()
        mask4 = torch.any(sensor_data4, dim=1).unsqueeze(-1).float()
        mask5 = torch.any(sensor_data5, dim=1).unsqueeze(-1).float()

        sensorID1 = 0.0
        sensorID2 = 0.25
        sensorID3 = 0.5
        sensorID4 = 0.75
        sensorID5 = 1.0

        # delTsRange = [0, 100]
        # d_ts1 = (d_ts1-delTsRange[0])/(delTsRange[1]-delTsRange[0])
        # d_ts2 = (d_ts2-delTsRange[0])/(delTsRange[1]-delTsRange[0])
        # d_ts3 = (d_ts3-delTsRange[0])/(delTsRange[1]-delTsRange[0])
        # d_ts4 = (d_ts4-delTsRange[0])/(delTsRange[1]-delTsRange[0])
        # d_ts5 = (d_ts5-delTsRange[0])/(delTsRange[1]-delTsRange[0])

        delta_ts1 = torch.full((self.no_objs, 1), d_ts1)
        delta_ts2 = torch.full((self.no_objs, 1), d_ts2)
        delta_ts3 = torch.full((self.no_objs, 1), d_ts3)
        delta_ts4 = torch.full((self.no_objs, 1), d_ts4)
        delta_ts5 = torch.full((self.no_objs, 1), d_ts5)

        sensor_data_1 = torch.cat([sensor_data1, mask1 * sensorID1], dim=1)
        sensor_data_2 = torch.cat([sensor_data2, mask2 * sensorID2], dim=1)
        sensor_data_3 = torch.cat([sensor_data3, mask3 * sensorID3], dim=1)
        sensor_data_4 = torch.cat([sensor_data4, mask4 * sensorID4], dim=1)
        sensor_data_5 = torch.cat([sensor_data5, mask5 * sensorID5], dim=1)

        sensor_data_1 = torch.cat([sensor_data_1, mask1 * delta_ts1], dim=-1)
        sensor_data_2 = torch.cat([sensor_data_2, mask2 * delta_ts2], dim=-1)
        sensor_data_3 = torch.cat([sensor_data_3, mask3 * delta_ts3], dim=-1)
        sensor_data_4 = torch.cat([sensor_data_4, mask4 * delta_ts4], dim=-1)
        sensor_data_5 = torch.cat([sensor_data_5, mask5 * delta_ts5], dim=-1)

        inputs = [
            sensor_data_1,
            sensor_data_2,
            sensor_data_3,
            sensor_data_4,
            sensor_data_5,
        ]  # input params are 8 after removing motion params

        # target preprocess
        ground_truth = dataDict["data_gt"]
        gt_data = structured_to_unstructured(
            ground_truth[
                [
                    "x",
                    "y",
                    "length",
                    "width",
                    "v_x",
                    "v_y",
                    "heading",
                    "class_id",
                    "p_exist",
                ]
            ]
        )
        gt_data = normalize2dData(gt_data)
        target_bbox = torch.tensor(gt_data[:, :4])
        # target_class = torch.tensor(structured_to_unstructured(ground_truth[["class_id"]], dtype="float32"))
        target_class = torch.tensor(gt_data[:, 7]).unsqueeze(1)
        target_class = self.map_class_labels(
            target_class
        )  # encode target class labels (0-n)
        target_motion = torch.tensor(gt_data[:, 4:6])

        # target_bbox, target_motion, target_class = ground_truth.split([4, 3, 1], dim=-1)

        targets = {"labels": target_class, "BBox": target_bbox, "mot": target_motion}

        return inputs, targets

    def map_class_labels(self, class_labels):
        label_map = {
            9: 0,
            7: 1,
            10: 2,
            11: 3,
            12: 4,
        }  # {truck, car, motorcycle, bicycle, pedestrian}
        # Map tensor elements to new values
        if torch.is_tensor(class_labels):
            class_labels_mapped = torch.tensor(
                [[label_map[val.item()] for val in row] for row in class_labels]
            )
        else:
            class_labels_mapped = [label_map[labels] for labels in class_labels]

        return class_labels_mapped


def collate_fn(batch):
    inputs, targets = zip(*batch)

    # Stack the input tensors along a new dimension
    input = []
    for i in range(len(inputs)):
        input.append(torch.stack(inputs[i], dim=0))
    inputs = torch.stack(input, dim=0)

    return inputs, targets
    # batch = zip(*batch)
    # return tuple(batch)


def create_dataloader(datasetPath, batchSize):
    # path to datasets
    # absolute_path = os.path.dirname(__file__)
    sensor_data_folder_path = os.path.normpath(datasetPath)

    ds = FusionAndTrackingDataset(sensor_data_folder_path, Noise=False)
    dataloader = DataLoader(
        dataset=ds, batch_size=batchSize, collate_fn=collate_fn, shuffle=True
    )

    return dataloader


# Custom data transformation function to add white noise
class AddWhiteNoise(object):
    def __init__(self, noise_std=0.1):
        self.noise_std = noise_std

    def __call__(self, sample):

        # Add white noise to the first 4 parameters of the inputs
        noisy_inputs = sample.clone()
        for i in range(noisy_inputs.shape[0]):
            if torch.any(noisy_inputs[i, :4] != 0):
                noisy_inputs[i, :4] += (
                    torch.randn_like(noisy_inputs[i, :4]) * self.noise_std
                )

        return noisy_inputs


def create_trainVal_dataloader(datasetPath, batchSize):
    # path to datasets
    # absolute_path = os.path.dirname(__file__)
    sensor_data_folder_path = os.path.normpath(datasetPath)

    train_data_paths_file = os.path.join(sensor_data_folder_path, "train.p")
    val_data_paths_file = os.path.join(sensor_data_folder_path, "val.p")

    train_dataset = FusionAndTrackingDataset(train_data_paths_file, Noise=False)
    val_dataset = FusionAndTrackingDataset(val_data_paths_file, Noise=False)

    # ds = FusionAndTrackingDataset(sensor_data_folder_path, Noise=False)

    # Define the train-val split ratio
    # train_ratio = 0.7
    # dataset_size = len(ds)
    # train_size = int(train_ratio * dataset_size)
    # val_size = dataset_size - train_size

    # Perform the train-val split
    # train_dataset, val_dataset = random_split(ds, [train_size, val_size])
    # train_dataloader = DataLoader(dataset=train_dataset, batch_size=batchSize, collate_fn=collate_fn, shuffle=False, drop_last=True)
    # val_dataloader = DataLoader(dataset=val_dataset, batch_size=batchSize, collate_fn=collate_fn, shuffle=False, drop_last=True)

    # train_dataset = Subset(ds, range(612339))
    # val_dataset = Subset(ds, range(612339, dataset_size))
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batchSize,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batchSize,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=True,
    )

    return train_dataloader, val_dataloader


def create_testloader(datasetPath, batchSize):
    # path to datasets
    sensor_data_folder_path = os.path.normpath(datasetPath)

    test_data_paths_file = os.path.join(sensor_data_folder_path, "test.p")
    test_dataset = FusionAndTrackingDataset(test_data_paths_file, Noise=False)

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batchSize,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
    )

    return test_dataloader
