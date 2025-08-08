import cv2
import os
import numpy as np
import torch.nn.functional as F
from einops import rearrange, repeat

# Define the dimensions of the empty image
image_width = 1280  # Adjust as needed
image_height = 720  # Adjust as needed

# Create an empty black image
road_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

# Draw bounding boxes
agent_color_gt = (0, 255, 0)  # Green color for GT agent bounding boxes
agent_color_pred = (255, 0, 0)  # Blue color for predicted agent bounding boxes
agent_thickness = 2  # Adjust as needed

# Define the range of coordinates for x and y
x_range = (-60, 120)  # came up with inspecting the bbox positions, may vary
y_range = (-30, 30)

# Calculate scaling factors for x and y
x_scale = image_width / (x_range[1] - x_range[0])
y_scale = image_height / (y_range[1] - y_range[0])

# path to data files
dataPath = "results_overfit"
absolute_path = os.path.dirname(__file__)
data_folder_path = os.path.join(absolute_path, dataPath)

num_files = len(os.listdir(data_folder_path))

file_path = os.path.join(data_folder_path, f"output_0.npy")
data = np.load(file_path, allow_pickle=True).item()
targBox = data["target_bbox"].to("cpu")
# targMot = data["target_motion"].to('cpu')
targCls = data["target_class"].to("cpu")

Box = data["BBox"].to("cpu")
# Mot = data["Motion_params"].to('cpu')
Cls = data["Object_class"].to("cpu")
# compute class lables
prob = Cls.softmax(-1)
scores, labels = prob.max(-1)

print(labels, targCls)
# Create a boolean tensor where True indicates matching labels
# matching_indices = (labels.to('cpu') == targCls.squeeze(1).to(int))

for i in range(Box.shape[0]):
    box = np.asarray(Box[i])
    if labels[i] != 13:  # check if the lables match
        center_x = box[0]
        center_y = box[1]
        length = box[2]
        width = box[3]
        # Scale the coordinates to fit the visualization resolution
        scaled_center_x = ((center_x - x_range[0]) * x_scale).astype(int)
        scaled_center_y = ((center_y - y_range[0]) * y_scale).astype(int)
        scaled_length = (length * x_scale).astype(int)
        scaled_width = (width * y_scale).astype(int)
        # Calculate top-left and bottom-right coordinates
        top_left = (
            (scaled_center_x - scaled_length / 2).astype(int),
            (scaled_center_y - scaled_width / 2).astype(int),
        )
        bottom_right = (
            (scaled_center_x + scaled_length / 2).astype(int),
            (scaled_center_y + scaled_width / 2).astype(int),
        )
        cv2.rectangle(
            road_image, top_left, bottom_right, agent_color_pred, agent_thickness
        )

        # Define the label text
        label = f"{labels[i]}"
        # Determine the position to place the label (above the top-left corner of the bounding box)
        label_position = (
            scaled_center_x,
            scaled_center_y,
        )  # Adjust -10 as needed to position the label correctly
        # Add the label to the image
        cv2.putText(
            road_image,
            label,
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            agent_color_pred,
            2,
        )

for i in range(targBox.shape[0]):
    box = np.asarray(targBox[i])
    center_x = box[0]
    center_y = box[1]
    length = box[2]
    width = box[3]
    # Scale the coordinates to fit the visualization resolution
    scaled_center_x = ((center_x - x_range[0]) * x_scale).astype(int)
    scaled_center_y = ((center_y - y_range[0]) * y_scale).astype(int)
    scaled_length = (length * x_scale).astype(int)
    scaled_width = (width * y_scale).astype(int)
    # Calculate top-left and bottom-right coordinates
    top_left = (
        (scaled_center_x - scaled_length / 2).astype(int),
        (scaled_center_y - scaled_width / 2).astype(int),
    )
    bottom_right = (
        (scaled_center_x + scaled_length / 2).astype(int),
        (scaled_center_y + scaled_width / 2).astype(int),
    )
    cv2.rectangle(road_image, top_left, bottom_right, agent_color_gt, agent_thickness)

    # Define the label text
    label = f"{targCls[i].int().item()}"
    # Determine the position to place the label (above the top-left corner of the bounding box)
    label_position = (
        scaled_center_x,
        scaled_center_y,
    )  # Adjust -10 as needed to position the label correctly
    # Add the label to the image
    cv2.putText(
        road_image,
        label,
        label_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        agent_color_gt,
        2,
    )


# Display frames in sequence
cv2.imshow("Sensor fusion Visualization", road_image)
cv2.waitKey(0)  # Adjust the delay (in milliseconds) between frames as needed
