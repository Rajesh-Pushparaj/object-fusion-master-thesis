import numpy as np
import cv2
import os

from dataset.utils import data_utils

# Define the dimensions of the empty image
image_width = 1280  # Adjust as needed
image_height = 720  # Adjust as needed

# Create an empty black image
road_image = np.ones((image_height, image_width, 3), dtype=np.uint8)
# road_image[:, :, 0] = 225
# road_image[:, :, 1] = 225
# road_image[:, :, 2] = 225
# 229, 228, 226
# Draw lane markings (example)
# lane_color = (0, 255, 0)  # Green color for lane markings
# lane_thickness = 5  # Adjust as needed

# Draw left lane
# cv2.line(road_image, (100, image_height), (500, 600), lane_color, lane_thickness)

# # Draw right lane
# cv2.line(road_image, (image_width - 100, image_height), (image_width - 500, 600), lane_color, lane_thickness)

# Draw bounding boxes
ego_color = (0, 0, 255)
agent_color_gt = (0, 255, 0)  # Green color for GT agent bounding boxes
agent_color_pred = (225, 172, 28)  # Blue color for predicted agent bounding boxes
agent_color_input = (100, 149, 237)  # red color for input agent bounding boxes
agent_thickness = 2  # Adjust as needed

# Define the range of coordinates for x and y
x_range = (
    data_utils.xRange[0],
    data_utils.xRange[1],
)  # (-60, 120)   # came up with inspecting the bbox positions, may vary
y_range = (data_utils.yRange[0], data_utils.yRange[1])  # (-30, 30)

# Calculate scaling factors for x and y
x_scale = image_width / (x_range[1] - x_range[0])
y_scale = image_height / (y_range[1] - y_range[0])

# path to data files
dataPath = "runViz"
absolute_path = os.path.dirname(__file__)
data_folder_path = os.path.join(absolute_path, dataPath)


def visualize(exportVideo=False):
    num_files = len(os.listdir(data_folder_path))
    # Create a list of frames to display
    frames = []
    for i in range(num_files):
        # Create a copy of the empty road image for each frame
        frame = road_image.copy()

        # ego vehicle
        # Define the size and color of the rectangle
        ego_cx = 0
        ego_cy = 0
        ego_l = 4.5
        ego_w = 1.5
        # Scale the coordinates to fit the visualization resolution
        scaled_center_x = int((ego_cx - x_range[0]) * x_scale)
        scaled_center_y = int((ego_cy - y_range[0]) * y_scale)
        scaled_length = int(ego_l * x_scale)
        scaled_width = int(ego_w * y_scale)
        # Calculate top-left and bottom-right coordinates
        top_left = (
            int(scaled_center_x - scaled_length / 2),
            int(scaled_center_y - scaled_width / 2),
        )
        bottom_right = (
            int(scaled_center_x + scaled_length / 2),
            int(scaled_center_y + scaled_width / 2),
        )
        cv2.rectangle(frame, top_left, bottom_right, ego_color, agent_thickness)

        file_path = os.path.join(data_folder_path, f"output_{i}.npy")
        data = np.load(file_path, allow_pickle=True).item()

        targBox = data["target_bbox"].to("cpu")
        targBox = data_utils.denormalizeBBox(targBox)  # de-normalize bbox
        # targMot = data["target_motion"].to('cpu')
        targCls = data["target_class"].to("cpu")

        Box = data["BBox"][:, :4].to("cpu")
        Box = data_utils.denormalizeBBox(Box)  # de-normalize bbox
        # Mot = data["Motion_params"].to('cpu')
        Cls = data["Object_class"].to("cpu")

        if "input_bbox" in data:
            inpBox = data["input_bbox"].to("cpu")
            inpBox = data_utils.denormalizeBBox(inpBox)  # de-normalize bbox
            inpCls = data["input_class"].to("cpu")

        # compute class lables
        prob = Cls.softmax(-1)
        scores, labels = prob.max(-1)

        for i in range(Box.shape[0]):
            box = np.asarray(Box[i])
            if labels[i] != 5:  # check if the lables match
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
                    frame, top_left, bottom_right, agent_color_pred, agent_thickness
                )

                # Define the label text
                label = f"{labels[i]}"
                # Determine the position to place the label (above the top-left corner of the bounding box)
                label_position = (
                    scaled_center_x - 10,
                    scaled_center_y,
                )  # Adjust -10 as needed to position the label correctly
                # Add the label to the image
                cv2.putText(
                    frame,
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
            cv2.rectangle(
                frame, top_left, bottom_right, agent_color_gt, agent_thickness
            )

            # Define the label text
            label = f"{targCls[i].int().item()}"
            # Determine the position to place the label (above the top-left corner of the bounding box)
            label_position = (
                scaled_center_x - 10,
                scaled_center_y,
            )  # Adjust -10 as needed to position the label correctly
            # Add the label to the image
            cv2.putText(
                frame,
                label,
                label_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                agent_color_gt,
                2,
            )

        if "input_bbox" in data:
            for i in range(inpBox.shape[0]):
                box = np.asarray(inpBox[i])
                # if labels[i] != 13:     # check if the lables match
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
                    frame, top_left, bottom_right, agent_color_input, agent_thickness
                )

                # Define the label text
                label = f"{inpCls[i].int().item()}"
                # Determine the position to place the label (above the top-left corner of the bounding box)
                label_position = (
                    scaled_center_x - 10,
                    scaled_center_y - 10,
                )  # Adjust -10 as needed to position the label correctly
                # Add the label to the image
                cv2.putText(
                    frame,
                    label,
                    label_position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    agent_color_input,
                    2,
                )

        frames.append(frame)

    if exportVideo:
        codec = cv2.VideoWriter_fourcc(*"mp4v")  # Use 'mp4v' codec for MP4 format
        # Create a VideoWriter object
        out = cv2.VideoWriter(
            "../output.mp4", codec, 12, (image_width, image_height)
        )  # Use .mp4 file extension
        # Loop through your frames and add them to the video writer
        for (
            frame
        ) in (
            frames
        ):  # Assuming frame_list is a list of numpy arrays representing your frames
            out.write(frame)
        # Release the VideoWriter and close the output file
        out.release()

    # Display frames in sequence
    for frame in frames:
        cv2.imshow("Sensor fusion Visualization", frame)
        cv2.waitKey(0)  # Adjust the delay (in milliseconds) between frames as needed

    # cv2.destroyAllWindows()


if __name__ == "__main__":
    visualize()
