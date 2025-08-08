import numpy as np
import os
from pathlib import Path
import hydra
from dataset.utils import data_utils
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
matplotlib.use('TkAgg')


sensor2color = {
    'CAMERA_FC0': (0.95, 0.87, 0.87), # (241, 222, 222)
    'RADAR_FL0': (0.34, 0.29, 0.89), #(87, 74, 226),
    'RADAR_FR0': (1, 0.58, 0.55), #(254, 147, 140),
    'RADAR_RL0': (0.83, 0.59, 0.65), #(212, 150, 167),
    'RADAR_RR0': (0.42, 0.83, 1), #(108, 212, 255),
    'LIDAR_GT': (0, 0.69, 0.15), #(0, 176, 38)
    'HLFTF': (0.27, 0.51, 0.71),
    'AKFA': (1, 0.84, 0)
}


@hydra.main(config_path="config/plot", config_name="plot", version_base="1.3")
def main(cfg):
    # path to data files
    AKFA_dataPath = cfg['AKFA']
    HLFTF_dataPath = cfg['HLFTF']
    absolute_path = os.path.dirname(__file__)
    AKFA_data_folder_path =  os.path.join(absolute_path, AKFA_dataPath)
    HLFTF_data_folder_path =  os.path.join(absolute_path, AKFA_dataPath)
    
    absolute_path = os.path.dirname(__file__)
    plots_path = cfg['plots_path']
    Path(plots_path).mkdir(parents=True, exist_ok=True)

    num_files = len(os.listdir(AKFA_data_folder_path))

    alphaGT = cfg['alphaGT']
    alphaFO_HLFTF = cfg['alphaFO_HLFTF']
    alphaFO_AFKA = cfg['alphaFO_AFKA']
    alphaDet = cfg['alphaDet']

    for file_num in range(num_files):
        # file paths
        file_path = os.path.join(HLFTF_dataPath, f'output_{file_num}.npy')
        data = np.load(file_path, allow_pickle=True).item()
        file_path_akfa = os.path.join(AKFA_dataPath, f'output_{file_num}.npy')
        data_akfa = np.load(file_path_akfa, allow_pickle=True).item()


        targBox = data["target_bbox"].to('cpu')
        targBox = data_utils.denormalizeBBox(targBox) #de-normalize bbox
        # targMot = data["target_motion"].to('cpu')
        targCls = data["target_class"].to('cpu')

        # HLFTF data
        Box = data["BBox"][:,:4].to('cpu')
        Box = data_utils.denormalizeBBox(Box) #de-normalize bbox
        # Mot = data["Motion_params"].to('cpu')
        Cls = data["Object_class"].to('cpu')
        if "input_bbox" in data:
            inpBox = data["input_bbox"].to('cpu')
            inpBox = data_utils.denormalizeBBox(inpBox) #de-normalize bbox
            inpCls = data["input_class"].to('cpu')
        # compute class lables
        prob = Cls.softmax(-1)
        scores, labels = prob.max(-1)

        
        #AKFA data
        Box_akfa = data_akfa["output_bbox"][:,:4]
        Cls_akfa = data_akfa["output_class"]
        label_map = {9:0, 7:1, 10:2, 11:3, 12:4}
        Cls_akfa = [label_map[value] for value in Cls_akfa]
        

        # plt.ion()  # Turn on interactive mode
        # Create a figure and axis
        fig, ax = plt.subplots()
            # Set background color
        ax.set_facecolor(cfg['facecolor'])
        # Set axis limits
        ax.set_xlim(cfg['x_min'], cfg['x_max'])
        ax.set_ylim(cfg['y_min'], cfg['y_max'])

        # Choose specific labels to include in the legend
        selected_labels = {'Ego Vehicle': ['tomato', 'solid', 0, 1, None], 
                            'AKFA': [sensor2color['AKFA'], 'solid', 1, alphaFO_AFKA, None],
                            'HLFTF': [sensor2color['HLFTF'], 'solid', 1, alphaFO_HLFTF, None],
                            'Ground Truth': [sensor2color['LIDAR_GT'], 'solid', 1, alphaGT, None],
                            # 'Sensor Detection': ['darkorange', '--', 0, alphaDet, None]
                            'Camera Det.': [sensor2color['CAMERA_FC0'], '--', 0, alphaDet, None],
                            'Radar 1 Det.': [sensor2color['RADAR_FL0'], '--', 0, alphaDet, None],
                            'Radar 2 Det.': [sensor2color['RADAR_FR0'], '--', 0, alphaDet, None],
                            'Radar 3 Det.': [sensor2color['RADAR_RL0'], '--', 0, alphaDet, None],
                            'Radar 4 Det.': [sensor2color['RADAR_RR0'], '--', 0, alphaDet, None]
        }

        # Create legend entries only for the selected labels
        legend_handles = [patches.Rectangle((0, 0), 1, 1, color=selected_labels[label][0], edgecolor=selected_labels[label][0], linestyle=selected_labels[label][1], 
                                            fill=selected_labels[label][2], linewidth=1, label=label, alpha=selected_labels[label][3], hatch=selected_labels[label][4]) for label in selected_labels]
        
        if cfg['legends']:
            # Add the legend with the selected entries
            loc = cfg['legend_loc']
            ax.legend(handles=legend_handles, loc=loc, ncol=3, fontsize=10)
        if cfg['labels']:
            plt.xlabel(' Longitudinal Axis [m]', fontsize=12)
            plt.ylabel(' Lateral Axis [m]', fontsize=12)

        ego_l = cfg['ego_l']
        ego_w = cfg['ego_w']
        ego_cx = cfg['ego_cx'] - ego_l / 2
        ego_cy = cfg['ego_cy'] - ego_w / 2
        # Create a rectangle patch
        # plt.scatter(0,0,marker='x', color='tomato')
        rectangle = patches.Rectangle((ego_cx, ego_cy), ego_l, ego_w, linewidth=1.5, edgecolor='tomato', facecolor='none', label='Ego Vehicle')
        # Calculate arrow coordinates
        arrow_start = (0 , 0)
        arrow_end = ((ego_l/2), 0 )
        
        if cfg['direction_arrow']:
            # Create an arrow patch
            arrow = patches.FancyArrowPatch(arrow_start, arrow_end, mutation_scale=10, color='tomato', label='Ego Vehicle')
            ax.add_patch(arrow)
        # Add the rectangle patch to the plot
        ax.add_patch(rectangle)

        for i in range(Box_akfa.shape[0]):
            box = np.asarray(Box[i])
            if labels[i] != 5:     # check if the lables match
                center_x = box[0]
                center_y = box[1]
                length = box[2]
                width = box[3]
                # Calculate the center coordinates
                x = center_x - length / 2
                y = center_y - width / 2
                # Create a rectangle patch
                rectangle = patches.Rectangle((x, y), length, width, linewidth=1, color=sensor2color['AKFA'], fill=True, alpha=alphaFO_AFKA, label='Fused Object')
                if i == 2:
                    continue
                # Add the rectangle patch to the plot
                ax.add_patch(rectangle)

        for i in range(Box.shape[0]):
            box = np.asarray(Box[i])
            if labels[i] != 5:     # check if the lables match
                center_x = box[0]
                center_y = box[1]
                length = box[2]
                width = box[3]
                # Calculate the center coordinates
                x = center_x - length / 2
                y = center_y - width / 2
                # Create a rectangle patch
                rectangle = patches.Rectangle((x, y), length, width, linewidth=1, color=sensor2color['HLFTF'], fill=True, alpha=alphaFO_HLFTF, label='Fused Object')
                if i == 2:
                    continue
                # Add the rectangle patch to the plot
                ax.add_patch(rectangle)

        for i in range(targBox.shape[0]):
            box = np.asarray(targBox[i])
            center_x = box[0]
            center_y = box[1]
            length = box[2]
            width = box[3]
            # Calculate the center coordinates
            x = center_x - length / 2
            y = center_y - width / 2
            # Create a rectangle patch
            rectangle = patches.Rectangle((x, y), length, width, linewidth=1, color= sensor2color['LIDAR_GT'], fill=True, alpha=alphaGT, label='Ground Truth') #'dimgrey', hatch='XX'
            # Add the rectangle patch to the plot
            ax.add_patch(rectangle)

        if "input_bbox" in data:
            for i in range(inpBox.shape[0]):

                if i < 10:
                    edgColor = sensor2color['CAMERA_FC0']
                if i >= 10 and i < 20:
                    edgColor = sensor2color['RADAR_FL0']
                if i >= 20 and i < 30:
                    edgColor = sensor2color['RADAR_FR0']
                if i >= 30 and i < 40:
                    edgColor = sensor2color['RADAR_RL0']
                if i >= 40 and i < 50:
                    edgColor = sensor2color['RADAR_RR0']

                box = np.asarray(inpBox[i])
                # if labels[i] != 13:     # check if the lables match
                center_x = box[0]
                center_y = box[1]
                length = box[2]
                width = box[3]
                x = center_x - length / 2   
                y = center_y - width / 2
                rectangle = patches.Rectangle((x, y), length, width, linewidth=1.5, edgecolor=edgColor, facecolor='none', linestyle='--', alpha = alphaDet)
                # # Add the rectangle patch to the plot
                ax.add_patch(rectangle)

        ax.set_aspect('equal', adjustable='box')
        file_name = f'output_{file_num}.svg'
        plt.savefig(os.path.join(plots_path, file_name), bbox_inches='tight', transparent=False)
        if cfg['show_plots']:
            plt.show()

if __name__ == "__main__":
    main()