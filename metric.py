import numpy as np
import os

import torch
from loss.box_util import compute_iou

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")

# path to data files
dataPath = "runViz"
absolute_path = os.path.dirname(__file__)
data_folder_path = os.path.join(absolute_path, dataPath)
num_files = len(os.listdir(data_folder_path))
classes = [1, 2]
iou_thres = [0.5, 0.7, 0.9]

# get pred sorted with confidence score
def getScoreSorted(class_id):
    sortedPredGT = []
    totalGT = 0
    for i in range(num_files):
        file_path = os.path.join(data_folder_path, f"output_{i}.npy")
        data = np.load(file_path, allow_pickle=True).item()
        # target
        targBox = data["target_bbox"].to("cpu")
        targCls = data["target_class"].to("cpu")
        # predictions
        Box = data["BBox"][:, :4].to("cpu")
        Cls = data["Object_class"].to("cpu")
        # compute class lables
        prob = Cls.softmax(-1)
        scores, labels = prob.max(-1)

        mask1 = (targCls.int() == class_id).squeeze(1)
        targ = targBox[mask1]

        mask2 = labels == class_id
        pred = Box[mask2]
        scores = scores.unsqueeze(1)
        score = scores[mask2]

        for id in range(pred.shape[0]):
            sortedPredGT.append(
                {
                    "score": score[id],
                    "pred": pred[id],
                    "targ": targ,
                }
            )
            totalGT += targ.shape[0]
    sorted_data = sorted(sortedPredGT, key=lambda x: x["score"], reverse=True)

    return sorted_data, totalGT  # sortedPredGT, totalGT


# plot the precision-recall curve
def plot_pr_curve(precisions, recalls):
    # plots the precision recall values for each threshold
    # and save the graph to disk
    plt.plot(recalls, precisions, linewidth=4, color="red")
    plt.xlabel("Recall", fontsize=12, fontweight="bold")
    plt.ylabel("Precision", fontsize=12, fontweight="bold")
    plt.title("Precision-Recall Curve", fontsize=15, fontweight="bold")
    plt.savefig("../precision-recall.png")
    plt.show()


# compute precision and recalls for given iou threshold
def get_pr(iou_threshold):

    prec_rec = {}
    for cls in classes:

        TP = 0
        FP = 0
        precision = []
        recall = []

        sortedPred, totalGT = getScoreSorted(cls)

        if totalGT == 0:
            prec_rec[cls] = (None, None)
            continue

        for i in range(len(sortedPred)):
            # target
            targ = sortedPred[i]["targ"]
            # predictions
            pred = sortedPred[i]["pred"].unsqueeze(0)

            iou = compute_iou(pred, targ)

            if not targ.shape[0]:
                FP += pred.shape[0]
            else:
                # For each prediction
                for i in range(pred.shape[0]):
                    max_iou = iou[i, :].max()
                    # Check if the maximum IoU is above the threshold
                    if max_iou >= iou_threshold:
                        TP += 1  # Mark as True Positive
                    else:
                        FP += 1  # Mark as False Positive

            prec = TP / (TP + FP)
            rec = TP / totalGT

            precision.append(prec)
            recall.append(rec)

        prec_rec[cls] = (precision, recall)
        # plot precision-recall curve
        plot_pr_curve(precision, recall)
    return prec_rec


# get 11 point interpolation Average Precision
def get_AP11(prec_rec):
    AP11 = []
    ap11 = 0.0
    # 11 point interpolation AP
    recall_levels = np.linspace(0, 1, 11)  # 11 equally spaced recall levels

    for cls in classes:
        if prec_rec[cls][0]:
            precis = np.array(prec_rec[cls][0])
            if precis.size:
                recl = prec_rec[cls][1]
                # Iterate through the 11 recall levels
                for recall_level in recall_levels:
                    # Find the maximum precision value at or to the right of the current recall level
                    max_precision = 0
                    for precision, recall in zip(precis, recl):
                        if recall >= recall_level:
                            max_precision = max(max_precision, precision)
                    # Accumulate the maximum precision values
                    ap11 += max_precision
                ap11 /= 11
                AP11.append(ap11)
                continue
            AP11.append(0)
    return AP11


# get full interpolation Average Precision
def get_AP(pre_rec):
    # Initialize variables for full interpolated AP calculation
    AP = []
    ap = []

    # Full interpolation AP
    for cls in classes:
        if prec_rec[cls][0]:
            if len(prec_rec[cls][0]):
                # prec_rec[cls][0].insert(0,1)    # Add 1 to the precision set
                # precis = np.array(prec_rec[cls][0])
                # prec_rec[cls][1].insert(0,0)    # Add 0 to the recall set
                # recl = np.array(prec_rec[cls][1])
                # Sort precision and recall values by recall in descending order
                precision_recall_tuples = list(zip(prec_rec[cls][0], prec_rec[cls][1]))
                # sorted_tuples = sorted(precision_recall_tuples, key=lambda x: x[1], reverse=True)
                # total_precision = 0
                # total_recall_levels = len(sorted_tuples)

                # for i in range(1, total_recall_levels):
                #     recall, precision = sorted_tuples[i]
                #     prev_recall, prev_precision = sorted_tuples[i - 1]

                #     interpolated_precision = max(precision for r, precision in sorted_tuples[i:])
                #     total_precision += interpolated_precision * (recall - prev_recall)

                # ap = total_precision / total_recall_levels
                # rcl, pcs = zip(*sorted_pairs)
                # rcl = np.array(rcl)
                # pcs = np.array(pcs)
                # # Calculate the area under the precision-recall curve segment using trapezoidal rule
                # # difference between recall levels
                # delta_recall = np.diff(rcl)
                # # Calculate the area of each trapezoid
                # area_trapezoids = delta_recall * (pcs[:-1] + pcs[1:]) / 2.0
                # # Sum the areas to obtain AP (AUC)
                # ap = np.sum(area_trapezoids)
                ap = compute_average_precision(precision_recall_tuples)
                AP.append(ap)
                continue
            AP.append(0)
    return AP


def compute_average_precision(precision_recall_list):
    sorted_precision_recall = sorted(
        precision_recall_list, key=lambda x: x[1], reverse=False
    )

    average_precision = 0.0
    recall_previous = 0.0

    for precision, recall in sorted_precision_recall:
        # Calculate the area under the precision-recall curve
        average_precision += precision * (recall - recall_previous)
        # Update the previous recall value
        recall_previous = recall

    return average_precision


def compute_mAP(iouThresholds):

    mAP_11P = []
    mAP_full = []

    for iouThres in iouThresholds:
        prec_rec = get_pr(iouThres)

        AP11 = get_AP11(prec_rec)
        AP = get_AP(prec_rec)

        mAP11 = sum(AP11) / len(classes)
        mAP = sum(AP) / len(classes)

        mAP_11P.append(mAP11)
        mAP_full.append(mAP)

    return mAP_11P, mAP_full


# Detections from all frames has to be aligned by class scores and
# cumulative precision and recall should be computed.
# Plot precision-recall graph
# Compute average precision per class.
# Compute mean Average Precision.
if __name__ == "__main__":

    for iouThres in iou_thres:
        prec_rec = get_pr(iouThres)

        AP11 = get_AP11(prec_rec)
        AP = get_AP(prec_rec)

        mAP11 = sum(AP11) / len(classes)
        mAP = sum(AP) / len(classes)
        print(f"AP{int(iouThres*100)}")
        print(f"AP11: {AP11}\nmAP11: {mAP11}")
        print(f"AP: {AP}\nmAP: {mAP}")
        print("*********************************************")
