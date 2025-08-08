import numpy as np
import torch
from shapely.geometry import Polygon
from scipy.optimize import linear_sum_assignment


def bbox2poly(gt_bbox):
    center_x, center_y, length, width, hdg = gt_bbox
    sin_hdg = np.sin(hdg)
    cos_hdg = np.cos(hdg)

    x1 = center_x + length / 2 * cos_hdg - width / 2 * sin_hdg
    y1 = center_y + length / 2 * sin_hdg + width / 2 * cos_hdg
    x2 = center_x - length / 2 * cos_hdg - width / 2 * sin_hdg
    y2 = center_y - length / 2 * sin_hdg + width / 2 * cos_hdg
    x3 = center_x - length / 2 * cos_hdg + width / 2 * sin_hdg
    y3 = center_y - length / 2 * sin_hdg - width / 2 * cos_hdg
    x4 = center_x + length / 2 * cos_hdg + width / 2 * sin_hdg
    y4 = center_y + length / 2 * sin_hdg - width / 2 * cos_hdg

    return Polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])


def calculate_iou_matrix(fused_bboxes, gt_bboxes, iou_threshold=0.5):
    # build iou matrix using gt_bboxes and fused_bbox with shapely.Geometry.Polygon
    iou_matrix = np.zeros((gt_bboxes.shape[0], fused_bboxes.shape[0]))
    for i, gt_bbox in enumerate(gt_bboxes):
        gt_poly = bbox2poly(gt_bbox)
        for j, fused_bbox in enumerate(fused_bboxes):
            fused_poly = bbox2poly(fused_bbox)
            iou_matrix[i, j] = (
                gt_poly.intersection(fused_poly).area / gt_poly.union(fused_poly).area
            )
    # append columns with threshold iou values
    th_iou = iou_threshold * np.eye(gt_bboxes.shape[0], gt_bboxes.shape[0])
    iou_matrix = np.hstack((iou_matrix, th_iou))
    return iou_matrix


def do_assignment(num_fused_objs, iou_matrix):
    assignments = linear_sum_assignment(iou_matrix, maximize=True)
    assignments = np.stack(assignments, axis=1)
    unassigned_mask = assignments[:, 1] > num_fused_objs - 1
    assignments[unassigned_mask, 1] = -1

    for i in range(num_fused_objs):
        if i not in assignments[:, 1]:
            assignments = np.append(assignments, [[-1, i]], axis=0)

    return assignments


def evaluate_fusion(output, target, iou_threshold=0.5):
    batch_size = len(target)
    fused_bboxes_t = output["BBox"]
    fused_bboxes_B = fused_bboxes_t.cpu().numpy()
    fused_classes_t = torch.argmax(
        torch.softmax(output["Object_class"], dim=-1), dim=-1
    )
    fused_classes_B = fused_classes_t.cpu().numpy()

    sum_true_positives = 0
    sum_false_positives = 0
    sum_false_negatives = 0
    sum_assigned_ious = 0
    sum_true_classified = 0
    sum_false_classified = 0

    for idx in np.arange(batch_size):
        gt_bboxes_t = torch.cat(
            (target[idx]["BBox"], target[idx]["mot"][:, 1:2]), dim=-1
        )
        gt_classes_t = target[idx]["labels"].squeeze(-1)
        gt_bboxes = gt_bboxes_t.cpu().numpy()
        gt_classes = gt_classes_t.cpu().numpy()

        fused_bboxes = fused_bboxes_B[idx, ...]
        fused_classes = fused_classes_B[idx, ...]
        is_obj_mask = fused_classes != 5
        fused_obj_bboxes = fused_bboxes[is_obj_mask]
        fused_obj_classes = fused_classes[is_obj_mask]

        iou_matrix = calculate_iou_matrix(
            fused_obj_bboxes, gt_bboxes, iou_threshold=iou_threshold
        )

        assignments = do_assignment(fused_obj_bboxes.shape[0], iou_matrix)

        # calculate misses
        false_negatives = np.sum(assignments[:, 1] == -1)

        # calculate false positives
        # fused_classes_extended = np.append(fused_classes, 5)
        # assignment_is_obj_mask = fused_classes_extended[assignments[:, 1]] != 5
        false_positives = np.sum(assignments[:, 0] == -1)

        # calculate true positives
        true_positives = gt_bboxes.shape[0] - false_negatives

        # extraxt ious of matched objects
        assignments_mask = np.all(assignments != -1, axis=-1)
        true_assignments = assignments[assignments_mask]
        assigned_ious = iou_matrix[true_assignments[:, 0], true_assignments[:, 1]]

        # compare classes of matched objects
        assigned_gt_classes = gt_classes[true_assignments[:, 0]]
        assigned_fused_classes = fused_obj_classes[true_assignments[:, 1]]

        true_classified = np.sum(assigned_gt_classes == assigned_fused_classes)
        false_classified = np.sum(assigned_gt_classes != assigned_fused_classes)

        sum_true_positives += true_positives
        sum_false_positives += false_positives
        sum_false_negatives += false_negatives
        sum_assigned_ious += np.sum(assigned_ious)
        sum_true_classified += true_classified
        sum_false_classified += false_classified

    # mean_iou = sum_assigned_ious / sum_true_positives
    # precision = sum_true_positives / (sum_true_positives+sum_false_positives)
    # recall = sum_true_positives / (sum_true_positives+sum_false_negatives)
    # cls_precision = sum_true_classified / (sum_true_classified+sum_false_classified)

    return (
        sum_true_positives,
        sum_false_positives,
        sum_false_negatives,
        sum_assigned_ious,
        sum_true_classified,
        sum_false_classified,
    )
