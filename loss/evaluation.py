import torch


def calculate_iou_matrix(bboxes1, bboxes2, iou_threshold=0.5):
    iou_matrix = torch.zeros(
        (bboxes1.shape[0], bboxes2.shape[0]), device=bboxes1.device
    )
    for i, bbox1 in enumerate(bboxes1):
        for j, bbox2 in enumerate(bboxes2):
            iou = calculate_iou(bbox1, bbox2)
            iou_matrix[i, j] = iou

    return iou_matrix


def evaluate_fusion(
    gt_bboxes, fused_bboxes, gt_classes, fused_classes, iou_threshold=0.5
):
    iou_matrix = calculate_iou_matrix(
        fused_bboxes, gt_bboxes, iou_threshold=iou_threshold
    )

    assignments = do_assignment(fused_bboxes.shape[0], iou_matrix)

    # calculate misses
    misses = torch.sum(assignments[:, 1] == -1)

    # calculate false positives
    false_positives = torch.sum(assignments[:, 0] == -1)

    # calculate true positives
    true_positives = gt_bboxes.shape[0] - misses

    # extraxt ious of matched objects
    assignments_mask = torch.all(assignments != -1, axis=-1)
    true_assignments = assignments[assignments_mask]
    assigned_ious = iou_matrix[true_assignments[:, 0], true_assignments[:, 1]]

    # compare classes of matched objects
    assigned_gt_classes = gt_classes[true_assignments[:, 0]]
    assigned_fused_classes = fused_classes[true_assignments[:, 1]]

    true_classified = torch.sum(assigned_gt_classes == assigned_fused_classes)
    false_classified = torch.sum(assigned_gt_classes != assigned_fused_classes)

    return (
        true_positives,
        false_positives,
        misses,
        assigned_ious,
        true_classified,
        false_classified,
    )
