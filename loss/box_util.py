"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_iou, generalized_box_iou, box_convert


# def convert_to_x1y1x2y2(bbox):
#     center_x, center_y, length, width = bbox.unbind(-1)
#     x1 = center_x - length / 2
#     y1 = center_y - width / 2
#     x2 = center_x + length / 2
#     y2 = center_y + width / 2
#     return torch.stack((x1, y1, x2, y2), dim=-1)


def convert_to_x1y1x2y2(bbox):
    b = box_convert(bbox, "cxcywh", "xyxy")
    return b


def compute_giou(box1, box2):
    box1 = convert_to_x1y1x2y2(box1)
    box2 = convert_to_x1y1x2y2(box2)
    # Calculate the GIoU
    # iou = box_iou(box1, box2)
    giou = generalized_box_iou(box1, box2)
    return giou


def compute_iou(box1, box2):
    box1 = convert_to_x1y1x2y2(box1)
    box2 = convert_to_x1y1x2y2(box2)
    # Calculate the GIoU
    # iou = box_iou(box1, box2)
    iou = box_iou(box1, box2)
    return iou
