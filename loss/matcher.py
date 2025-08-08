import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from loss.box_util import compute_giou, convert_to_x1y1x2y2
from einops import rearrange, repeat

import cv2
import numpy as np


class HungarianMatcher(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def remove_padded_objects(self, target_bbox, target_cls, target_mot, target_exists):

        # Convert target_exists to boolean mask
        target_mask = target_exists.squeeze(dim=-1).bool()

        # Use the boolean mask to filter out non-existent objects
        valid_target_bbox = target_bbox[target_mask]
        valid_target_cls = target_cls[target_mask]
        valid_target_mot = target_mot[target_mask]

        return valid_target_bbox, valid_target_cls, valid_target_mot

    def forward(self, outputs, targets):
        batch_size, num_queries = outputs["BBox"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_bbox = rearrange(
            outputs["BBox"], "B Q C -> (B Q) C", B=batch_size, Q=num_queries
        )
        outBbox = out_bbox[:, :4].to(torch.float32)
        out_class = rearrange(
            outputs["Object_class"], "B Q C -> (B Q) C", B=batch_size, Q=num_queries
        ).softmax(-1)
        # out_motion  = rearrange(outputs["Motion_params"], 'B Q C -> (B Q) C')

        # targetBBox, targetClass, targetMot = self.remove_padded_objects(target_bbox, target_class,
        #                                                             target_motion, target_exist)
        # targetClass = targetClass.int().to('cuda')
        # targetBBox = targetBBox.to('cuda')
        # targetMot = targetMot.to('cuda')

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["BBox"] for v in targets])
        # tgt_mot = torch.cat([v["mot"] for v in targets])

        targetClass = rearrange(tgt_ids, "T C -> (T C)").int().to("cuda")
        targetBBox = tgt_bbox.to("cuda").to(torch.float32)
        # targetMot = tgt_mot.to('cuda')

        # Compute the classification cost.
        cost_class = -out_class[:, targetClass]

        # Compute the L1 cost between boxes
        # cost_bbox = torch.cdist(out_bbox[:batch_size*num_targets,:], targetBBox, p=1)
        cost_bbox = torch.cdist(outBbox, targetBBox, p=1)

        # Compute the giou cost betwen boxes
        # src_boxesNorm = out_bbox.sigmoid()
        # target_boxesNorm = targetBBox.sigmoid()
        # cost_giou = -compute_giou(src_boxesNorm, target_boxesNorm)
        cost_giou = -compute_giou(outBbox, targetBBox)

        # Compute motion parameter loss
        # cost_mot = self.mse(out_motion[:batch_size*num_targets, :], targetMot)
        # cost_mot = torch.cdist(out_motion, targetMot, p=1)

        # Final cost matrix
        C = (
            self.weights["giou"] * cost_giou
            + self.weights["bbox"] * cost_bbox
            + self.weights["ce"] * cost_class
        )  # + cost_mot
        C = C.view(
            batch_size, num_queries, -1
        ).cpu()  # push to cpu for optimizing, remove grad and convert to numpy array

        # Apply the Hungarian algorithm
        # sizes =  [torch.count_nonzero(target_exist[i]) for i in range(target_exist.shape[0])]
        # sizes =  [len(v) for v in target_class]
        sizes = [len(v["BBox"]) for v in targets]
        indices = [
            linear_sum_assignment(c[i].detach().numpy())
            for i, c in enumerate(C.split(sizes, -1))
        ]

        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]
