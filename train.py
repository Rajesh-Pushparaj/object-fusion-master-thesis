import torch
torch.manual_seed(421337)

import numpy as np
np.random.seed(133742)

from pathlib import Path

import hydra
from omegaconf import OmegaConf
import wandb
import torch.optim as optim

from dataset.data_loader import *
from model.model import create_model
from loss.loss import create_loss
from tqdm import tqdm
import os
from datetime import datetime
from einops import rearrange

from utils import evaluate_fusion


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg):
    best_val_loss = 10.0
    best_giou_and_bbox_loss = 10.0
    best_model_weights = None
    best_optim_metric = 0.0

    if cfg["wandb"]["enable"]:
        wandb.login(key=".....fill your key.....")
        wandb.init(
            entity="fusion_transformer",
            project="ITSC 2024",
            config=OmegaConf.to_object(cfg),
            group="training fusion model",
        )

    # log sin frequencies for ablation
    if cfg["wandb"]["enable"]:
        wandb.config.update(
            {
                "architecture": "Multi-modal Attention",
            }
        )
        wandb.config["embedding_frequencies"] = 2
    # create train dataloader
    datasetPath = cfg["dataset"]["dataset_path"]
    batch_size = cfg["dataset"]["batch_size"]
    # train_loader = create_dataloader(datasetPath, batch_size)
    train_loader, val_loader = create_trainVal_dataloader(datasetPath, batch_size)
    # Initialize your model, loss function, and optimizer
    model = create_model(cfg["model"])
    if cfg["wandb"]["enable"] and cfg["wandb"]["watch"]:
        wandb.watch(model, log="all", log_freq=250)
    model = model.to("cuda")

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)
    if cfg["wandb"]["enable"]:
        wandb.config["Num. of params"] = n_parameters
        wandb.config["train_data_size"] = len(train_loader) * batch_size
    criterion = create_loss(cfg)  # loss criterion
    criterion.to("cuda")
    optimizer = optim.Adam(
        model.parameters(), lr=cfg["training"]["learning_rate"]
    )  # , weight_decay=0.0001)

    print("Start training")
    # Training loop
    num_epochs = cfg["training"]["epochs"]
    train_samples = len(train_loader)
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        model.train()
        criterion.train()

        running_loss = 0.0
        running_loss_ce = 0.0
        running_loss_bbox = 0.0
        running_loss_giou = 0.0
        running_loss_head = 0.0

        for batch_idx, (inputs, targets) in enumerate(
            tqdm(train_loader, leave=False, desc="Train Batches")
        ):
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            loss, loss_ce, loss_bbox, loss_giou, loss_head = criterion(outputs, targets)

            # Backpropagation and optimization
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            running_loss += loss.item()
            running_loss_ce += loss_ce.item()
            running_loss_bbox += loss_bbox.item()
            running_loss_giou += loss_giou.item()
            running_loss_head += loss_head.item()

            if cfg["wandb"]["enable"] and batch_idx % 100 == 0:
                wandb.log(
                    {
                        "train/loss_ce": loss_ce.item(),
                        "train/loss_bbox": loss_bbox.item(),
                        "train/loss_giou": loss_giou.item(),
                        "train/loss_heading": loss_head.item(),
                        "train/loss": loss.item(),
                        "step": batch_idx + epoch * len(train_loader),
                    }
                )

        avg_loss = running_loss / (len(train_loader))
        avg_loss_ce = running_loss_ce / (len(train_loader))
        avg_loss_bbox = running_loss_bbox / (len(train_loader))
        avg_loss_giou = running_loss_giou / (len(train_loader))
        avg_loss_head = running_loss_head / (len(train_loader))
        print("", f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss}")
        # log metrics to wandb
        if cfg["wandb"]["enable"]:
            wandb.log(
                {
                    "train/avg_loss_ce": avg_loss_ce,
                    "train/avg_loss_bbox": avg_loss_bbox,
                    "train/avg_loss_giou": avg_loss_giou,
                    "train/avg_loss_heading": avg_loss_head,
                    "train/avg_loss": avg_loss,
                    "epoch": epoch + 1,
                }
            )

        # Test loop
        model.eval()  # Set the model to evaluation mode
        criterion.eval()
        running_val_loss = 0.0
        running_val_loss_ce = 0.0
        running_val_loss_bbox = 0.0
        running_val_loss_giou = 0.0
        running_val_loss_head = 0.0

        sum_true_positives = 0
        sum_false_positives = 0
        sum_false_negatives = 0
        sum_assigned_ious = 0
        sum_true_classified = 0
        sum_false_classified = 0

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(
                tqdm(val_loader, leave=False, desc="Validation Batches")
            ):
                # Forward pass
                outputs = model(inputs)
                _, loss_ce, loss_bbox, loss_giou, loss_head = criterion(
                    outputs, targets
                )

                val_loss = 1.5 * loss_ce + 100 * loss_bbox + loss_giou + loss_head

                running_val_loss += val_loss.item()
                running_val_loss_ce += loss_ce.item()
                running_val_loss_bbox += loss_bbox.item()
                running_val_loss_giou += loss_giou.item()
                running_val_loss_head += loss_head.item()

                (
                    true_positives,
                    false_positives,
                    false_negatives,
                    sum_assigned_ious_,
                    true_classified,
                    false_classified,
                ) = evaluate_fusion(outputs, targets)

                sum_true_positives += true_positives
                sum_false_positives += false_positives
                sum_false_negatives += false_negatives
                sum_assigned_ious += sum_assigned_ious_
                sum_true_classified += true_classified
                sum_false_classified += false_classified

            avg_val_loss = running_val_loss / (len(val_loader))
            avg_val_loss_ce = running_val_loss_ce / (len(val_loader))
            avg_val_loss_bbox = running_val_loss_bbox / (len(val_loader))
            avg_val_loss_giou = running_val_loss_giou / (len(val_loader))
            avg_val_loss_head = running_val_loss_head / (len(val_loader))

            mean_iou = sum_assigned_ious / sum_true_positives
            precision = sum_true_positives / (sum_true_positives + sum_false_positives)
            recall = sum_true_positives / (sum_true_positives + sum_false_negatives)
            f1_score = (
                2
                * sum_true_positives
                / (2 * sum_true_positives + sum_false_positives + sum_false_negatives)
            )
            cls_precision = sum_true_classified / (
                sum_true_classified + sum_false_classified
            )
            optim_metric = f1_score + mean_iou + cls_precision
            print("", f"val Loss: {avg_val_loss}")
            # log metrics to wandb
            if cfg["wandb"]["enable"]:
                wandb.log(
                    {
                        "optim_metric": optim_metric,
                        "val/avg_loss": avg_val_loss,
                        "val/loss_ce": avg_val_loss_ce,
                        "val/loss_bbox": avg_val_loss_bbox,
                        "val/loss_giou": avg_val_loss_giou,
                        "val/loss_heading": avg_val_loss_head,
                        "val/mIoU": mean_iou,
                        "val/det_precision": precision,
                        "val/det_recall": recall,
                        "val/det_f1": f1_score,
                        "val/cls_precision": cls_precision,
                        "epoch": epoch + 1,
                    }
                )
            # Check if the current model has the best validation accuracy
            if cfg["save_model"] and avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_weights = model.state_dict()
                best_model_optim_state = optimizer.state_dict()
                checkppoint_state = {
                    "model_state": best_model_weights,
                    "optimizer_state": best_model_optim_state,
                }
                save_path = os.path.join(
                    os.path.normpath(cfg["output_path"]),
                    str(wandb.run.id),
                    f"best_model_weights.pth",
                )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(checkppoint_state, save_path)

            if (
                cfg["save_model"]
                and avg_val_loss_bbox + avg_val_loss_giou < best_giou_and_bbox_loss
            ):
                best_giou_and_bbox_loss = avg_val_loss_bbox + avg_val_loss_giou
                best_bbox_model_weights = model.state_dict()
                best_bbox_model_optim_state = optimizer.state_dict()
                checkppoint_state = {
                    "model_state": best_bbox_model_weights,
                    "optimizer_state": best_bbox_model_optim_state,
                }
                save_path = os.path.join(
                    os.path.normpath(cfg["output_path"]),
                    str(wandb.run.id),
                    f"best_bbox_model_weights.pth",
                )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(checkppoint_state, save_path)

            if cfg["save_model"] and optim_metric > best_optim_metric:
                best_optim_metric = optim_metric
                best_model_weights = model.state_dict()
                best_model_optim_state = optimizer.state_dict()
                checkppoint_state = {
                    "model_state": best_model_weights,
                    "optimizer_state": best_model_optim_state,
                }
                save_path = os.path.join(
                    os.path.normpath(cfg["output_path"]),
                    str(wandb.run.id),
                    f"best_optim_metric_model_weights.pth",
                )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(checkppoint_state, save_path)

    # [optional] finish the wandb run, necessary in notebooks
    if cfg["wandb"]["enable"]:
        wandb.finish()
    print("Training finished")


if __name__ == "__main__":
    main()
