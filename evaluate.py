import hydra
from omegaconf import OmegaConf
from dataset.data_loader import *
from model.model import create_model

from pathlib import Path
from datetime import datetime
from einops import rearrange

from utils import evaluate_fusion

import pickle as p

# from metric import *
from tqdm import tqdm
import yaml


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg):
    # create train dataloader
    datasetPath = cfg["dataset"]["dataset_path"]
    batch_size = 1
    # train_loader = create_dataloader(datasetPath, batch_size)
    test_loader = create_testloader(datasetPath, batch_size)

    # Initialize your model
    model_path = cfg["evaluate"]["model_state_path"]

    model = create_model(cfg["model"])
    model.to("cuda")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state"])

    results_path = Path(cfg["evaluate"]["output_path"])
    now_str = datetime.now().strftime("%Y%m%d%H%M%S")
    runVizPath = results_path / now_str / "viz"
    rundir = results_path / now_str / "eval"
    rundir.mkdir(parents=True, exist_ok=True)
    runVizPath.mkdir(parents=True, exist_ok=True)

    with open(results_path / now_str / "config.yaml", "w") as f:
        yaml.dump(OmegaConf.to_object(cfg), f, default_flow_style=False)
    # os.mkdir(rundir)

    model.eval()

    sum_true_positives = 0
    sum_false_positives = 0
    sum_false_negatives = 0
    sum_assigned_ious = 0
    sum_true_classified = 0
    sum_false_classified = 0

    with torch.no_grad():
        file_num = 0
        for batch_idx, (inputs, targets) in enumerate(
            tqdm(test_loader, desc="Saving results", leave=False)
        ):
            if inputs.shape[0] != batch_size:
                continue

            out = model(inputs)
            inp = rearrange(
                inputs, "B S O C -> B (S O) C", B=batch_size, S=5, O=model.max_seq_len
            )

            (
                true_positives,
                false_positives,
                false_negatives,
                sum_assigned_ious_,
                true_classified,
                false_classified,
            ) = evaluate_fusion(out, targets)

            sum_true_positives += true_positives
            sum_false_positives += false_positives
            sum_false_negatives += false_negatives
            sum_assigned_ious += sum_assigned_ious_
            sum_true_classified += true_classified
            sum_false_classified += false_classified

            for batch_id in range(len(targets)):
                inpBox = inp[batch_id, :, :4].squeeze(0)
                inpCls = inp[batch_id, :, 7].squeeze(0)
                output = {
                    "BBox": out["BBox"][batch_id],
                    "Object_class": out["Object_class"][batch_id],
                    # "Motion_params": out["Motion_params"][batch_id],
                    "target_bbox": targets[batch_id]["BBox"],
                    "target_class": targets[batch_id]["labels"],
                    "target_motion": targets[batch_id]["mot"],
                    "input_bbox": inpBox,
                    "input_class": inpCls,
                }
                # Save the object lists as a numpy file
                data_pair_filename = f"output_{file_num}.npy"
                np.save(
                    os.path.join(rundir, data_pair_filename), output, allow_pickle=True
                )
                np.save(
                    os.path.join(runVizPath, data_pair_filename),
                    output,
                    allow_pickle=True,
                )
                file_num += 1

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

        # IoUs = [0.5, 0.7, 0.9]

        # mAP11, mAPfull = compute_mAP(IoUs)

        evaluations = {
            "mean_iou": mean_iou,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "cls_precision": cls_precision,
            "true_positives": sum_true_positives,
            "false_positives": sum_false_positives,
            "false_negatives": sum_false_negatives,
            "true_classified": sum_true_classified,
            "false_classified": sum_false_classified,
            # 'mAP11': mAP11,
            # 'mAP': mAPfull
        }
        with open(os.path.join(rundir, "evaluation.txt"), "w") as file:
            for k, v in evaluations.items():
                file.write(f"{k}: {v} \n")

    if cfg["evaluate"]["visualization"]:
        from visualization_cv import visualize

        visualize(exportVideo=cfg["evaluate"]["visualization"])


if __name__ == "__main__":
    main()
