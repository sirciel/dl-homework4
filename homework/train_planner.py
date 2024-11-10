"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

#print("Time to train")

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import MLPPlanner, TransformerPlanner, CNNPlanner, WaypointLoss, load_model, save_model
from .datasets.road_dataset import load_data, RoadDataset
from .datasets.road_transforms import EgoTrackProcessor

def train(
    exp_dir: str = "logs",
    model_name: str = "mlp_planner",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    train_data = load_data("drive_data/train/", shuffle=False, batch_size=batch_size)
    val_data = load_data("drive_data/val/", shuffle=False)

    # create loss function and optimizer
    loss_func = WaypointLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    metrics = {"train_loss": [], "val_loss": []}

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        for key in metrics:
            metrics[key].clear()

        model.train()

        for batch in train_data:
            track_left = batch['track_left'].to(device)
            track_right = batch['track_right'].to(device)
            waypoints = batch['waypoints'].to(device)
            waypoints_mask = batch['waypoints_mask'].to(device)

            # TODO: implement training step
            # Forward pass: Compute model predictions (logits and depth)
            predicted_waypoints = model(track_left=track_left, track_right=track_right)

            # Compute loss
            loss = loss_func(predicted_waypoints, waypoints, waypoints_mask)
            metrics["train_loss"].append(loss.item())

            # Backward pass and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            for batch in val_data:
                track_left = batch['track_left'].to(device)
                track_right = batch['track_right'].to(device)
                waypoints = batch['waypoints'].to(device)
                waypoints_mask = batch['waypoints_mask'].to(device)

                # Forward pass
                predicted_waypoints = model(track_left=track_left, track_right=track_right)

                # Calculate validation loss
                val_loss = loss_func(predicted_waypoints, waypoints, waypoints_mask)
                metrics["val_loss"].append(val_loss.item())


        # Log average training and validation loss to TensorBoard
        epoch_train_loss = torch.tensor(metrics["train_loss"]).mean()
        epoch_val_loss = torch.tensor(metrics["val_loss"]).mean()

        logger.add_scalar('train_loss', epoch_train_loss, global_step=global_step)
        logger.add_scalar('val_loss', epoch_val_loss, global_step=global_step)

        # Print progress for specific epochs
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_loss={epoch_train_loss:.4f} "
                f"val_loss={epoch_val_loss:.4f}"
            )

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Computes accuracy by comparing predicted labels with ground truth labels.

    Args:
        logits: tensor (b, num_classes), raw model predictions (unnormalized)
        labels: tensor (b,), true labels

    Returns:
        accuracy: scalar tensor representing the accuracy for the batch
    """
    # Get the index of the max logit for each sample, which represents the predicted class
    _, preds = torch.max(logits, dim=1)

    # Compare predicted class with actual labels and calculate the accuracy
    correct = (preds == labels).float()  # Convert to float for mean calculation
    accuracy = correct.mean()  # Average of correct predictions

    return accuracy