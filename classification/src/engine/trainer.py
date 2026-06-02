"""Unified training loop for 2D and 3D T-stage classification."""

import os
import time
import numpy as np
import torch
import torch.cuda.amp as amp
from torch.utils.tensorboard import SummaryWriter

from ..utils.utils import time_to_str, get_learning_rate
from .evaluator import evaluate


class Trainer:
    """Unified trainer for both 2D and 3D classification.

    Args:
        model: PyTorch model
        train_loader: training DataLoader
        val_loader: validation DataLoader
        config: full config dict (from YAML)
        label_key: key for labels in batch dict ('label' for 2D, 'T_stage' for 3D)
    """

    def __init__(self, model, train_loader, val_loader, config, label_key="label"):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.label_key = label_key

        train_cfg = config.get("training", {})
        loss_cfg = config.get("loss", {})
        opt_cfg = config.get("optimizer", {})
        sched_cfg = config.get("scheduler", {})
        output_cfg = config.get("output", {})

        self.epochs = train_cfg.get("epochs", 300)
        self.mixed_precision = train_cfg.get("mixed_precision", True)
        self.skip_save_epochs = output_cfg.get("checkpoint_every", 3)

        self.bce_weight = loss_cfg.get("bce_weight", 0.5)
        self.l1s_weight = loss_cfg.get("smooth_l1_weight", 0.5)
        self.dice_weight = loss_cfg.get("dice_weight", 0.0)

        # Setup optimizer
        lr = opt_cfg.get("lr", 1e-5)
        betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
        eps = opt_cfg.get("eps", 1e-7)
        wd = opt_cfg.get("weight_decay", 1e-4)
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=wd,
        )

        # Setup scheduler
        step_size = sched_cfg.get("step_size", 25)
        gamma = sched_cfg.get("gamma", 0.5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=step_size, gamma=gamma,
        )

        # AMP scaler
        self.scaler = amp.GradScaler(enabled=self.mixed_precision)

        # Output directory
        self.save_dir = output_cfg.get("save_dir", "outputs")
        os.makedirs(f"{self.save_dir}/checkpoint", exist_ok=True)
        self.writer = SummaryWriter(self.save_dir)
        self.log_file = open(f"{self.save_dir}/log.train.txt", mode="a")

        # Tensor keys for moving to GPU
        self.tensor_keys = ["image", self.label_key]

    def _log(self, msg):
        self.log_file.write(msg + "\n")
        self.log_file.flush()

    def train(self):
        """Run the full training loop."""
        model = self.model
        num_epoch = self.epochs
        num_iteration = num_epoch * len(self.train_loader)
        iter_save = int(len(self.train_loader) * self.skip_save_epochs)
        iter_valid = iter_save
        iter_log = iter_save

        self._log(f"** Config **")
        self._log(f"epochs: {num_epoch}")
        self._log(f"optimizer: {self.optimizer}")
        self._log(f"scheduler: {self.scheduler}")
        self._log(f"\n** Start Training **\n")

        # State
        start_timer = time.time()
        iteration = 0
        epoch = 0.0
        rate = 0.0

        valid_loss = np.zeros(6, np.float32)
        train_loss = np.zeros(4, np.float32)
        batch_loss = np.zeros(4, np.float32)
        sum_train_loss = np.zeros(4, np.float32)
        sum_train = 0

        while iteration < num_iteration:
            self.writer.add_scalar("Train/lr", rate, int(epoch) + 1)

            for t, batch in enumerate(self.train_loader):
                # Save checkpoint
                if iteration % iter_save == 0 and iteration > 0 and epoch >= self.skip_save_epochs:
                    torch.save({
                        "state_dict": model.state_dict(),
                        "iteration": iteration,
                        "epoch": int(epoch),
                    }, f"{self.save_dir}/checkpoint/{int(epoch):03d}_model.pth")

                # Validate
                if iteration > 0 and iteration % iter_valid == 0:
                    valid_loss = evaluate(
                        model, self.val_loader, self.config,
                        label_key=self.label_key,
                    )

                # Log
                if iteration % iter_log == 0 or (iteration > 0 and iteration % iter_valid == 0):
                    rate = get_learning_rate(self.optimizer)
                    elapsed = time_to_str(time.time() - start_timer, "min")
                    msg = (
                        f"{rate:.2e}  {iteration:08d}  {epoch:6.2f} | "
                        f"val: {' '.join(f'{v:.4f}' for v in valid_loss)} | "
                        f"train: {' '.join(f'{v:.4f}' for v in train_loss)} | "
                        f"{elapsed}"
                    )
                    self._log(msg)

                # Training step
                rate = get_learning_rate(self.optimizer)
                batch_size = len(batch["index"])

                for k in self.tensor_keys:
                    batch[k] = batch[k].cuda()

                model.train()
                model.output_type = ["loss", "inference"]

                with amp.autocast(enabled=self.mixed_precision):
                    output = model(batch)
                    loss1 = output["bce_loss"].mean()
                    loss2 = output["l1s_loss"].mean()
                    loss3 = output["dice_loss"].mean()
                    loss = (loss1 * self.bce_weight +
                            loss2 * self.l1s_weight +
                            loss3 * self.dice_weight)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Track losses
                batch_loss[:] = [
                    loss.item(),
                    self.bce_weight * loss1.item(),
                    self.l1s_weight * loss2.item(),
                    self.dice_weight * loss3.item(),
                ]
                sum_train_loss += batch_loss
                sum_train += 1

                if t % 100 == 0:
                    train_loss = sum_train_loss / (sum_train + 1e-12)
                    sum_train_loss[...] = 0
                    sum_train = 0

                epoch += 1.0 / len(self.train_loader)
                iteration += 1

            self.scheduler.step()
            self.writer.add_scalar("Train/train_loss", loss.item(), int(epoch) + 1)
            torch.cuda.empty_cache()

        self._log("\n** Training Complete **\n")
        self.log_file.close()
        self.writer.close()
