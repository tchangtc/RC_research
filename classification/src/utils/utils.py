"""Utility functions for training and inference."""

import os
import random
import yaml
import torch
import numpy as np
from collections import defaultdict
from torch.optim import Optimizer


def set_all_random_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed


def time_to_str(t, mode="min"):
    """Format time duration as human-readable string.

    Args:
        t: time in seconds
        mode: 'min' for 'X hr Y min', 'sec' for 'X min Y sec'
    """
    if mode == "min":
        t = int(t) / 60
        hr = t // 60
        mn = t % 60
        return f"{int(hr):2d} hr {int(mn):02d} min"
    elif mode == "sec":
        t = int(t)
        mn = t // 60
        sec = t % 60
        return f"{mn:2d} min {sec:02d} sec"
    else:
        raise NotImplementedError


def get_learning_rate(optimizer):
    """Get current learning rate from optimizer."""
    return optimizer.param_groups[0]["lr"]


def adjust_learning_rate(optimizer, lr):
    """Manually set learning rate for all param groups."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def load_config(yaml_path):
    """Load YAML configuration file."""
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    return config


class dotdict(dict):
    """Dictionary subclass that allows attribute-style access.

    Example:
        d = dotdict({'a': 1, 'b': 2})
        d.a  # returns 1
    """

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


class Lookahead(Optimizer):
    """Lookahead optimizer wrapper.

    Paper: https://arxiv.org/abs/1907.08610

    Args:
        optimizer: inner optimizer
        alpha: interpolation factor (0-1). 1.0 = pure inner optimizer.
        k: number of lookahead steps
        pullback_momentum: 'reset', 'pullback', or 'none'
    """

    def __init__(self, optimizer, alpha=0.5, k=6, pullback_momentum="none"):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid alpha: {alpha}")
        if not 1 <= k:
            raise ValueError(f"Invalid k: {k}")

        self.optimizer = optimizer
        self.param_groups = self.optimizer.param_groups
        self.alpha = alpha
        self.k = k
        self.step_counter = 0
        self.pullback_momentum = pullback_momentum
        self.state = defaultdict(dict)

        assert pullback_momentum in ["reset", "pullback", "none"]

        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                param_state["cached_params"] = torch.zeros_like(p.data)
                param_state["cached_params"].copy_(p.data)

    def __getstate__(self):
        return {
            "state": self.state,
            "optimizer": self.optimizer,
            "alpha": self.alpha,
            "step_counter": self.step_counter,
            "k": self.k,
            "pullback_momentum": self.pullback_momentum,
        }

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def _backup_and_load_cache(self):
        """Switch to slow (cached) weights for evaluation."""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                param_state["backup_params"] = torch.zeros_like(p.data)
                param_state["backup_params"].copy_(p.data)
                p.data.copy_(param_state["cached_params"])

    def _clear_and_load_backup(self):
        """Restore fast weights after evaluation."""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                p.data.copy_(param_state["backup_params"])
                del param_state["backup_params"]

    def step(self, closure=None):
        """Perform a single Lookahead optimization step."""
        loss = self.optimizer.step(closure)
        self.step_counter += 1

        if self.step_counter >= self.k:
            self.step_counter = 0
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    param_state = self.state[p]
                    p.data.mul_(self.alpha).add_(
                        param_state["cached_params"], alpha=1.0 - self.alpha
                    )
                    param_state["cached_params"].copy_(p.data)
                    if self.pullback_momentum == "pullback":
                        internal_momentum = self.optimizer.state[p]["momentum_buffer"]
                        self.optimizer.state[p]["momentum_buffer"] = (
                            internal_momentum.mul_(self.alpha).add_(
                                param_state["cached_mom"], alpha=1.0 - self.alpha
                            )
                        )
                        param_state["cached_mom"] = self.optimizer.state[p][
                            "momentum_buffer"
                        ]
                    elif self.pullback_momentum == "reset":
                        self.optimizer.state[p]["momentum_buffer"] = torch.zeros_like(
                            p.data
                        )

        return loss
