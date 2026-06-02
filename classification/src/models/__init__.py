"""Model definitions for rectal cancer T-stage classification."""

from .efficientnet_2d import EffNet
from .ynet_2d import YNet2D, Classification2D
from .ynet_3d import YNet3D, Classification3D


def build_model(config: dict) -> object:
    """Build model from config dict.

    Args:
        config: dict with key 'model' containing at least 'name'.
                Supported names: 'efficientnet_b0', 'ynet2d', 'ynet_3d'.

    Returns:
        Instantiated model.
    """
    model_cfg = config.get("model", {})
    name = model_cfg.get("name", "efficientnet_b0")

    if name in ("efficientnet_b0", "efficientnet_b2", "efficientnet_b4"):
        return EffNet(encoder_name=name, pretrained=model_cfg.get("pretrained", True))
    elif name == "ynet2d":
        return Classification2D()
    elif name == "ynet_3d":
        input_channels = model_cfg.get("input_channels", 2)
        return Classification3D(input_channels=input_channels)
    else:
        raise ValueError(f"Unknown model: {name}")
