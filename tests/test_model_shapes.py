import torch

from src.champion_model import (
    ChampionEnsemble,
    EEGNet,
    MultiScaleTCN,
    RegularizedEEGCNN,
    ShallowConvNet,
)


def test_regularized_cnn_output_shape() -> None:
    model = RegularizedEEGCNN().eval()
    x = torch.randn(3, 250, 62)
    with torch.inference_mode():
        y = model(x)
    assert y.shape == (3, 2)


def test_eegnet_output_shape() -> None:
    model = EEGNet().eval()
    x = torch.randn(2, 250, 62)
    with torch.inference_mode():
        y = model(x)
    assert y.shape == (2, 2)


def test_shallow_convnet_output_shape() -> None:
    model = ShallowConvNet().eval()
    x = torch.randn(2, 250, 62)
    with torch.inference_mode():
        y = model(x)
    assert y.shape == (2, 2)


def test_multiscale_tcn_output_shape() -> None:
    model = MultiScaleTCN().eval()
    x = torch.randn(2, 250, 62)
    with torch.inference_mode():
        y = model(x)
    assert y.shape == (2, 2)


def test_champion_ensemble_can_reload_member_state() -> None:
    base = RegularizedEEGCNN().eval()
    checkpoint = {
        "members": [
            {
                "kind": "regularized_cnn",
                "config": {},
                "state_dict": base.state_dict(),
                "tta_shifts": [0],
                "coefficient": 1.0,
            }
        ],
        "margin_threshold": 0.0,
    }
    model = ChampionEnsemble(checkpoint).eval()
    x = torch.randn(2, 250, 62)
    with torch.inference_mode():
        y = model(x)
    assert y.shape == (2, 2)
