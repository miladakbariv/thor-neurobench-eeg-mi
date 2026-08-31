#!/usr/bin/env python3
"""Run the public NeuroBench validation adapter for a compatible checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark a compatible THOR EEG-MI checkpoint with NeuroBench."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a champion checkpoint compatible with src.champion_model.",
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Directory used by NeuroBench for the ThorEEGMI dataset cache.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Validation batch size (default: 64).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    import torch
    from torch.utils.data import DataLoader

    from neurobench.benchmarks import Benchmark
    from neurobench.datasets import ThorEEGMI
    from neurobench.metrics.static import ConnectionSparsity, Footprint, ParameterCount
    from neurobench.metrics.workload import (
        ActivationSparsity,
        ClassificationAccuracy,
        SynapticOperations,
    )
    from neurobench.models import TorchModel

    from src.champion_model import load_champion

    device = torch.device("cpu")
    validation_set = ThorEEGMI(root=args.data_dir, split="val", download=True)
    loader = DataLoader(
        validation_set,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = load_champion(args.checkpoint, map_location=device).to(device).eval()

    def postprocess(logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=1)

    benchmark = Benchmark(
        model=TorchModel(model),
        dataloader=loader,
        preprocessors=[],
        postprocessors=[postprocess],
        metric_list=[
            [Footprint, ParameterCount, ConnectionSparsity],
            [ClassificationAccuracy, ActivationSparsity, SynapticOperations],
        ],
    )

    print(benchmark.run())


if __name__ == "__main__":
    main()
