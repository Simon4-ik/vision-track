from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune YOLO model weights (optional optimization).")
    parser.add_argument(
        "--weights",
        default="models/checkpoints/best.pt",
        help="Path to input YOLO weights.",
    )
    parser.add_argument(
        "--output",
        default="models/checkpoints/best_pruned.pt",
        help="Path to save pruned weights.",
    )
    parser.add_argument(
        "--amount",
        type=float,
        default=0.2,
        help="Global unstructured pruning amount (0.0-0.9).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch
        import torch.nn.utils.prune as prune
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "PyTorch with pruning utilities is required to run this script."
        ) from exc

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model = ckpt["model"]
    else:
        model = ckpt
    parameters_to_prune = []

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            parameters_to_prune.append((module, "weight"))

    if not parameters_to_prune:
        raise RuntimeError("No Conv2d layers found to prune.")

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=args.amount,
    )

    # Remove pruning reparameterization to make weights dense again
    for module, _ in parameters_to_prune:
        prune.remove(module, "weight")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, out_path)
    print(f"Saved pruned model to {out_path}")


if __name__ == "__main__":
    main()

