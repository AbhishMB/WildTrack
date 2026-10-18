"""Fine-tune YOLO26 on the dataset built by prepare_dataset.py and install it for WildTrack.

    python training/train.py --data data/wildtrack/wildtrack.yaml            # GPU 0, defaults below
    python training/train.py --data data/wildtrack/wildtrack.yaml --epochs 3 --device cpu   # quick check

Starts from COCO-pretrained YOLO26 weights (yolo26s.pt by default, downloaded on first use),
trains at 1280 px so small animals keep enough pixels, then copies the best checkpoint to
laptop/models/wildtrack-yolo26s.pt, the path laptop/config.yaml already points at. It ends by
printing the class names, which must match classes.targets / classes.confusers in the config.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Fine-tune YOLO26 for WildTrack")
    ap.add_argument("--data", type=Path, required=True, help="wildtrack.yaml from prepare_dataset.py")
    ap.add_argument("--base", default="yolo26s.pt",
                    help="starting weights: yolo26n/s/m.pt (s balances speed on a laptop GPU and accuracy)")
    ap.add_argument("--imgsz", type=int, default=1280, help="match detector.imgsz and the --tile used")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8, help="lower it if the GPU runs out of memory")
    ap.add_argument("--device", default="0", help='"0" for the first GPU, "cpu" for a slow check')
    ap.add_argument("--patience", type=int, default=20, help="stop early after this many epochs without gain")
    ap.add_argument("--project", type=Path, default=REPO / "training" / "runs")
    ap.add_argument("--name", default="wildtrack")
    ap.add_argument("--install-to", type=Path, default=REPO / "laptop" / "models" / "wildtrack-yolo26s.pt")
    args = ap.parse_args(argv)

    from ultralytics import YOLO   # imported here so --help works without the dependency

    model = YOLO(args.base)
    model.train(
        data=str(args.data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
    )

    best = Path(model.trainer.best)
    if not best.is_file():
        raise SystemExit(f"training finished but {best} is missing")
    args.install_to.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, args.install_to)

    names = list(YOLO(str(args.install_to)).names.values())
    print(f"\ninstalled {best} -> {args.install_to}")
    print("classes in the model:", ", ".join(names))
    print("Check laptop/config.yaml: classes.targets and classes.confusers must use these names.")


if __name__ == "__main__":
    main()
