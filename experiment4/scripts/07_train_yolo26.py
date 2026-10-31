"""
实验4 任务2-3：在扩充后的个人数据集上训练 YOLO26n（CPU 也可运行）
用法：
    python 07_train_yolo26.py --epochs 30 --imgsz 640 --batch 16
"""
import argparse
import sys
from pathlib import Path

EXP4 = Path(__file__).resolve().parent.parent
ROOT = EXP4.parent
SRC = ROOT / "ultralytics-main(1)" / "ultralytics-main"
sys.path.insert(0, str(SRC))

from ultralytics import YOLO  # noqa: E402

MODEL_DIR = EXP4 / "models"
DATA_YAML = EXP4 / "dataset" / "data.yaml"
OUT_DIR = EXP4 / "output"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(MODEL_DIR / "yolo26n.pt"), help="预训练权重")
    parser.add_argument("--data", default=str(DATA_YAML))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--name", default="yolo26n_cells")
    args = parser.parse_args()

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device="cpu",
        project=str(OUT_DIR),
        name=args.name,
        exist_ok=True,
        patience=args.patience,
        seed=0,
        deterministic=True,
        plots=True,
        verbose=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n训练完成，最佳权重：{best}")
    metrics = model.val(data=args.data, split="val", imgsz=args.imgsz, device="cpu", verbose=False)
    print(f"验证集 mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}  "
          f"precision={metrics.box.mp:.4f}  recall={metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()
