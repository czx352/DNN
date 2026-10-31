"""
实验4 任务2-3：使用训练得到的最佳权重对验证集做推理并保存可视化结果
用法：python 08_predict_trained.py
"""
import sys
from pathlib import Path

EXP4 = Path(__file__).resolve().parent.parent
ROOT = EXP4.parent
SRC = ROOT / "ultralytics-main(1)" / "ultralytics-main"
sys.path.insert(0, str(SRC))

from ultralytics import YOLO  # noqa: E402

BEST = EXP4 / "output" / "yolo26n_cells" / "weights" / "best.pt"
VAL_IMAGES = EXP4 / "dataset" / "images" / "val"
OUT_DIR = EXP4 / "output"


def count_gt_boxes(label_path):
    if not label_path.exists():
        return 0
    return len([ln for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()])


def main():
    if not BEST.exists():
        raise FileNotFoundError(f"未找到训练权重：{BEST}，请先运行 07_train_yolo26.py")

    model = YOLO(str(BEST))
    results = model.predict(source=str(VAL_IMAGES), imgsz=640, conf=0.25,
                            save=True, project=str(OUT_DIR),
                            name="task2_predict_val", exist_ok=True,
                            device="cpu", verbose=False)

    print(f"{'图像':<28s}{'真实框':>8s}{'预测框':>8s}")
    total_gt = total_pred = 0
    for r in results:
        stem = Path(r.path).stem
        gt = count_gt_boxes(EXP4 / "dataset" / "labels" / "val" / f"{stem}.txt")
        pred = len(r.boxes)
        total_gt += gt
        total_pred += pred
        print(f"{stem:<28s}{gt:>8d}{pred:>8d}")
    print(f"{'合计':<28s}{total_gt:>8d}{total_pred:>8d}")
    print(f"可视化结果目录：{OUT_DIR / 'task2_predict_val'}")


if __name__ == "__main__":
    main()
