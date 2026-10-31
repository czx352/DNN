"""
实验4 任务2-1：使用 YOLO26 预训练模型完成推理（CPU 即可运行）
使用提供的 ultralytics 源码（实验素材中的 ultralytics-main），模型默认 yolo26n.pt
用法示例：
    python 05_yolo26_predict.py --source ../../data(1)/images --model yolo26n.pt
"""
import argparse
import sys
from pathlib import Path

EXP4 = Path(__file__).resolve().parent.parent
ROOT = EXP4.parent
# 优先使用实验素材中的 ultralytics 源码（8.4.x，支持 YOLO26）
SRC = ROOT / "ultralytics-main(1)" / "ultralytics-main"
sys.path.insert(0, str(SRC))

from ultralytics import YOLO  # noqa: E402

MODEL_DIR = EXP4 / "models"
OUT_DIR = EXP4 / "output"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(MODEL_DIR / "yolo26n.pt"), help="模型权重")
    parser.add_argument("--source", default=str(ROOT / "data(1)" / "images"), help="图像/目录")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--name", default="task1_yolo26_predict")
    args = parser.parse_args()

    model = YOLO(args.model)
    results = model.predict(source=args.source, imgsz=args.imgsz, conf=args.conf,
                            save=True, save_txt=False, project=str(OUT_DIR),
                            name=args.name, exist_ok=True, device="cpu", verbose=False)

    print(f"共处理 {len(results)} 张图像")
    for r in results[:5]:
        names = [model.names[int(c)] for c in r.boxes.cls.tolist()]
        counts = {n: names.count(n) for n in set(names)}
        print(f"  {Path(r.path).name}: {counts if counts else '无检测目标'}")
    print(f"可视化结果目录：{OUT_DIR / args.name}")


if __name__ == "__main__":
    main()
