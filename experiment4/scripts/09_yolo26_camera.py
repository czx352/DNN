"""
实验4 任务2 扩展：把训练好的 YOLO26n 权重接到摄像头上做实时检测
用法：python 09_yolo26_camera.py [摄像头编号，默认 0]
按 q / ESC 退出，按 s 保存截图到 output/camera_yolo26.jpg
"""
import argparse
import time
from pathlib import Path

import cv2

EXP4 = Path(__file__).resolve().parent.parent
BEST = EXP4 / "output" / "yolo26n_cells" / "weights" / "best.pt"
OUT_DIR = EXP4 / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    parser.add_argument("--weights", default=str(BEST))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    # 延迟导入，保证脚本可直接运行
    import sys
    ROOT = EXP4.parent
    sys.path.insert(0, str(ROOT / "ultralytics-main(1)" / "ultralytics-main"))
    from ultralytics import YOLO

    model = YOLO(args.weights)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头/视频源：{source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("开始实时检测：按 q 退出，按 s 保存截图")
    prev, fps = time.time(), 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        t0 = time.time()
        r = model.predict(frame, imgsz=args.imgsz, conf=args.conf, device="cpu", verbose=False)[0]
        frame = r.plot()
        infer_ms = (time.time() - t0) * 1000

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - prev, 1e-6)
        prev = now
        cv2.putText(frame, f"FPS: {fps:.1f}  infer: {infer_ms:.0f}ms  objects: {len(r.boxes)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("YOLO26 Camera Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            path = OUT_DIR / "camera_yolo26.jpg"
            cv2.imwrite(str(path), frame)
            print(f"已保存截图：{path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
