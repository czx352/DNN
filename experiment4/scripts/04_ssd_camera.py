"""
实验4 任务1-2：调用本机摄像头，实时执行 SSD 目标检测
用法：python 04_ssd_camera.py [摄像头编号，默认 0]
按 q / ESC 退出，按 s 保存当前画面到 output/camera_ssd.jpg
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

EXP4 = Path(__file__).resolve().parent.parent
SSD_DIR = EXP4 / "models" / "ssd"
OUT_DIR = EXP4 / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONF_THRESHOLD = 0.3
INPUT_SIZE = (300, 300)
MEAN = (127.5, 127.5, 127.5)


def load_model():
    with open(SSD_DIR / "object_detection_classes_pascal_voc.txt", "rt", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]
    net = cv2.dnn.readNetFromCaffe(str(SSD_DIR / "MobileNetSSD_deploy.prototxt.txt"),
                                   str(SSD_DIR / "MobileNetSSD_deploy.caffemodel"))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    np.random.seed(0)
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")
    return net, classes, colors


def detect(frame, net):
    H, W = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, INPUT_SIZE), 0.007843, INPUT_SIZE, MEAN)
    net.setInput(blob)
    detections = net.forward()

    boxes, confidences, class_ids = [], [], []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < CONF_THRESHOLD:
            continue
        class_id = int(detections[0, 0, i, 1])
        x1 = int(detections[0, 0, i, 3] * W)
        y1 = int(detections[0, 0, i, 4] * H)
        x2 = int(detections[0, 0, i, 5] * W)
        y2 = int(detections[0, 0, i, 6] * H)
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        confidences.append(confidence)
        class_ids.append(class_id)
    return boxes, confidences, class_ids


def draw(frame, boxes, confidences, class_ids, classes, colors):
    for (x, y, w, h), conf, cid in zip(boxes, confidences, class_ids):
        color = [int(c) for c in colors[cid % len(colors)]]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = f"{classes[cid]}: {conf * 100:.0f}%"
        ty = max(y - 6, 18)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x, ty - th - 4), (x + tw + 4, ty + 4), color, -1)
        cv2.putText(frame, text, (x + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="摄像头编号，默认 0")
    args = parser.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    net, classes, colors = load_model()
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头/视频源：{source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("开始实时检测：按 q 退出，按 s 保存截图")
    prev = time.time()
    fps = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("读取画面失败，退出")
            break
        frame = cv2.flip(frame, 1)

        boxes, confs, cids = detect(frame, net)
        frame = draw(frame, boxes, confs, cids, classes, colors)

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - prev, 1e-6)
        prev = now
        cv2.putText(frame, f"FPS: {fps:.1f}  objects: {len(boxes)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("SSD Camera Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            path = OUT_DIR / "camera_ssd.jpg"
            cv2.imwrite(str(path), frame)
            print(f"已保存截图：{path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
