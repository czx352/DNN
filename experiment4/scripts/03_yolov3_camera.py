"""
实验4 任务1-2：调用本机摄像头，实时执行 YOLOv3 目标检测
用法：python 03_yolov3_camera.py [摄像头编号，默认 0]
按 q / ESC 退出，按 s 保存当前画面到 output/camera_yolov3.jpg
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

EXP4 = Path(__file__).resolve().parent.parent
YOLO_DIR = EXP4 / "models" / "yolo"
OUT_DIR = EXP4 / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4


def load_model():
    with open(YOLO_DIR / "coco.names", "rt", encoding="utf-8") as f:
        classes = f.read().strip().split("\n")
    net = cv2.dnn.readNetFromDarknet(str(YOLO_DIR / "yolov3.cfg"),
                                     str(YOLO_DIR / "yolov3.weights"))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    np.random.seed(0)
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")
    return net, net.getUnconnectedOutLayersNames(), classes, colors


def detect(frame, net, out_names):
    """对一帧图像做 YOLOv3 推理，返回保留框"""
    H, W = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0 / 255.0, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(out_names)

    boxes, confidences, class_ids = [], [], []
    for out in outs:
        for candidate in out:
            scores = candidate[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence > CONF_THRESHOLD:
                box = candidate[0:4] * np.array([W, H, W, H])
                cx, cy, w, h = box.astype("int")
                boxes.append([int(cx - w / 2), int(cy - h / 2), int(w), int(h)])
                confidences.append(confidence)
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
    indexes = np.array(indexes).flatten().tolist() if len(indexes) else []
    return [boxes[i] for i in indexes], [confidences[i] for i in indexes], [class_ids[i] for i in indexes]


def draw(frame, boxes, confidences, class_ids, classes, colors):
    for (x, y, w, h), conf, cid in zip(boxes, confidences, class_ids):
        color = [int(c) for c in colors[cid]]
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

    net, out_names, classes, colors = load_model()

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
        # 摄像头画面镜像显示，操作更自然
        frame = cv2.flip(frame, 1)

        boxes, confs, cids = detect(frame, net, out_names)
        frame = draw(frame, boxes, confs, cids, classes, colors)

        # 实时 FPS
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - prev, 1e-6)
        prev = now
        cv2.putText(frame, f"FPS: {fps:.1f}  objects: {len(boxes)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("YOLOv3 Camera Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            path = OUT_DIR / "camera_yolov3.jpg"
            cv2.imwrite(str(path), frame)
            print(f"已保存截图：{path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
