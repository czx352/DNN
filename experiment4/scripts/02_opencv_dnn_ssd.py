"""
实验4 任务1-1：使用 OpenCV DNN 模块调用 SSD(MobileNet) 完成单张图像目标检测
"""
from pathlib import Path

import cv2
import numpy as np

EXP4 = Path(__file__).resolve().parent.parent
ROOT = EXP4.parent
SSD_DIR = EXP4 / "models" / "ssd"
OUT_DIR = EXP4 / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONF_THRESHOLD = 0.3   # SSD 置信度普遍低于 YOLOv3，阈值取 0.3
INPUT_SIZE = (300, 300)
MEAN = (127.5, 127.5, 127.5)


def load_classes(path):
    with open(path, "rt", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def detect_ssd(image, net, classes):
    """SSD 推理：输出格式为 [batchId, classId, confidence, x1, y1, x2, y2]（归一化坐标）"""
    H, W = image.shape[:2]
    # 预处理：均值 127.5、缩放 1/127.5，得到 300x300 的 blob
    blob = cv2.dnn.blobFromImage(cv2.resize(image, INPUT_SIZE), 0.007843,
                                 INPUT_SIZE, MEAN)
    net.setInput(blob)
    detections = net.forward()   # shape: (1, 1, 100, 7)

    boxes, confidences, class_ids = [], [], []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < CONF_THRESHOLD:
            continue
        class_id = int(detections[0, 0, i, 1])
        # 归一化坐标 -> 像素坐标
        x1 = int(detections[0, 0, i, 3] * W)
        y1 = int(detections[0, 0, i, 4] * H)
        x2 = int(detections[0, 0, i, 5] * W)
        y2 = int(detections[0, 0, i, 6] * H)
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        confidences.append(confidence)
        class_ids.append(class_id)
    return boxes, confidences, class_ids


def draw_ssd(image, boxes, confidences, class_ids, classes):
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")
    for box, conf, cid in zip(boxes, confidences, class_ids):
        x, y, w, h = box
        color = [int(c) for c in colors[cid % len(colors)]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        label = classes[cid] if cid < len(classes) else str(cid)
        text = f"{label}: {conf * 100:.1f}%"
        tx = max(x, 0)
        ty = max(y - 6, 18)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (tx, ty - th - 4), (tx + tw + 4, ty + 4), color, -1)
        cv2.putText(image, text, (tx + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return image


def build_net():
    net = cv2.dnn.readNetFromCaffe(str(SSD_DIR / "MobileNetSSD_deploy.prototxt.txt"),
                                   str(SSD_DIR / "MobileNetSSD_deploy.caffemodel"))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def main():
    classes = load_classes(SSD_DIR / "object_detection_classes_pascal_voc.txt")
    net = build_net()

    image = cv2.imread(str(SSD_DIR / "test2.jpg"))
    if image is None:
        raise FileNotFoundError("无法读取 test2.jpg")

    boxes, confidences, class_ids = detect_ssd(image, net, classes)
    result = draw_ssd(image.copy(), boxes, confidences, class_ids, classes)

    save_path = OUT_DIR / "ssd_result.jpg"
    cv2.imwrite(str(save_path), result)

    print(f"[SSD] 保留 {len(boxes)} 个目标：")
    for box, conf, cid in zip(boxes, confidences, class_ids):
        print(f"  {classes[cid]:<12s} 置信度={conf:.3f}  框={tuple(box)}")
    print(f"结果已保存到：{save_path}")


if __name__ == "__main__":
    main()
