"""
实验4 任务1-1：使用 OpenCV DNN 模块调用 YOLOv3 完成单张图像目标检测
"""
import os
from pathlib import Path

import cv2
import numpy as np

EXP4 = Path(__file__).resolve().parent.parent          # e:\xx1\11\experiment4
ROOT = EXP4.parent                                     # e:\xx1\11
# 原始素材位于“DNN目标检测任务(1)/YOLO”，为避免 OpenCV 在中文路径下的读取问题，
# 这里使用已复制到 ASCII 路径下的模型副本（内容与原始素材完全一致）
YOLO_DIR = EXP4 / "models" / "yolo"
OUT_DIR = EXP4 / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONF_THRESHOLD = 0.5   # 置信度阈值
NMS_THRESHOLD = 0.4    # 非极大值抑制阈值


def detect_yolov3(image, net, out_names, classes):
    """对单张图像执行 YOLOv3 推理，返回 (boxes, confidences, class_ids)"""
    H, W = image.shape[:2]
    # 步骤2：图像预处理，得到 416x416 的 blob
    blob = cv2.dnn.blobFromImage(image, 1.0 / 255.0, (416, 416), (0, 0, 0), True, crop=False)
    # 步骤3：设置网络输入
    net.setInput(blob)
    # 步骤4：前向运算，outs 为 3 个输出层（大/中/小目标）
    outs = net.forward(out_names)

    boxes, confidences, class_ids = [], [], []
    for out in outs:
        for candidate in out:
            scores = candidate[5:]                       # 80 个类别的置信度
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence > CONF_THRESHOLD:
                # 中心点 + 宽高（相对原图的比例）
                box = candidate[0:4] * np.array([W, H, W, H])
                center_x, center_y, width, height = box.astype("int")
                # 转成左上角坐标
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                boxes.append([x, y, int(width), int(height)])
                confidences.append(confidence)
                class_ids.append(class_id)

    # 非极大值抑制，去掉大量重叠框
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
    indexes = np.array(indexes).flatten().tolist() if len(indexes) else []
    return boxes, confidences, class_ids, indexes


def draw(image, boxes, confidences, class_ids, indexes, classes, colors):
    for i in indexes:
        x, y, w, h = boxes[i]
        color = [int(c) for c in colors[class_ids[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.0f}%".format(classes[class_ids[i]], confidences[i] * 100)
        tx = max(x, 0)
        ty = max(y - 6, 18)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (tx, ty - th - 4), (tx + tw + 4, ty + 4), color, -1)
        cv2.putText(image, text, (tx + 2, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return image


def main():
    image_path = YOLO_DIR / "test2.jpg"
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"无法读取图像：{image_path}")

    # 类别与颜色
    with open(YOLO_DIR / "coco.names", "rt", encoding="utf-8") as f:
        classes = f.read().strip().split("\n")
    np.random.seed(0)
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")

    # 步骤1：读取 Darknet 网络模型（cfg 结构 + weights 权重）
    net = cv2.dnn.readNetFromDarknet(str(YOLO_DIR / "yolov3.cfg"),
                                     str(YOLO_DIR / "yolov3.weights"))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    out_names = net.getUnconnectedOutLayersNames()

    boxes, confidences, class_ids, indexes = detect_yolov3(image, net, out_names, classes)
    result = draw(image.copy(), boxes, confidences, class_ids, indexes, classes, colors)

    save_path = OUT_DIR / "yolov3_result.jpg"
    cv2.imwrite(str(save_path), result)

    print(f"[YOLOv3] 候选框 {len(boxes)} 个，NMS 后保留 {len(indexes)} 个：")
    for i in indexes:
        x, y, w, h = boxes[i]
        print(f"  {classes[class_ids[i]]:<12s} 置信度={confidences[i]:.3f}  框=({x},{y},{w},{h})")
    print(f"结果已保存到：{save_path}")


if __name__ == "__main__":
    main()
