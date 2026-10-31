"""
实验4 任务2-2：把 labelme 标注转换为 YOLO 格式，并通过数据增强扩大数据集
输入：data(1)/images（100 张 1024x1024 切图）、data(1)/jsons（labelme 矩形标注）
输出：experiment4/dataset
    images/train、images/val、labels/train、labels/val、data.yaml
说明：
    1. 先按 8:2 划分原始图像，避免同一张图的增强副本同时进入训练集和验证集（防止数据泄漏）；
    2. 只对训练集原图做 3 种几何增强（水平翻转、垂直翻转、顺时针旋转 90°），
       数据集由 80 张训练图扩充为 320 张；
    3. 标注框随图像同步变换，并裁剪到图像范围内。
"""
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

EXP4 = Path(__file__).resolve().parent.parent
ROOT = EXP4.parent
IMG_DIR = ROOT / "data(1)" / "images"
JSON_DIR = ROOT / "data(1)" / "jsons"
DS_DIR = EXP4 / "dataset"
OUT_DIR = EXP4 / "output"

VAL_RATIO = 0.2
SEED = 0

# labelme 中的标签 -> YOLO 类别 id（本数据集仅一个类别 c，即细胞）
CLASS_MAP = {"c": 0}
CLASS_NAMES = ["cell"]


def labelme_to_boxes(json_path):
    """读取 labelme json，返回 [(class_id, x1, y1, x2, y2), ...]（像素坐标）"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    boxes = []
    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "rectangle":
            continue
        label = shape["label"]
        if label not in CLASS_MAP:
            continue
        (x1, y1), (x2, y2) = shape["points"][:2]
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        boxes.append((CLASS_MAP[label], x1, y1, x2, y2))
    return boxes


def clamp_boxes(boxes, w, h):
    out = []
    for cid, x1, y1, x2, y2 in boxes:
        x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
        y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
        if x2 - x1 >= 2 and y2 - y1 >= 2:      # 过滤退化的极小框
            out.append((cid, x1, y1, x2, y2))
    return out


def boxes_to_yolo_lines(boxes, w, h):
    lines = []
    for cid, x1, y1, x2, y2 in boxes:
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def augment(kind, image, boxes):
    """几何增强：返回增强后的图像与变换后的标注框"""
    h, w = image.shape[:2]
    if kind == "hflip":
        new_img = image[:, ::-1].copy()
        boxes = [(c, w - x2, y1, w - x1, y2) for c, x1, y1, x2, y2 in boxes]
    elif kind == "vflip":
        new_img = image[::-1, :].copy()
        boxes = [(c, x1, h - y2, x2, h - y1) for c, x1, y1, x2, y2 in boxes]
    elif kind == "rot180":
        new_img = image[::-1, ::-1].copy()
        boxes = [(c, w - x2, h - y2, w - x1, h - y1) for c, x1, y1, x2, y2 in boxes]
    elif kind == "rot90":
        new_img = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        # 顺时针旋转 90°：新图宽=原图高，新图高=原图宽
        boxes = [(c, h - y2, x1, h - y1, x2) for c, x1, y1, x2, y2 in boxes]
    else:
        raise ValueError(kind)
    nh, nw = new_img.shape[:2]
    return new_img, clamp_boxes(boxes, nw, nh)


def save_sample(sub, stem, image, boxes):
    img_path = DS_DIR / "images" / sub / f"{stem}.jpg"
    lbl_path = DS_DIR / "labels" / sub / f"{stem}.txt"
    cv2.imwrite(str(img_path), image)
    h, w = image.shape[:2]
    with open(lbl_path, "w", encoding="utf-8") as f:
        f.write("\n".join(boxes_to_yolo_lines(boxes, w, h)))
    return img_path


def draw_preview(image, boxes, save_path):
    img = image.copy()
    for cid, x1, y1, x2, y2 in boxes:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(img, f"labels: {len(boxes)} (class: {CLASS_NAMES[0]})", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.imwrite(str(save_path), img)


def main():
    if DS_DIR.exists():
        shutil.rmtree(DS_DIR)
    for p in ["images/train", "images/val", "labels/train", "labels/val"]:
        (DS_DIR / p).mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stems = sorted(p.stem for p in JSON_DIR.glob("*.json"))
    random.seed(SEED)
    random.shuffle(stems)
    n_val = int(len(stems) * VAL_RATIO)
    val_stems, train_stems = sorted(stems[:n_val]), sorted(stems[n_val:])
    print(f"原始图像 {len(stems)} 张 -> 训练 {len(train_stems)} 张 / 验证 {len(val_stems)} 张")

    total_train = total_val = 0
    total_boxes = 0
    preview_done = False

    for stem in train_stems:
        image = cv2.imread(str(IMG_DIR / f"{stem}.jpg"))
        boxes = clamp_boxes(labelme_to_boxes(JSON_DIR / f"{stem}.json"),
                            image.shape[1], image.shape[0])
        if not boxes:
            print(f"  [跳过] {stem} 无有效标注")
            continue
        # 原图 + 3 种几何增强（训练集扩充 4 倍）
        for kind in ["none", "hflip", "vflip", "rot90"]:
            if kind == "none":
                aug_img, aug_boxes = image, boxes
                suffix = ""
            else:
                aug_img, aug_boxes = augment(kind, image, boxes)
                suffix = f"_{kind}"
            save_sample("train", f"{stem}{suffix}", aug_img, aug_boxes)
            total_train += 1
            total_boxes += len(aug_boxes)
            if not preview_done:
                draw_preview(aug_img, aug_boxes, OUT_DIR / "dataset_gt_preview.jpg")
                preview_done = True

    for stem in val_stems:
        image = cv2.imread(str(IMG_DIR / f"{stem}.jpg"))
        boxes = clamp_boxes(labelme_to_boxes(JSON_DIR / f"{stem}.json"),
                            image.shape[1], image.shape[0])
        if not boxes:
            continue
        save_sample("val", stem, image, boxes)
        total_val += 1

    # 写 data.yaml
    yaml_path = DS_DIR / "data.yaml"
    yaml_path.write_text(
        f"path: {DS_DIR.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n  0: {CLASS_NAMES[0]}\n",
        encoding="utf-8",
    )

    print(f"训练集图像 {total_train} 张（原始 {len(train_stems)} × 4 增强），验证集 {total_val} 张")
    print(f"标注框总数（训练集增强后）{total_boxes} 个")
    print(f"数据集目录：{DS_DIR}")
    print(f"配置文件：{yaml_path}")


if __name__ == "__main__":
    main()
