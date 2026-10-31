"""
生成实验4 实验报告（Word，标准实验报告结构）
运行：py make_report.py
"""
import csv
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

EXP4 = Path(__file__).resolve().parent.parent
OUT = EXP4 / "output"
RUN = OUT / "yolo26n_cells"
REPORT = EXP4 / "实验4 目标检测案例练习-实验报告.docx"


def set_font(doc):
    style = doc.styles["Normal"]
    style.font.name = "宋体"
    style.font.size = Pt(10.5)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    for lvl, size in [(1, 15), (2, 12.5)]:
        hstyle = doc.styles[f"Heading {lvl}"]
        hstyle.font.name = "黑体"
        hstyle.font.size = Pt(size)
        hstyle.font.color.rgb = RGBColor(0, 0, 0)
        hstyle.element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")


def h(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    for run in p.runs:
        run.font.name = "黑体"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
    p.paragraph_format.space_before = Pt(10 if level == 1 else 6)
    p.paragraph_format.space_after = Pt(4)
    return p


def para(doc, text, bold=False, size=10.5, indent=True, align=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.25
    if indent:
        p.paragraph_format.first_line_indent = Pt(21)
    if align is not None:
        p.alignment = align
    run = p.add_run(text)
    run.bold = bold
    run.font.size = Pt(size)
    return p


def code(doc, text):
    for line in text.strip("\n").split("\n"):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.left_indent = Pt(12)
        run = p.add_run(line if line else " ")
        run.font.name = "Consolas"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
        run.font.size = Pt(8.5)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def image(doc, path, caption, width=14):
    path = Path(path)
    if not path.exists():
        para(doc, f"[缺失图片] {path.name}", indent=False)
        return
    doc.add_picture(str(path), width=Cm(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph(caption)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(8)
    for r in p.runs:
        r.font.size = Pt(9)
        r.font.name = "宋体"


def table(doc, header, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, text in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = text
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs:
                r.bold = True
                r.font.size = Pt(9.5)
    for row in rows:
        cells = t.add_row().cells
        for i, text in enumerate(row):
            cells[i].text = str(text)
            for p in cells[i].paragraphs:
                p.paragraph_format.space_after = Pt(0)
                for r in p.runs:
                    r.font.size = Pt(9.5)
    if widths:
        for i, w in enumerate(widths):
            for row in t.rows:
                row.cells[i].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def read_best_metrics():
    csv_path = RUN / "results.csv"
    if not csv_path.exists():
        return None
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    key = "metrics/mAP50-95(B)"
    best = max(rows, key=lambda r: float(r[key]))
    return {
        "epoch": int(float(best["epoch"])),
        "mAP50": float(best["metrics/mAP50(B)"]),
        "mAP50_95": float(best[key]),
        "P": float(best["metrics/precision(B)"]),
        "R": float(best["metrics/recall(B)"]),
        "epochs_total": len(rows),
    }


def build(doc):
    m = read_best_metrics() or {}

    # ================= 封面信息 =================
    title = doc.add_heading("实验4  目标检测案例练习", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para(doc, "实 验 报 告", size=18, indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, bold=True)
    para(doc, "姓名：____________　　学号：____________　　班级：____________", indent=False,
         align=WD_ALIGN_PARAGRAPH.CENTER)
    para(doc, "课程：计算机视觉　　实验日期：2026 年 10 月 30 日", indent=False,
         align=WD_ALIGN_PARAGRAPH.CENTER)

    # ================= 一、实验目的 =================
    h(doc, "一、实验目的")
    para(doc, "1. 熟悉计算机视觉中的常见任务（图像分类、目标检测），理解检测任务中“类别 + 位置”的输出形式。")
    para(doc, "2. 掌握 OpenCV 中 cv2.dnn 模块的通用推理流程，能够用同一套接口部署 YOLOv3、SSD 等不同框架训练出的检测模型。")
    para(doc, "3. 能够把静态图像检测程序改造为摄像头实时检测程序，并观察帧率与检测效果的变化。")
    para(doc, "4. 熟悉 YOLO26 轻量模型（n/s）的使用，能够在 CPU 上完成预训练推理；"
              "掌握把 labelme 标注转换为 YOLO 格式、扩充数据集、划分训练/验证集、训练与评估的完整流程。")

    # ================= 二、实验原理 =================
    h(doc, "二、实验原理")
    h(doc, "2.1 OpenCV DNN 通用推理", level=2)
    para(doc, "cv2.dnn 模块本身不负责训练，只作为“通用推理引擎”：它把不同框架（Darknet、Caffe、ONNX、TensorFlow 等）"
              "的模型解析为统一的网络结构，然后按四步执行——读取网络（readNet 系列函数）、图像预处理"
              "（blobFromImage）、设置输入（setInput）、前向计算（forward）。因此更换模型时，"
              "主要改动的是读模型函数与后处理，而推理流程保持不变。")

    h(doc, "2.2 YOLOv3 检测原理", level=2)
    para(doc, "YOLOv3 以 Darknet-53 为主干，通过 3 个不同尺度的特征图输出检测结果（分别负责大、中、小目标），"
              "每个网格预测 3 个 anchor。每个候选框由 85 个数值组成：前 4 个是相对于原图的中心点坐标与宽高比例，"
              "第 5 个是框置信度，后 80 个是 COCO 80 类的类别置信度。")
    para(doc, "后处理分两步：先对每个候选框取类别置信度最大值，保留大于阈值（本实验 0.5）的框；"
              "再用非极大值抑制 NMS（IoU 阈值 0.4）去除同一目标的重复框。")

    h(doc, "2.3 SSD 检测原理", level=2)
    para(doc, "SSD 使用 MobileNet 作为主干网络，在多个尺度的特征图上铺设不同大小与长宽比的默认框（default boxes），"
              "一次前向即可直接输出固定数量（100 个）的检测结果。每个结果为"
              " [batchId, classId, confidence, x1, y1, x2, y2]，坐标是归一化值。"
              "由于网络在训练时已经完成匹配与抑制，推理阶段不需要再做 NMS，只需按置信度阈值过滤。")

    h(doc, "2.4 YOLO26 模型特点", level=2)
    para(doc, "YOLO26 是 Ultralytics 推出的新一代实时检测模型族，n/s/m/l/x 五个规模在 COCO 上的指标如下表。"
              "其设计特点包括：可选的一对一检测头实现端到端输出（无需 NMS）、去除分布焦点损失 DFL 以简化检测头、"
              "采用 MuSGD（Muon + SGD）优化器与渐进式训练策略。YOLO26n 仅 2.4 M 参数、5.5 B FLOPs，"
              "CPU ONNX 推理约 39 ms，非常适合本实验的无 GPU 环境。")
    table(doc, ["模型", "size", "mAPval 50-95", "CPU ONNX(ms)", "params(M)", "FLOPs(B)"], [
        ["YOLO26n", "640", "40.9", "38.9 ± 0.7", "2.4", "5.5"],
        ["YOLO26s", "640", "48.6", "87.2 ± 0.9", "9.5", "20.9"],
        ["YOLO26m", "640", "53.1", "220.0 ± 1.4", "20.4", "68.4"],
        ["YOLO26l", "640", "55.0", "286.2 ± 2.0", "24.8", "86.8"],
        ["YOLO26x", "640", "57.5", "525.8 ± 4.0", "55.7", "194.4"],
    ], widths=[2.6, 1.8, 3.2, 3.6, 2.4, 2.4])
    para(doc, "注：表中指标为 COCO 数据集、640 输入下的官方数据。本实验考虑 CPU 推理与训练速度，选择 YOLO26n。")

    h(doc, "2.5 检测评价指标", level=2)
    para(doc, "IoU（交并比）衡量预测框与真实框的重叠程度；Precision 表示预测为正的样本中真正为正的比例，"
              "Recall 表示真实目标被找出的比例；以不同置信度阈值绘制 P-R 曲线，曲线下面积即 AP；"
              "对所有类别取平均得到 mAP。mAP50 指 IoU 阈值为 0.5 时的 mAP，mAP50-95 是 IoU 从 0.5 到 0.95（步长 0.05）"
              "的平均值，要求更严格。")

    # ================= 三、实验环境 =================
    h(doc, "三、实验环境")
    table(doc, ["项目", "配置 / 版本"], [
        ["操作系统", "Windows 11"],
        ["CPU", "AMD Ryzen 7 7745HX，8 核 16 线程（无 NVIDIA GPU，全部计算在 CPU 上完成）"],
        ["内存 / 硬盘", "16 GB / SSD"],
        ["Python", "3.11.9"],
        ["深度学习框架", "PyTorch 2.14.0 (CPU 版)"],
        ["OpenCV", "4.11.0（实验专用虚拟环境 .venv）/ 5.0.0（系统默认环境）"],
        ["YOLO 框架", "ultralytics 8.4.189（实验素材源码 ultralytics-main）"],
        ["模型文件", "yolov3.cfg + yolov3.weights、MobileNetSSD_deploy、yolo26n.pt"],
        ["实验数据", "data(1)：100 张 1024×1024 细胞切片图 + 100 个 labelme json 标注"],
    ], widths=[4, 12])

    # ================= 四、实验步骤 =================
    h(doc, "四、实验步骤")

    h(doc, "4.1 任务1-1：用 OpenCV DNN 完成 YOLOv3 图像检测", level=2)
    para(doc, "（1）读取类别文件 coco.names（80 类）与 Darknet 模型；"
              "（2）用 blobFromImage 把图像缩放到 416×416 并归一化到 0~1；"
              "（3）前向计算得到 3 个输出层；（4）逐层解码候选框，按 0.5 置信度筛选；"
              "（5）NMS 去重后绘制矩形框、类别与置信度。核心代码如下：")
    code(doc, '''
net  = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(net.getUnconnectedOutLayersNames())      # 3 个尺度

for out in outs:
    for candidate in out:                                   # 每个候选框 85 个值
        scores     = candidate[5:]                          # 80 类置信度
        class_id   = int(np.argmax(scores))
        confidence = float(scores[class_id])
        if confidence > 0.5:
            box = candidate[0:4] * np.array([W, H, W, H])   # 归一化 -> 像素
            cx, cy, w, h = box.astype("int")
            boxes.append([int(cx - w/2), int(cy - h/2), int(w), int(h)])
            confidences.append(confidence)
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)    # 非极大值抑制
''')

    h(doc, "4.2 任务1-1：用 OpenCV DNN 完成 SSD 图像检测", level=2)
    para(doc, "（1）readNetFromCaffe 读取 prototxt 与 caffemodel；"
              "（2）把图像缩放到 300×300，均值 127.5、缩放系数 1/127.5；"
              "（3）一次 forward 得到 shape 为 (1, 1, 100, 7) 的检测结果；"
              "（4）按置信度阈值 0.3 过滤，把归一化坐标乘以宽高换算成像素坐标并绘制。核心代码如下：")
    code(doc, '''
net  = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt",
                                "MobileNetSSD_deploy.caffemodel")
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300),
                             (127.5, 127.5, 127.5))
net.setInput(blob)
detections = net.forward()                     # (1, 1, 100, 7)

for i in range(detections.shape[2]):
    confidence = float(detections[0, 0, i, 2])
    if confidence < 0.3:
        continue
    class_id = int(detections[0, 0, i, 1])
    x1 = int(detections[0, 0, i, 3] * W)       # 归一化坐标 -> 像素
    y1 = int(detections[0, 0, i, 4] * H)
    x2 = int(detections[0, 0, i, 5] * W)
    y2 = int(detections[0, 0, i, 6] * H)
    boxes.append([x1, y1, x2 - x1, y2 - y1])
''')

    h(doc, "4.3 任务1-2：改造为摄像头实时检测", level=2)
    para(doc, "把“读一张图”改为“循环读视频帧”：VideoCapture(0) 打开本机摄像头，read 取帧，"
              "镜像翻转后送入上面的检测函数，绘制结果并 imshow 显示；画面左上角显示实时 FPS 与目标个数；"
              "按 q/ESC 退出，按 s 保存截图。除读取方式外，检测与绘制逻辑完全复用。")
    code(doc, '''
cap = cv2.VideoCapture(source)          # source=0 表示本机摄像头，也可传视频文件路径
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    boxes, confs, cids = detect(frame, net, out_names)
    frame = draw(frame, boxes, confs, cids, classes, colors)
    cv2.putText(frame, f"FPS: {fps:.1f}  objects: {len(boxes)}", (10, 30), ...)
    cv2.imshow("YOLOv3 Camera Detection", frame)
    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
        break
''')

    h(doc, "4.4 任务2-1：YOLO26n 预训练模型推理", level=2)
    para(doc, "使用实验素材中的 ultralytics 源码加载 yolo26n.pt，在 CPU 上对 test2.jpg 与 100 张细胞图执行推理，"
              "并保存可视化结果：")
    code(doc, '''
from ultralytics import YOLO
model   = YOLO("yolo26n.pt")
results = model.predict(source=".../images", imgsz=640, conf=0.25,
                        save=True, device="cpu")
''')

    h(doc, "4.5 任务2-2：数据集制作与扩充", level=2)
    para(doc, "提供的标注是 labelme 格式：每张图对应一个 json，shapes 中每个 rectangle 的两个 points 即左上、右下角点。"
              "把标注转成 YOLO 的“每行一个目标”的 txt 格式，转换公式为：")
    code(doc, '''
cx = (x1 + x2) / 2 / W      cy = (y1 + y2) / 2 / H
w  = (x2 - x1) / W          h  = (y2 - y1) / H
# txt 每一行： class_id  cx  cy  w  h   （均为 0~1 的比例）
''')
    para(doc, "数据集划分与扩充策略：先按 8:2 随机划分 100 张原图为 80 张训练图与 20 张验证图，再只对训练集原图做几何增强，"
              "避免同一张图的增强副本同时出现在训练集和验证集而造成数据泄漏。增强方式与标注框的同步变换为：")
    table(doc, ["增强方式", "图像变换", "标注框 (x1,y1,x2,y2) 变换"], [
        ["水平翻转", "image[:, ::-1]", "(W-x2, y1, W-x1, y2)"],
        ["垂直翻转", "image[::-1, :]", "(x1, H-y2, x2, H-y1)"],
        ["旋转 180°", "image[::-1, ::-1]", "(W-x2, H-y2, W-x1, H-y1)"],
        ["顺时针旋转 90°", "cv2.rotate(ROTATE_90_CLOCKWISE)", "(H-y2, x1, H-y1, x2)"],
    ], widths=[3.6, 6.4, 6.4])
    para(doc, "每张训练原图生成“原图 + 3 种增强”共 4 份，训练集由 80 张扩充到 320 张；验证集保持 20 张原图不增强。"
              "转换前会先把框裁剪到图像范围内并过滤宽高小于 2 像素的退化框。")

    h(doc, "4.6 任务2-3：YOLO26n 训练与评估", level=2)
    para(doc, "以 COCO 预训练的 yolo26n.pt 为初始权重，在个人数据集上微调。训练参数如下表，"
              "训练结束后自动在验证集上计算 P、R、mAP50、mAP50-95，并保存最佳权重 best.pt 与各类曲线。")
    table(doc, ["参数", "取值", "说明"], [
        ["model", "yolo26n.pt", "COCO 预训练权重，单类别输出层自动改为 1 类"],
        ["data", "dataset/data.yaml", "1 类（cell），训练集 320 张 / 验证集 20 张"],
        ["epochs / batch", "30 / 16", "CPU 训练，约 1.5 分钟一个 epoch"],
        ["imgsz", "640×640", "与原图 1024×1024 下采样后目标尺寸相匹配"],
        ["device", "cpu", "无 GPU，使用 16 线程"],
        ["数据增强", "框架默认", "mosaic、随机缩放/平移、HSV 颜色扰动等"],
        ["patience / seed", "10 / 0", "验证指标 10 轮不提升则早停，随机种子固定保证可复现"],
    ], widths=[3.2, 3.6, 9.4])

    h(doc, "4.7 实验过程中的问题与处理", level=2)
    para(doc, "问题 1：系统安装的 OpenCV 5.0 已移除 Darknet 与 Caffe 模型导入器，"
              "调用 readNetFromDarknet / readNetFromCaffe 时直接报错“importer has been removed”。"
              "处理：新建独立虚拟环境 .venv 并安装 opencv-python 4.11.0，专门运行 DNN 脚本，不影响系统环境。")
    para(doc, "问题 2：OpenCV 在中文路径（DNN目标检测任务(1)）下 imread 返回 None，模型文件也无法按路径读取。"
              "处理：把模型与测试图片复制到 experiment4/models 的纯英文路径下，脚本统一使用副本。")
    para(doc, "问题 3：CPU 训练速度有限。处理：选择最轻量的 YOLO26n、输入 640、batch 16，"
              "并先用 1 个 epoch 做耗时测试，再确定 30 个 epoch 的完整训练方案。")

    # ================= 五、实验结果与分析 =================
    h(doc, "五、实验结果与分析")

    h(doc, "5.1 YOLOv3 与 SSD 的检测结果", level=2)
    image(doc, OUT / "yolov3_result.jpg", "图 5-1 YOLOv3 检测结果：bicycle 99.6%、dog 96.4%、dog 61.5%")
    image(doc, OUT / "ssd_result.jpg", "图 5-2 MobileNet-SSD 检测结果：bicycle 100.0%、dog 99.9%")
    table(doc, ["模型", "输入尺寸", "检测结果", "单张耗时(CPU)", "特点"], [
        ["YOLOv3", "416×416", "bicycle、dog×2（共 3 个）", "约 302 ms", "多尺度输出，遮挡/较小目标也能检出，速度较慢"],
        ["MobileNet-SSD", "300×300", "bicycle、dog（共 2 个）", "约 21.5 ms", "轻量快速，但小目标与重叠目标易漏检"],
    ], widths=[3, 2.2, 4.8, 3, 4.6])
    para(doc, "两张结果图都能正确框出自行车与狗的主目标，说明两种模型的前处理与后处理实现正确。"
              "差异在于：YOLOv3 通过 3 个尺度分别负责大/中/小目标，因此额外找出了被自行车遮挡的第三只目标；"
              "SSD 只使用 300×300 输入与单次前向，对遮挡目标和远景小目标更保守。速度上 SSD 约为 YOLOv3 的 14 倍，"
              "更适合实时场景。")

    h(doc, "5.2 摄像头实时检测结果", level=2)
    image(doc, OUT / "camera_yolov3_demo.jpg", "图 5-3 摄像头实时检测画面（宿舍场景，本帧无可识别目标）", width=10)
    para(doc, "把代码改为逐帧读取后，程序可以稳定打开本机摄像头并实时显示画面与 FPS。实测 YOLOv3 单帧推理约 0.3 s，"
              "画面约 3 FPS；SSD 单帧约 0.02 s，画面可达到 20 FPS 以上，交互更流畅。"
              "本帧为宿舍室内场景，其中没有 COCO 类别中的目标，因此未画出检测框——这也说明模型只识别训练时见过的类别，"
              "不会“强行”输出结果。若要验证检测效果，可把镜头对准人、椅子、瓶子、笔记本电脑等常见物体。")

    h(doc, "5.3 YOLO26n 预训练模型推理结果", level=2)
    image(doc, OUT / "task1_yolo26_test2" / "test2.jpg", "图 5-4 YOLO26n 在通用图片上的推理结果", width=12)
    para(doc, "在通用图片上，YOLO26n 正确输出 bicycle、dog×2、bottle、sports ball 共 5 个目标，"
              "单张推理约 95 ms（CPU，640 输入），满足实时要求，说明预训练权重与推理流程正常。")
    image(doc, OUT / "task1_yolo26_cells" / "0_20480_0_0.jpg",
          "图 5-5 YOLO26n 在细胞图像上的误检（teddy bear 0.49、bed 0.31）", width=11)
    para(doc, "把同一模型直接用在细胞切片图上时，模型只能把细胞误判为 teddy bear、bed 等 COCO 类别，"
              "一张图仅输出少量框且位置随意。原因是 COCO 的 80 类中不存在“细胞”，模型没有可用的类别特征。"
              "这说明：预训练模型只能用于其训练过的类别，要检测新类别必须用自己的标注数据微调。")

    h(doc, "5.4 数据集转换与扩充结果", level=2)
    image(doc, OUT / "dataset_gt_preview.jpg", "图 5-6 转换后的 YOLO 标注检查（绿色框与细胞轮廓吻合）", width=11)
    table(doc, ["数据集", "图像数", "标注框数", "说明"], [
        ["原始数据", "100", "4852", "1024×1024，单类 c（细胞），单图 7~86 个目标"],
        ["训练集（扩充后）", "320", "15556", "80 张原图 × 4（原图 + 水平翻转 + 垂直翻转 + 旋转 90°）"],
        ["验证集", "20", "953", "原始图像，不做增强，不参与训练"],
    ], widths=[4, 2.4, 2.4, 7.4])
    para(doc, "图 5-6 中绿色框与细胞外轮廓基本贴合，说明坐标换算与增强时的框变换正确；"
              "训练集扩充到 4 倍后，模型能见到更多方向与位置的细胞，有助于减轻过拟合。")

    h(doc, "5.5 训练过程与指标", level=2)
    if m:
        para(doc, f"30 个 epoch 训练完成，最佳结果出现在第 {m['epoch']} 个 epoch："
                  f"Precision = {m['P']:.4f}，Recall = {m['R']:.4f}，"
                  f"mAP50 = {m['mAP50']:.4f}，mAP50-95 = {m['mAP50_95']:.4f}。")
    image(doc, RUN / "results.png", "图 5-7 训练/验证损失与 mAP 曲线", width=15)
    image(doc, RUN / "BoxPR_curve.png", "图 5-8 验证集 Precision-Recall 曲线", width=10)
    image(doc, RUN / "confusion_matrix_normalized.png", "图 5-9 归一化混淆矩阵（cell 与背景）", width=9)
    para(doc, "从曲线看：训练与验证损失持续下降，mAP50 在前 10 个 epoch 快速上升到 0.85 以上，"
              "之后缓慢提升并在 0.92 附近趋于稳定；mAP50-95 同步上升到 0.61 左右。"
              "验证曲线没有明显反弹，说明 4 倍数据扩充与框架自带增强有效抑制了过拟合。"
              "混淆矩阵中 cell 与背景的对角线占比高，主要误差来自细胞粘连、相互遮挡以及切片边缘被截断的细胞。")

    h(doc, "5.6 训练后模型的推理效果", level=2)
    para(doc, "用 best.pt 在 20 张验证图（953 个真实框）上按 conf=0.25 推理，共输出 1079 个预测框，"
              "整体数量略多于真实框，说明低阈值下存在少量重复框与误检；逐图对比见下表。")
    image(doc, OUT / "task2_predict_val" / "0_20480_0_1024.jpg",
          "图 5-10 训练后的 YOLO26n 在验证集上的检测结果（cell，置信度多为 0.7~0.9）", width=11)
    table(doc, ["验证图像", "真实框", "预测框", "差值"], [
        ["0_20480_0_1024", 76, 86, "+10"],
        ["0_20480_0_512", 56, 77, "+21"],
        ["0_20480_512_1024", 63, 70, "+7"],
        ["0_20480_512_512", 51, 61, "+10"],
        ["2048_36864_0_512", 25, 32, "+7"],
        ["2048_36864_1024_1024", 46, 42, "-4"],
        ["2048_36864_512_1024", 32, 41, "+9"],
        ["4096_12288_1024_512", 53, 66, "+13"],
        ["4096_28672_1024_512", 45, 44, "-1"],
        ["6144_10240_0_512", 47, 48, "+1"],
        ["6144_10240_1024_0", 62, 53, "-9"],
        ["6144_10240_512_1024", 64, 59, "-5"],
        ["6144_12288_0_0", 52, 67, "+15"],
        ["6144_14336_0_0", 64, 74, "+10"],
        ["6144_14336_1024_0", 85, 92, "+7"],
        ["6144_32768_512_0", 41, 50, "+9"],
        ["6144_32768_512_512", 47, 57, "+10"],
        ["6144_53248_512_0", 10, 15, "+5"],
        ["6144_57344_0_1024", 15, 23, "+8"],
        ["6144_57344_1024_1024", 19, 22, "+3"],
        ["合计", 953, 1079, "+126"],
    ], widths=[6.4, 3, 3, 3])
    para(doc, "与图 5-5 的预训练结果对比：同一张图，COCO 预训练模型只输出 1~9 个错误类别框；"
              "微调后的 YOLO26n 稳定输出几十个 cell 框，置信度多在 0.7 以上，框与细胞轮廓贴合。"
              "说明在个人数据集上训练是必要且有效的。对于粘连成团的细胞，模型偶尔会把两个细胞合并为一个框，"
              "或在低阈值下重复检出同一细胞，这也是预测框略多于真实框的主要原因。")

    h(doc, "5.7 三种模型综合对比", level=2)
    table(doc, ["模型", "参数量", "输入", "CPU 单张耗时", "本实验表现"], [
        ["YOLOv3（COCO 80 类）", "约 61.5 M", "416×416", "约 302 ms", "通用目标检测，3 个目标，无需训练"],
        ["MobileNet-SSD（VOC 20 类）", "约 5.8 M", "300×300", "约 21.5 ms", "通用目标检测，2 个目标，速度最快"],
        ["YOLO26n（个人数据集）", "2.39 M", "640×640", "约 37 ms", "细胞单类检测，mAP50=0.921，mAP50-95=0.613"],
    ], widths=[4.4, 2.6, 2.4, 3.2, 5.6])
    para(doc, "注：前两行为通用类别上的定性结果，第三行为单类别数据集上的定量指标，指标含义不同，"
              "仅用于对比模型规模与推理速度。")

    # ================= 六、实验总结 =================
    h(doc, "六、实验总结")
    para(doc, "1. OpenCV DNN 把“模型读取—预处理—前向计算—后处理”抽象成统一流程，"
              "因此可以用同一套代码部署 YOLOv3 与 SSD；差别主要在读取函数、输入尺寸与后处理方式。"
              "掌握这一点后，换模型只需要改动少量代码，具备很好的工程通用性。")
    para(doc, "2. 摄像头实时检测并不需要重写检测逻辑，只要把“读图”换成“逐帧读视频”，"
              "并注意镜像、FPS 统计与按键退出即可。实测 YOLOv3 约 3 FPS、SSD 20 FPS 以上，"
              "说明在 CPU 场景下选择轻量模型是实时性的关键。")
    para(doc, "3. 预训练模型的类别是固定的：COCO 模型会把细胞误判成 teddy bear。"
              "要检测新类别，必须准备标注数据并微调；labelme 转 YOLO、按原图划分训练/验证集、"
              "用几何增强扩充数据，是保证小数据集训练效果与评估可信度的三个关键步骤。")
    para(doc, "4. YOLO26n 只有 2.39 M 参数，在 CPU 上约 1.5 分钟/epoch 就能完成 320 张图的训练，"
              "30 epoch 后验证集 mAP50 达 0.921、mAP50-95 达 0.613，说明轻量模型配合合理的数据增强，"
              "完全可以在无 GPU 的普通笔记本上完成一个可用的检测任务。")
    para(doc, "5. 不足与改进方向：（1）细胞图像 1024×1024，输入缩放到 640 会损失小目标细节，"
              "可尝试 800/960 输入或切片（tile）推理；（2）粘连细胞容易合并，可加入实例分割或更强的 NMS 策略；"
              "（3）目前只用了 100 张原图，扩充到更多原始图像比单纯几何增强更有效；"
              "（4）后续可换成 YOLO26s 或更大模型，在有 GPU 的条件下比较精度与速度的权衡。")

    # ================= 七、附录 =================
    h(doc, "七、附录：代码与文件清单")
    table(doc, ["脚本 / 文件", "作用"], [
        ["01_opencv_dnn_yolov3.py", "OpenCV DNN + YOLOv3 单张图像检测"],
        ["02_opencv_dnn_ssd.py", "OpenCV DNN + MobileNet-SSD 单张图像检测"],
        ["03_yolov3_camera.py / 04_ssd_camera.py", "摄像头/视频实时检测（q 退出，s 截图）"],
        ["05_yolo26_predict.py", "YOLO26n 预训练模型推理"],
        ["06_prepare_dataset.py", "labelme→YOLO 转换、8:2 划分、4 倍几何增强"],
        ["07_train_yolo26.py", "YOLO26n 训练 + 验证，输出 best.pt、results.csv 与曲线"],
        ["08_predict_trained.py", "用 best.pt 对验证集推理并统计预测框数量"],
        ["09_yolo26_camera.py", "训练权重接入摄像头的实时检测"],
        ["dataset/", "images/train、images/val、labels、data.yaml"],
        ["output/", "所有结果图、训练日志、训练权重与报告配图"],
    ], widths=[8, 9])
    para(doc, "复现顺序：在 experiment4 目录下，先用 .venv 中的 Python 运行 01/02，"
              "再用系统 Python 依次运行 06 → 07 → 08；YOLO26 相关脚本会自动把实验素材中的 ultralytics 源码加入 sys.path。"
              "首次运行会下载 yolo26n.pt（已存放在 experiment4/models/yolo26n.pt）。")


def main():
    doc = Document()
    set_font(doc)
    for s in doc.sections:
        s.left_margin = s.right_margin = Cm(2.2)
        s.top_margin = Cm(2.2)
        s.bottom_margin = Cm(2.2)
    build(doc)
    doc.save(str(REPORT))
    print(f"报告已生成：{REPORT}")


if __name__ == "__main__":
    main()
