import os
import cv2
import torch
from ultralytics import YOLO

# 配置
MODEL_PATH = "weights/best-yolov8n.pt"
INPUT_IMAGE = r"D:\PycharmProject(D)\外包项目\外包2025\250507-基于YOLO的教室人数统计系统\HeadDatasets300\test\images\20241016_144318_frame_004.jpg"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_IMAGE = os.path.join(OUTPUT_DIR, "custom_vis_numbered.jpg")

# 加载模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(MODEL_PATH)

# 读取并推理
img = cv2.imread(INPUT_IMAGE)
results = model(img, imgsz=640, conf=0.1, iou=0.3)[0]
boxes = results.boxes.xyxy.cpu().numpy()  # shape: (N, 4)

# 参数：字体、颜色、线宽
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
thickness = 1
box_color = (0, 255, 0)  # BGR 绿色
text_color = (255, 0, 0)  # 与框同色
total_color = (0, 0, 255)  # BGR 红色，用于“Total”

# 在每个框左上角画框和编号
for idx, (x1, y1, x2, y2) in enumerate(boxes, start=1):
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
    # 画框
    cv2.rectangle(img, (x1, y1), (x2, y2), box_color, thickness)
    # 标注序号：放在框的左上角，坐标向上偏一点以免遮框
    text = str(idx)
    ((text_w, text_h), _) = cv2.getTextSize(text, font, font_scale, thickness)
    text_org = (x1, y1 - 5 if y1 - 5 > text_h else y1 + text_h + 5)
    cv2.putText(
        img, text, text_org, font, font_scale,
        text_color, thickness, lineType=cv2.LINE_AA
    )

# 在整图左上方显示总数
total_person = len(boxes)
total_text = f"Total: {total_person}"
# 测量文本尺寸，方便画背景框（可选）
((w, h), _) = cv2.getTextSize(total_text, font, font_scale, thickness)
# 画一个半透明背景矩形（可选增强可读性）
cv2.rectangle(img, (5, 5), (5 + w + 10, 5 + h + 10), (0, 0, 0), cv2.FILLED)

# 再写字
cv2.putText(img, total_text, (10, 5 + h + 2), font, font_scale, total_color, thickness + 1, lineType=cv2.LINE_AA)

# 保存结果
cv2.imwrite(OUTPUT_IMAGE, img)
cv2.imshow("det", img)
cv2.waitKey(0)
print(f"Saved visualization to {OUTPUT_IMAGE}")
