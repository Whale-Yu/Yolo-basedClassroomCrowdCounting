import os
import sys
import cv2
import torch
from datetime import datetime
from ultralytics import YOLO
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QComboBox,
    QDoubleSpinBox, QFileDialog, QTextEdit, QHBoxLayout,
    QVBoxLayout, QGroupBox, QGridLayout, QSizePolicy, QColorDialog
)

from main_style_ui import apply_styles


# 模型目录及默认模型
MODEL_DIR = "weights"
DEFAULT_MODEL_NAME = "test-0508.pt"
DEFAULT_MODEL = os.path.join(MODEL_DIR, DEFAULT_MODEL_NAME)

class VideoThread(QThread):
    change_pixmap = pyqtSignal(object)
    log_signal = pyqtSignal(str)

    def __init__(self, src, model, conf_func, iou_func, box_color_func, number_color_func):
        super().__init__()
        self.src = src
        self.model = model
        self.conf_func = conf_func
        self.iou_func = iou_func
        self.box_color_func = box_color_func
        self.number_color_func = number_color_func
        self._run_flag = True
        self.frame_count = 0  # 用于跳帧


    def run(self):
        cap = cv2.VideoCapture(self.src)
        if not cap.isOpened():
            self.log_signal.emit("无法打开视频源")
            return
        self.log_signal.emit(f"开始捕获: {self.src}")
        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                break
            # 跳帧处理：只处理每隔2帧，减轻负载
            self.frame_count += 1
            if self.frame_count % 2 != 0:
                continue
            # 降低推理分辨率，加快速度
            height, width = frame.shape[:2]
            scale = 640 / max(width, height)
            small = cv2.resize(frame, None, fx=scale, fy=scale)
            conf = float(self.conf_func())
            iou = float(self.iou_func())
            results = self.model(small, imgsz=640, conf=conf, iou=iou)[0]
            # 恢复原尺寸框坐标
            boxes = results.boxes.xyxy.cpu().numpy()
            boxes = (boxes / scale).astype(int)
            # 可视化
            box_color = tuple(map(int, self.box_color_func()))
            number_color = tuple(map(int, self.number_color_func()))
            for idx, (x1, y1, x2, y2) in enumerate(boxes, start=1):
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, str(idx), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, number_color, 2)
            total_text = f"Total: {len(boxes)}"
            (w, h), _ = cv2.getTextSize(total_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (5,5), (5+w+10, 5+h+10), (0,0,0), cv2.FILLED)
            cv2.putText(frame, total_text, (10, 5+h), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,0,255), 2)
            self.log_signal.emit(f"检测到 {len(boxes)} 个目标")

            # 发射到主线程显示
            self.change_pixmap.emit(frame)
        cap.release()
        self.log_signal.emit("捕获结束")

    def stop(self):
        self._run_flag = False
        self.wait()

class DetectApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于YOLO的教室人数统计系统")
        self.model = None
        self.thread = None
        # 默认颜色 (B, G, R)
        self.box_color_rgb = (0, 255, 0)
        self.number_color_rgb = (255, 0, 0)
        self._build_ui()
        # 自动加载默认模型
        if os.path.isfile(DEFAULT_MODEL):
            idx = self.model_combo.findText(DEFAULT_MODEL)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
                self.load_model()

    def _build_ui(self):
        # 左侧控制区，包含四个分区：模型、阈值、可视化、操作
        left_layout = QVBoxLayout()

        # 模型设置分区
        group_model = QGroupBox("模型设置")
        layout_model = QHBoxLayout()
        layout_model.addWidget(QLabel("选择模型:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(self._get_models())
        layout_model.addWidget(self.model_combo)
        btn_load = QPushButton("加载模型")
        btn_load.clicked.connect(self.load_model)
        layout_model.addWidget(btn_load)
        group_model.setLayout(layout_model)

        # 阈值设置分区
        group_threshold = QGroupBox("阈值设置")
        layout_th = QGridLayout()
        layout_th.addWidget(QLabel("置信度:"), 0, 0)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0,1); self.conf_spin.setSingleStep(0.05); self.conf_spin.setValue(0.1)
        layout_th.addWidget(self.conf_spin, 0, 1)
        layout_th.addWidget(QLabel("IoU:"), 1, 0)
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0,1); self.iou_spin.setSingleStep(0.05); self.iou_spin.setValue(0.3)
        layout_th.addWidget(self.iou_spin, 1, 1)
        group_threshold.setLayout(layout_th)

        # 可视化设置分区
        group_vis = QGroupBox("可视化设置")
        layout_vis = QGridLayout()
        btn_box_color = QPushButton("框颜色")
        btn_box_color.clicked.connect(self.choose_box_color)
        layout_vis.addWidget(btn_box_color, 0, 0)
        self.box_color_label = QLabel()
        self.box_color_label.setFixedSize(30,20)
        self.box_color_label.setStyleSheet(f"background-color: rgb({self.box_color_rgb[2]}, {self.box_color_rgb[1]}, {self.box_color_rgb[0]}); border:1px solid #000;")
        layout_vis.addWidget(self.box_color_label, 0, 1)
        btn_number_color = QPushButton("编号颜色")
        btn_number_color.clicked.connect(self.choose_number_color)
        layout_vis.addWidget(btn_number_color, 1, 0)
        self.number_color_label = QLabel()
        self.number_color_label.setFixedSize(30,20)
        self.number_color_label.setStyleSheet(f"background-color: rgb({self.number_color_rgb[2]}, {self.number_color_rgb[1]}, {self.number_color_rgb[0]}); border:1px solid #000;")
        layout_vis.addWidget(self.number_color_label, 1, 1)
        group_vis.setLayout(layout_vis)

        # 操作按钮分区
        group_action = QGroupBox("操作")
        layout_act = QGridLayout()
        btn_img = QPushButton("打开图片")
        btn_img.clicked.connect(self.open_image)
        layout_act.addWidget(btn_img, 0, 0)
        btn_vid = QPushButton("开始视频")
        btn_vid.clicked.connect(self.start_video)
        layout_act.addWidget(btn_vid, 0, 1)
        btn_cam = QPushButton("开始摄像头")
        btn_cam.clicked.connect(self.start_camera)
        layout_act.addWidget(btn_cam, 1, 0)
        btn_stop = QPushButton("停止")
        btn_stop.clicked.connect(self.stop_capture)
        layout_act.addWidget(btn_stop, 1, 1)
        group_action.setLayout(layout_act)

        left_layout.addWidget(group_model)
        left_layout.addWidget(group_threshold)
        left_layout.addWidget(group_vis)
        left_layout.addWidget(group_action)

        # 显示区
        self.display = QLabel()
        self.display.setAlignment(Qt.AlignCenter)
        self.display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.display.setMinimumSize(900, 680)

        # 日志区
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)

        # 主布局: 左侧设置+按钮，右侧显示区；底部日志输出
        top_layout = QHBoxLayout()
        top_layout.addLayout(left_layout)
        top_layout.addWidget(self.display, 1)

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addWidget(QLabel("日志输出:"))
        main_layout.addWidget(self.log_box)

        self.setLayout(main_layout)

    def _get_models(self):
        models = []
        if os.path.isdir(MODEL_DIR):
            files = sorted(os.listdir(MODEL_DIR))
            if DEFAULT_MODEL_NAME in files:
                models.append(os.path.join(MODEL_DIR, DEFAULT_MODEL_NAME))
                files.remove(DEFAULT_MODEL_NAME)
            for f in files:
                if f.endswith('.pt'):
                    models.append(os.path.join(MODEL_DIR, f))
        return models

    def log(self, msg):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_box.append(f"[{ts}] {msg}")

    def load_model(self):
        path = self.model_combo.currentText()
        if os.path.isfile(path):
            self.model = YOLO(path)
            self.log(f"加载模型: {path}")
        else:
            self.log(f"模型未找到: {path}")

    def choose_box_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.box_color_rgb = (color.blue(), color.green(), color.red())
            self.box_color_label.setStyleSheet(
                f"background-color: rgb({color.red()}, {color.green()}, {color.blue()}); border:1px solid #000;"
            )

    def choose_number_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.number_color_rgb = (color.blue(), color.green(), color.red())
            self.number_color_label.setStyleSheet(
                f"background-color: rgb({color.red()}, {color.green()}, {color.blue()}); border:1px solid #000;"
            )

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.jpg *.png)")
        if not path or not self.model:
            return
        self.log(f"图片: {path}")
        img = cv2.imread(path)
        results = self.model(img, imgsz=1280,
                              conf=float(self.conf_spin.value()),
                              iou=float(self.iou_spin.value()))[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        for idx, (x1, y1, x2, y2) in enumerate(boxes, start=1):
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cv2.rectangle(img, (x1, y1), (x2, y2), self.box_color_rgb, thickness+1)
            cv2.putText(img, str(idx), (x1, max(y1 - 5, 0)), font, scale, self.number_color_rgb, thickness, cv2.LINE_AA)
        total_text = f"Total: {len(boxes)}"
        ((w, h), _) = cv2.getTextSize(total_text, font, scale, thickness)
        cv2.rectangle(img, (5, 5), (5 + w + 10, 5 + h + 10), (0, 0, 0), cv2.FILLED)
        cv2.putText(img, total_text, (10, 5 + h), font, scale, (0, 0, 255), thickness+1, cv2.LINE_AA)
        self._update_display(img)
        self.log(f"检测到 {len(boxes)} 个目标")

    def start_video(self):
        if not self.model:
            return self.log("请先加载模型。")
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Videos (*.mp4 *.avi)")
        if not path:
            return
        self._start_thread(path)

    def start_camera(self):
        if not self.model:
            return self.log("请先加载模型。")
        self._start_thread(0)

    def _start_thread(self, src):
        self.stop_capture()
        self.thread = VideoThread(src, self.model,
                                  conf_func=self.conf_spin.value,
                                  iou_func=self.iou_spin.value,
                                  box_color_func=self.get_box_color,
                                  number_color_func=self.get_number_color)
        self.thread.change_pixmap.connect(self._update_display)
        self.thread.log_signal.connect(self.log)
        self.thread.start()

    def stop_capture(self):
        if self.thread:
            self.thread.stop()
            self.thread = None

    def get_box_color(self):
        return self.box_color_rgb

    def get_number_color(self):
        return self.number_color_rgb

    def _update_display(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(self.display.size(), Qt.KeepAspectRatio)
        self.display.setPixmap(pix)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    apply_styles(app)         # ← 美化
    win = DetectApp()
    win.show()
    sys.exit(app.exec_())