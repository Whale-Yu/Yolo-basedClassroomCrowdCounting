# style_ui.py

def apply_styles(app):
    """
    为整个应用程序设置现代化美化样式 (QSS)。

    用法: 在主程序 QApplication 创建后调用 apply_styles(app)
    """
    qss = """
    /* 主背景及通用字体 */
    QWidget {
        background-color: #fafbfd;
        font-family: 'Roboto', 'Segoe UI', sans-serif;
        font-size: 11pt;
    }

    /* 卡片式 QGroupBox */
    QGroupBox {
        background: white;
        border:  1px solid #d0d7de;
        border-radius: 12px;
        margin-top: 20px;
        padding: 16px;
        box-shadow: 0px 2px 8px rgba(0, 0, 0, 0.1);
    }
    QGroupBox:title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 4px;
        color: #222;
        font-size: 12pt;
        font-weight: 600;
    }

    /* 按钮现代扁平风 */
    QPushButton {
        background-color: #0078d4;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 16px;
        min-width: 80px;
        transition: background 0.2s ease;
    }
    QPushButton:hover {
        background-color: #005a9e;
    }
    QPushButton:pressed {
        background-color: #004578;
    }

    /* 下拉框与数值输入 */
    QComboBox, QDoubleSpinBox {
        background: white;
        border: 1px solid #ccc;
        border-radius: 6px;
        padding: 4px 8px;
    }

    /* 文本标签 */
    QLabel {
        color: #333;
    }

    /* 文本框和日志区 */
    QTextEdit {
        background: white;
        border: 1px solid #ddd;
        border-radius: 6px;
    }

    /* 滑动条 */
    QSlider::groove:horizontal {
        height: 6px;
        background: #e1e1e1;
        border-radius: 3px;
    }
    QSlider::handle:horizontal {
        width: 14px;
        height: 14px;
        background: #0078d4;
        margin: -4px 0;
        border-radius: 7px;
    }
"""
    app.setStyleSheet(qss)
