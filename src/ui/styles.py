"""
Стили и темы UI
"""


def get_dark_theme():
    """Современная темная тема с улучшенным дизайном"""
    return """
        QMainWindow {
            background-color: #0f0f1a;
        }
        QWidget {
            background-color: #16162a;
            color: #e2e8f0;
            font-family: 'Segoe UI', 'SF Pro Display', -apple-system, Arial, sans-serif;
            font-size: 13px;
        }
        QGroupBox {
            border: 1px solid rgba(102, 126, 234, 0.3);
            border-radius: 12px;
            margin-top: 12px;
            padding: 15px;
            padding-top: 25px;
            font-weight: 600;
            font-size: 14px;
            background-color: rgba(22, 22, 42, 0.8);
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 15px;
            padding: 0 12px;
            color: #a78bfa;
            background-color: #16162a;
            border-radius: 4px;
        }
        QPushButton {
            background-color: #2d2d4a;
            border: 1px solid rgba(102, 126, 234, 0.4);
            padding: 8px 16px;
            border-radius: 8px;
            font-weight: 600;
            color: #e2e8f0;
        }
        QPushButton:hover {
            background-color: #3d3d5c;
            border-color: rgba(102, 126, 234, 0.7);
        }
        QPushButton:pressed {
            background-color: #1e1e3a;
        }
        QPushButton:disabled {
            background-color: #1a1a2e;
            color: #4a5568;
            border-color: rgba(74, 85, 104, 0.3);
        }
        QComboBox {
            background-color: #2d2d4a;
            border: 1px solid rgba(102, 126, 234, 0.3);
            padding: 8px 12px;
            border-radius: 8px;
            min-width: 150px;
        }
        QComboBox:hover {
            border-color: rgba(102, 126, 234, 0.6);
        }
        QComboBox:drop-down {
            border: none;
            padding-right: 10px;
        }
        QComboBox::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 6px solid #a78bfa;
            margin-right: 8px;
        }
        QComboBox QAbstractItemView {
            background-color: #2d2d4a;
            border: 1px solid rgba(102, 126, 234, 0.3);
            border-radius: 8px;
            selection-background-color: #667eea;
            padding: 4px;
        }
        QLineEdit {
            background-color: #2d2d4a;
            border: 1px solid rgba(102, 126, 234, 0.3);
            padding: 8px 12px;
            border-radius: 8px;
        }
        QLineEdit:focus {
            border-color: #667eea;
        }
        QTextEdit {
            background-color: #0f0f1a;
            border: 1px solid rgba(102, 126, 234, 0.2);
            border-radius: 10px;
            padding: 12px;
            selection-background-color: #667eea;
        }
        QTextEdit:focus {
            border-color: rgba(102, 126, 234, 0.5);
        }
        QProgressBar {
            text-align: center;
            border: none;
            border-radius: 10px;
            background-color: #1a1a2e;
            height: 20px;
        }
        QProgressBar::chunk {
            border-radius: 10px;
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #667eea, stop:0.5 #764ba2, stop:1 #a78bfa);
        }
        QScrollBar:vertical {
            background: #1a1a2e;
            width: 10px;
            border-radius: 5px;
            margin: 2px;
        }
        QScrollBar::handle:vertical {
            background: qlineargradient(y1:0, y2:1,
                stop:0 #667eea, stop:1 #764ba2);
            border-radius: 5px;
            min-height: 30px;
        }
        QScrollBar::handle:vertical:hover {
            background: qlineargradient(y1:0, y2:1,
                stop:0 #7c8ff0, stop:1 #8b5fcf);
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        QScrollBar:horizontal {
            background: #1a1a2e;
            height: 10px;
            border-radius: 5px;
            margin: 2px;
        }
        QScrollBar::handle:horizontal {
            background: qlineargradient(x1:0, x2:1,
                stop:0 #667eea, stop:1 #764ba2);
            border-radius: 5px;
            min-width: 30px;
        }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            width: 0px;
        }
        QSpinBox {
            background-color: #2d2d4a;
            border: 1px solid rgba(102, 126, 234, 0.3);
            padding: 6px 10px;
            border-radius: 8px;
            min-width: 70px;
        }
        QSpinBox:hover {
            border-color: rgba(102, 126, 234, 0.6);
        }
        QSpinBox::up-button, QSpinBox::down-button {
            background-color: transparent;
            border: none;
            width: 16px;
        }
        QSpinBox::up-arrow {
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-bottom: 5px solid #a78bfa;
        }
        QSpinBox::down-arrow {
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 5px solid #a78bfa;
        }
        QCheckBox {
            spacing: 10px;
        }
        QCheckBox::indicator {
            width: 20px;
            height: 20px;
            border-radius: 6px;
            border: 2px solid rgba(102, 126, 234, 0.4);
            background-color: #1a1a2e;
        }
        QCheckBox::indicator:hover {
            border-color: rgba(102, 126, 234, 0.7);
        }
        QCheckBox::indicator:checked {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #667eea, stop:1 #764ba2);
            border-color: #667eea;
        }
        QMenuBar {
            background-color: #16162a;
            color: #e2e8f0;
            border-bottom: 1px solid rgba(102, 126, 234, 0.2);
            padding: 4px;
        }
        QMenuBar::item {
            padding: 6px 12px;
            border-radius: 6px;
        }
        QMenuBar::item:selected {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #667eea, stop:1 #764ba2);
        }
        QMenu {
            background-color: #2d2d4a;
            border: 1px solid rgba(102, 126, 234, 0.3);
            border-radius: 8px;
            padding: 4px;
        }
        QMenu::item {
            padding: 8px 20px;
            border-radius: 4px;
        }
        QMenu::item:selected {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #667eea, stop:1 #764ba2);
        }
        QLabel {
            color: #e2e8f0;
        }
        QToolTip {
            background-color: #2d2d4a;
            color: #e2e8f0;
            border: 1px solid rgba(102, 126, 234, 0.4);
            border-radius: 6px;
            padding: 6px 10px;
        }
        QSplitter::handle {
            background: rgba(102, 126, 234, 0.3);
            height: 3px;
        }
        QSplitter::handle:hover {
            background: #667eea;
        }
    """
