import os
from datetime import datetime
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QFrame, QPushButton, QSizePolicy, QGraphicsDropShadowEffect,
                             QApplication, QGridLayout, QDialog, QListWidget,
                             QListWidgetItem, QComboBox, QProgressBar, QMessageBox, QAbstractItemView)
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QPropertyAnimation, QEasingCurve, QPoint, QThread, QRect
from PyQt6.QtGui import QFont, QColor, QPalette, QPixmap, QPainter, QPen, QBrush, QLinearGradient
import qtawesome as qta
from model_manager import ModelManager, MODEL_CATALOG


def build_splash_pixmap(width: int = 720, height: int = 420) -> QPixmap:
    """Draws the EquipAI splash screen programmatically (no external assets needed)."""
    from PyQt6.QtCore import Qt as _Qt
    pm = QPixmap(width, height)
    pm.fill(QColor("#080e18"))
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    grad = QLinearGradient(0, 0, 0, height)
    grad.setColorAt(0.0, QColor("#0c1a2e"))
    grad.setColorAt(1.0, QColor("#050b14"))
    painter.fillRect(0, 0, width, height, QBrush(grad))

    accent_y = int(height * 0.56)
    painter.setPen(QPen(QColor("#06b6d4"), 3))
    painter.drawLine(int(width * 0.5 - 110), accent_y, int(width * 0.5 + 110), accent_y)

    title_font = QFont("Inter", 32, QFont.Weight.Bold)
    painter.setFont(title_font)
    painter.setPen(QColor("#f1f5f9"))
    painter.drawText(QRect(0, int(height * 0.30), width, 90),
                     _Qt.AlignmentFlag.AlignCenter, "EquipAI Enterprise")
    painter.drawText(QRect(0, int(height * 0.46), width, 60),
                     _Qt.AlignmentFlag.AlignCenter | _Qt.AlignmentFlag.AlignTop, "v4.0")

    sub_font = QFont("Inter", 13)
    painter.setFont(sub_font)
    painter.setPen(QColor("#94a3b8"))
    painter.drawText(QRect(0, int(height * 0.62), width, 40),
                     _Qt.AlignmentFlag.AlignCenter,
                     "Industrial Maintenance & Troubleshooting Intelligence")
    painter.end()
    return pm

class ImagePreviewItem(QFrame):
    remove_requested = pyqtSignal()

    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setFixedSize(70, 70)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.bg = QFrame(self)
        self.bg.setStyleSheet(f"""
            QFrame {{
                background-image: url({image_path.replace('\\', '/')});
                background-position: center;
                background-repeat: no-repeat;
                border-radius: 12px;
                border: 2px solid #1e293b;
            }}
        """)
        self.bg.setFixedSize(70, 70)
        
        # Overlay remove button
        self.btn_remove = QPushButton(self)
        self.btn_remove.setIcon(qta.icon("fa5s.times-circle", color="#ef4444"))
        self.btn_remove.setStyleSheet("background: #0f172a; border-radius: 10px; border: 1px solid #1e293b;")
        self.btn_remove.setFixedSize(20, 20)
        self.btn_remove.move(48, 2)
        self.btn_remove.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_remove.clicked.connect(self.remove_requested.emit)

class ChatBubble(QFrame):
    feedback_clicked = pyqtSignal(int)

    def __init__(self, text, sender="user", image_path=None, parent=None):
        super().__init__(parent)
        self.sender = sender
        self.text = text
        
        # Full-width outer layout with generous side padding
        main_layout = QHBoxLayout(self)
        if sender == "user":
            main_layout.setContentsMargins(60, 6, 20, 6)
        else:
            main_layout.setContentsMargins(20, 6, 60, 6)
        main_layout.setSpacing(14)
        self.setStyleSheet("background: transparent; border: none;")

        # ---- Avatar ----
        self.avatar = QLabel()
        self.avatar.setFixedSize(34, 34)
        if sender == "user":
            self.avatar.setPixmap(qta.icon("fa5s.user-circle", color="#38bdf8").pixmap(22, 22))
            self.avatar.setStyleSheet("background-color: #1e293b; border: 1px solid #334155; border-radius: 17px; padding: 5px;")
        else:
            self.avatar.setPixmap(qta.icon("fa5s.robot", color="#06b6d4").pixmap(20, 20))
            self.avatar.setStyleSheet("background-color: #0c4a6e; border: 1px solid #0891b2; border-radius: 17px; padding: 6px;")

        # ---- Text column (expands to fill remaining space) ----
        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(6)

        if image_path:
            img_label = QLabel()
            pixmap = QPixmap(image_path).scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            img_label.setPixmap(pixmap)
            img_label.setStyleSheet("border-radius: 12px; border: 1px solid #334155;")
            text_col.addWidget(img_label)

        self.label = QLabel(text)
        self.label.setWordWrap(True)
        self.label.setTextFormat(Qt.TextFormat.MarkdownText)
        self.label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse |
            Qt.TextInteractionFlag.LinksAccessibleByMouse
        )

        if sender == "user":
            self.label.setStyleSheet("""
                QLabel {
                    color: #f8fafc;
                    font-size: 14px;
                    line-height: 1.6;
                    font-family: 'Inter', sans-serif;
                    background-color: #1e293b;
                    padding: 10px 16px;
                    border: 1px solid #334155;
                    border-radius: 16px;
                }
            """)
            text_col.addWidget(self.label)

        else:
            # ---- Classic Industrial Card for AI response ----
            card = QFrame()
            card.setStyleSheet("""
                QFrame {
                    background-color: #0f172a;
                    border-radius: 12px;
                    border-left: 4px solid #06b6d4;
                    border-top: 1px solid #1e293b;
                    border-right: 1px solid #1e293b;
                    border-bottom: 1px solid #1e293b;
                }
            """)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(18, 14, 18, 14)
            card_layout.setSpacing(10)

            # Header strip: system label
            hdr = QHBoxLayout()
            hdr.setSpacing(8)
            dot = QFrame()
            dot.setFixedSize(8, 8)
            dot.setStyleSheet("background-color: #10b981; border-radius: 4px;")
            hdr.addWidget(dot)
            sys_lbl = QLabel("FACTORYMIND AI  ›  Industrial Diagnostic")
            sys_lbl.setStyleSheet("""
                color: #06b6d4;
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 1px;
                font-family: 'Consolas', 'Courier New', monospace;
            """)
            hdr.addWidget(sys_lbl)
            hdr.addStretch()
            card_layout.addLayout(hdr)

            # Divider
            div = QFrame()
            div.setFixedHeight(1)
            div.setStyleSheet("background-color: #1e293b;")
            card_layout.addWidget(div)

            # Response text
            self.label.setStyleSheet("""
                QLabel {
                    color: #cbd5e1;
                    font-size: 13px;
                    line-height: 1.8;
                    font-family: 'Inter', sans-serif;
                    background: transparent;
                }
            """)
            card_layout.addWidget(self.label)

            # Feedback toolbar inside the card
            ic = "#64748b"
            fb_row = QHBoxLayout()
            fb_row.setContentsMargins(0, 4, 0, 0)
            fb_row.setSpacing(6)

            self.btn_copy = QPushButton()
            self.btn_copy.setIcon(qta.icon("fa5s.copy", color=ic))
            self.btn_copy.setFixedSize(24, 24)
            self.btn_copy.setToolTip("Copy response")
            self.btn_copy.setCursor(Qt.CursorShape.PointingHandCursor)
            self.btn_copy.setStyleSheet("""
                QPushButton { border: none; background: transparent; border-radius: 4px; }
                QPushButton:hover { background: #1e293b; }
            """)
            self.btn_copy.clicked.connect(self.copy_to_clipboard)
            fb_row.addWidget(self.btn_copy)

            self.btn_like = QPushButton()
            self.btn_like.setIcon(qta.icon("fa5s.thumbs-up", color=ic))
            self.btn_like.setFixedSize(24, 24)
            self.btn_like.setToolTip("Good response")
            self.btn_like.setCursor(Qt.CursorShape.PointingHandCursor)
            self.btn_like.setStyleSheet("""
                QPushButton { border: none; background: transparent; border-radius: 4px; }
                QPushButton:hover { background: #064e3b; }
            """)
            self.btn_like.clicked.connect(lambda: self.on_feedback(1))
            fb_row.addWidget(self.btn_like)

            self.btn_dislike = QPushButton()
            self.btn_dislike.setIcon(qta.icon("fa5s.thumbs-down", color=ic))
            self.btn_dislike.setFixedSize(24, 24)
            self.btn_dislike.setToolTip("Bad response")
            self.btn_dislike.setCursor(Qt.CursorShape.PointingHandCursor)
            self.btn_dislike.setStyleSheet("""
                QPushButton { border: none; background: transparent; border-radius: 4px; }
                QPushButton:hover { background: #7f1d1d; }
            """)
            self.btn_dislike.clicked.connect(lambda: self.on_feedback(-1))
            fb_row.addWidget(self.btn_dislike)

            fb_row.addStretch()
            card_layout.addLayout(fb_row)

            text_col.addWidget(card)

        # ---- Assemble: user = avatar right, assistant = avatar left ----
        if sender == "user":
            main_layout.addStretch()
            main_layout.addLayout(text_col)
            main_layout.addWidget(self.avatar, 0, Qt.AlignmentFlag.AlignTop)
        else:
            main_layout.addWidget(self.avatar, 0, Qt.AlignmentFlag.AlignTop)
            main_layout.addLayout(text_col)

        # Entry fade-in animation
        self.setWindowOpacity(0)
        self.fade_anim = QPropertyAnimation(self, b"windowOpacity")
        self.fade_anim.setDuration(350)
        self.fade_anim.setStartValue(0.0)
        self.fade_anim.setEndValue(1.0)
        self.fade_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.fade_anim.start()

    def copy_to_clipboard(self):
        QApplication.clipboard().setText(self.text)
        self.btn_copy.setIcon(qta.icon("fa5s.check", color="#10b981"))
        self.btn_copy.setToolTip("Copied!")

    def on_feedback(self, score):
        self.feedback_clicked.emit(score)
        self.btn_like.setEnabled(False)
        self.btn_dislike.setEnabled(False)
        if score > 0:
            self.btn_like.setIcon(qta.icon("fa5s.thumbs-up", color="#10b981"))
        else:
            self.btn_dislike.setIcon(qta.icon("fa5s.thumbs-down", color="#ef4444"))

class SidebarButton(QPushButton):
    def __init__(self, text, icon_name, parent=None):
        super().__init__(parent)
        self.text_label = f"  {text}"
        self.icon_name = icon_name
        self.setText(self.text_label)
        self.setIcon(qta.icon(icon_name, color="#94a3b8"))
        self.setCheckable(True)
        self.setFixedHeight(46)
        
        # Indicator for Active State
        self.indicator = QFrame(self)
        self.indicator.setFixedWidth(4)
        self.indicator.setFixedHeight(0) 
        self.indicator.setStyleSheet("background-color: #06b6d4; border-radius: 2px;")
        self.indicator.move(2, 10)
        
        self.setStyleSheet("""
            SidebarButton {
                text-align: left;
                padding-left: 12px;
                border: none;
                border-radius: 8px;
                color: #94a3b8;
                font-size: 14px;
                font-weight: 500;
                background: transparent;
                margin: 2px 8px;
            }
            SidebarButton:hover {
                background-color: #1e293b;
                color: #cbd5e1;
            }
            SidebarButton:checked {
                background-color: #0f172a;
                color: #06b6d4;
                font-weight: 600;
                border: 1px solid #1e293b;
            }
        """)
        self.toggled.connect(self.animate_indicator)

    def animate_indicator(self, checked):
        icon_color = "#06b6d4" if checked else "#94a3b8"
        self.setIcon(qta.icon(self.icon_name, color=icon_color))
        
        self.ind_anim = QPropertyAnimation(self.indicator, b"size")
        self.ind_anim.setDuration(300)
        if checked:
            self.ind_anim.setStartValue(QSize(4, 0))
            self.ind_anim.setEndValue(QSize(4, 26))
        else:
            self.ind_anim.setStartValue(QSize(4, 26))
            self.ind_anim.setEndValue(QSize(4, 0))
        self.ind_anim.setEasingCurve(QEasingCurve.Type.OutBack)
        self.ind_anim.start()

class GroupLabel(QLabel):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("""
            QLabel {
                color: #64748b;
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 1.2px;
                padding-left: 10px;
                margin-top: 20px;
                margin-bottom: 8px;
                font-family: 'Consolas', 'Courier New', monospace;
            }
        """)
        self.setText(text.upper())

class HistoryItem(QFrame):
    clicked = pyqtSignal(str)
    delete_requested = pyqtSignal(str)

    def __init__(self, title, date_str, session_id, parent=None):
        super().__init__(parent)
        self.session_id = session_id
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 8, 10)
        
        text_container = QWidget()
        text_layout = QVBoxLayout(text_container)
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(3)
        
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("color: #cbd5e1; font-size: 13px; font-weight: 500;")
        
        self.date_label = QLabel(date_str)
        self.date_label.setStyleSheet("color: #64748b; font-size: 11px;")
        
        text_layout.addWidget(self.title_label)
        text_layout.addWidget(self.date_label)
        layout.addWidget(text_container, 1)
        
        self.btn_delete = QPushButton()
        self.btn_delete.setFixedSize(24, 24)
        self.btn_delete.setIcon(qta.icon("fa5s.times", color="#64748b"))
        self.btn_delete.setStyleSheet("QPushButton { background: transparent; border: none; border-radius: 6px; } QPushButton:hover { background: rgba(239, 68, 68, 0.1); }")
        self.btn_delete.clicked.connect(lambda: self.delete_requested.emit(self.session_id))
        layout.addWidget(self.btn_delete)
        
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("""
            HistoryItem {
                background: transparent;
                border-radius: 10px;
            }
            HistoryItem:hover {
                background: #1e293b;
            }
        """)

    def mousePressEvent(self, event):
        if not self.btn_delete.underMouse():
            self.clicked.emit(self.session_id)
        super().mousePressEvent(event)

class WelcomeView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(24)
        
        # Header Box
        header_box = QFrame()
        header_box.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1e293b, stop:1 #0f172a);
                border: 1px solid #334155;
                border-radius: 12px;
                padding: 20px;
            }
        """)
        hb_layout = QHBoxLayout(header_box)
        hb_layout.setSpacing(18)
        
        icon_lbl = QLabel()
        icon_lbl.setPixmap(qta.icon("fa5s.microchip", color="#06b6d4").pixmap(48, 48))
        hb_layout.addWidget(icon_lbl)
        
        text_vbox = QVBoxLayout()
        text_vbox.setSpacing(4)
        
        title_lbl = QLabel("FactoryMind AI")
        title_lbl.setStyleSheet("color: #f8fafc; font-size: 24px; font-weight: 700; font-family: 'Inter', sans-serif;")
        
        desc_lbl = QLabel("Offline Industrial Manufacturing Intelligence & Diagnostic Assistant")
        desc_lbl.setStyleSheet("color: #94a3b8; font-size: 13px; font-family: 'Inter', sans-serif;")
        
        text_vbox.addWidget(title_lbl)
        text_vbox.addWidget(desc_lbl)
        hb_layout.addLayout(text_vbox)
        hb_layout.addStretch()
        
        layout.addWidget(header_box)
        
        # Telemetry Dashboard Cards
        grid = QGridLayout()
        grid.setSpacing(16)
        
        def create_telemetry_card(icon_name, title, status, description, color):
            card = QFrame()
            card.setStyleSheet(f"""
                QFrame {{
                    background-color: #0f172a;
                    border: 1px solid #1e293b;
                    border-radius: 10px;
                    padding: 12px;
                }}
                QFrame:hover {{
                    border-color: {color};
                }}
            """)
            clayout = QHBoxLayout(card)
            clayout.setSpacing(12)
            
            c_icon = QLabel()
            c_icon.setPixmap(qta.icon(icon_name, color=color).pixmap(24, 24))
            clayout.addWidget(c_icon, 0, Qt.AlignmentFlag.AlignTop)
            
            cvbox = QVBoxLayout()
            cvbox.setSpacing(2)
            
            ctitle = QLabel(title.upper())
            ctitle.setStyleSheet("color: #64748b; font-size: 9px; font-weight: bold; letter-spacing: 1px; font-family: 'Consolas', monospace;")
            
            cstatus = QLabel(status)
            cstatus.setStyleSheet(f"color: {color}; font-size: 13px; font-weight: bold; font-family: 'Inter', sans-serif;")
            
            cdesc = QLabel(description)
            cdesc.setStyleSheet("color: #94a3b8; font-size: 11px; font-family: 'Inter', sans-serif;")
            cdesc.setWordWrap(True)
            
            cvbox.addWidget(ctitle)
            cvbox.addWidget(cstatus)
            cvbox.addWidget(cdesc)
            clayout.addLayout(cvbox)
            
            return card
            
        grid.addWidget(create_telemetry_card("fa5s.server", "Neural RAG Engine", "ACTIVE", "Grounding queries in local machine manuals, SOPs, and reports.", "#10b981"), 0, 0)
        grid.addWidget(create_telemetry_card("fa5s.brain", "Llama Core", "READY", "Running 100% offline with zero external data transmission.", "#3b82f6"), 0, 1)
        grid.addWidget(create_telemetry_card("fa5s.eye", "Vision Inspector", "STANDBY", "Drag & drop screenshots or paste photos for technical visual analysis.", "#06b6d4"), 1, 0)
        grid.addWidget(create_telemetry_card("fa5s.shield-alt", "Safety Shield", "ACTIVE", "Strict adherence to LOTO, safety interlocks, and site protocols.", "#f59e0b"), 1, 1)
        
        grid_widget = QWidget()
        grid_widget.setLayout(grid)
        layout.addWidget(grid_widget)
        
        # Center align instructions
        info_lbl = QLabel("> Enter a machinery issue, fault code, or alarm above to start diagnostics.")
        info_lbl.setStyleSheet("color: #475569; font-size: 12px; font-family: 'Consolas', monospace; font-weight: bold;")
        info_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info_lbl)
        
        self.setStyleSheet("background: transparent;")

class StatusBadge(QFrame):
    def __init__(self, text, active=True, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        
        self.dot = QFrame()
        self.dot.setFixedSize(10, 10)
        self.dot.setStyleSheet(f"border-radius: 5px; background-color: {'#10b981' if active else '#ef4444'};")
        
        self.label = QLabel(text)
        self.label.setStyleSheet("color: #cbd5e1; font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 1px; font-family: 'Consolas', monospace;")
        
        layout.addWidget(self.dot)
        layout.addWidget(self.label)
        
        self.setStyleSheet(f"""
            StatusBadge {{
                background: #0f172a;
                border: 1px solid #1e293b;
                border-radius: 12px;
            }}
        """)
        
        if active:
            from PyQt6.QtCore import QTimer as QTimer3
            self.timer = QTimer3(self)
            self.timer.timeout.connect(self.do_pulse)
            self.timer.start(1000)
            self.pulse_state = True

    def do_pulse(self):
        self.pulse_state = not self.pulse_state
        self.dot.setVisible(self.pulse_state)

class DocumentItem(QFrame):
    delete_requested = pyqtSignal(str)

    def __init__(self, filename, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 8, 6)
        layout.setSpacing(8)
        
        self.filename = filename
        
        # Icon
        icon_label = QLabel()
        icon_label.setPixmap(qta.icon("fa5s.file-invoice", color="#64748b").pixmap(14, 14))
        layout.addWidget(icon_label)
        
        # Name
        name_label = QLabel(filename)
        name_label.setStyleSheet("color: #cbd5e1; font-size: 13px; font-weight: 500;")
        name_label.setToolTip(filename)
        layout.addWidget(name_label, 1)
        
        # Delete Button
        self.btn_delete = QPushButton()
        self.btn_delete.setFixedSize(24, 24)
        self.btn_delete.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_delete.setIcon(qta.icon("fa5s.trash-alt", color="#64748b"))
        self.btn_delete.setStyleSheet("""
            QPushButton { 
                background: transparent; 
                border: none; 
                border-radius: 6px;
            }
            QPushButton:hover { 
                background: rgba(239, 68, 68, 0.1); 
            }
        """)
        self.btn_delete.clicked.connect(lambda: self.delete_requested.emit(self.filename))
        layout.addWidget(self.btn_delete)
        
        self.setStyleSheet("""
            DocumentItem {
                background: transparent;
                border-radius: 10px;
            }
            DocumentItem:hover {
                background: #1e293b;
            }
        """)


class DownloadWorker(QThread):
    progress_updated = pyqtSignal(int, str)
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, manager, catalog_key):
        super().__init__()
        self.manager = manager
        self.catalog_key = catalog_key

    def run(self):
        try:
            def _hook(fraction, label):
                self.progress_updated.emit(int(fraction * 100), label)
            path = self.manager.download(self.catalog_key, progress=_hook)
            self.finished_ok.emit(path)
        except Exception as e:
            self.failed.emit(str(e))


class ModelManagerDialog(QDialog):
    """Professional model management: switch, delete and download local GGUF models."""

    def __init__(self, manager: ModelManager, current_path, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.current_path = current_path
        self.download_worker = None
        self.setWindowTitle("Model Management")
        self.resize(560, 540)
        self.setStyleSheet("""
            QDialog { background-color: #0d131f; }
            QLabel { color: #cbd5e1; font-size: 13px; }
            QListWidget {
                background-color: #0f172a; color: #cbd5e1;
                border: 1px solid #1e293b; border-radius: 8px; padding: 4px;
                font-size: 13px;
            }
            QListWidget::item { padding: 6px; border-radius: 6px; }
            QListWidget::item:selected { background-color: #1e293b; color: #06b6d4; }
            QComboBox {
                background: #0f172a; color: #cbd5e1;
                border: 1px solid #1e293b; border-radius: 8px; padding: 6px 10px;
            }
            QProgressBar {
                background-color: #0f172a; border: 1px solid #1e293b;
                border-radius: 5px; text-align: center; color: #cbd5e1; height: 14px;
            }
            QProgressBar::chunk { background-color: #06b6d4; border-radius: 4px; }
            QPushButton {
                background: #0f172a; color: #cbd5e1;
                border: 1px solid #1e293b; border-radius: 8px; padding: 7px 14px; font-weight: 600;
            }
            QPushButton:hover { background: #1e293b; border-color: #334155; }
            QPushButton:disabled { color: #475569; }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        title = QLabel("🧠 Local Model Library")
        title.setStyleSheet("font-size: 16px; font-weight: 700; color: #f8fafc;")
        layout.addWidget(title)

        layout.addWidget(QLabel("INSTALLED MODELS"))
        self.model_list = QListWidget()
        self.model_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        layout.addWidget(self.model_list, 1)

        self.btn_row = QHBoxLayout()
        self.btn_use = QPushButton("Use Selected")
        self.btn_use.clicked.connect(self._use_selected)
        self.btn_delete = QPushButton("Delete Selected")
        self.btn_delete.clicked.connect(self._delete_selected)
        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_row.addWidget(self.btn_use)
        self.btn_row.addWidget(self.btn_delete)
        self.btn_row.addWidget(self.btn_refresh)
        self.btn_row.addStretch()
        layout.addLayout(self.btn_row)

        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet("background-color: #1e293b;")
        layout.addWidget(divider)

        layout.addWidget(QLabel("DOWNLOAD FROM CATALOG"))
        cat_row = QHBoxLayout()
        self.catalog_combo = QComboBox()
        self._catalog_keys = [m["key"] for m in MODEL_CATALOG]
        self.catalog_combo.addItems(
            [f"{m['name']}  ({m['size_gb']} GB)" for m in MODEL_CATALOG])
        cat_row.addWidget(self.catalog_combo, 1)
        self.btn_download = QPushButton("⬇ Download")
        self.btn_download.clicked.connect(self._download)
        cat_row.addWidget(self.btn_download)
        layout.addLayout(cat_row)

        self.dl_progress = QProgressBar()
        self.dl_progress.setValue(0)
        self.dl_progress.setVisible(False)
        layout.addWidget(self.dl_progress)

        self.dl_status = QLabel("Catalog models are pulled from HuggingFace Hub and stored locally.")
        self.dl_status.setWordWrap(True)
        self.dl_status.setStyleSheet("color: #64748b; font-size: 11px;")
        layout.addWidget(self.dl_status)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)

        self.refresh()

    # ------------------------------------------------------------------ actions
    def refresh(self):
        self.model_list.clear()
        self._items = {}
        active = os.path.normcase(os.path.abspath(self.manager.active_model_path() or ""))
        active_hit = None
        for m in self.manager.list_main_models():
            mpath = os.path.normcase(os.path.abspath(m["path"]))
            is_active = mpath == active
            prefix = "✓ ACTIVE  " if is_active else ""
            item = QListWidgetItem(f"  {prefix}{m['name']}   ·   {m['size']}")
            item.setData(Qt.ItemDataRole.UserRole, m["path"])
            self._items[m["path"]] = m
            self.model_list.addItem(item)
            if is_active:
                self.model_list.setCurrentItem(item)
                active_hit = True
        if active_hit is None and self.model_list.count() > 0:
            self.model_list.setCurrentRow(0)

    def _selected(self):
        item = self.model_list.currentItem()
        if item is None:
            return None
        path = item.data(Qt.ItemDataRole.UserRole)
        return self._items.get(path)

    def _use_selected(self):
        m = self._selected()
        if not m:
            QMessageBox.information(self, "Model Management", "Select a model to activate.")
            return
        self.manager.set_active(m["path"])
        QMessageBox.information(
            self, "Model changed",
            f"Active model set to:\n{m['name']}\n\nRestart the application to load the new engine.")
        self.refresh()

    def _delete_selected(self):
        m = self._selected()
        if not m:
            return
        if m["path"] == self.current_path:
            QMessageBox.warning(self, "Model Management",
                                "Cannot delete the model that is currently loaded. "
                                "Switch to another model first, then retry.")
            return
        reply = QMessageBox.question(
            self, "Delete Model",
            f"Delete {m['name']} ({m['size']})?\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.manager.delete_model(m["path"])
                self.refresh()
            except Exception as e:
                QMessageBox.critical(self, "Model Management", str(e))

    def _download(self):
        if self.download_worker and self.download_worker.isRunning():
            return
        key = self._catalog_keys[self.catalog_combo.currentIndex()]
        entry = next((m for m in MODEL_CATALOG if m["key"] == key), None)
        if not entry:
            return
        self.btn_download.setEnabled(False)
        self.dl_progress.setValue(0)
        self.dl_progress.setVisible(True)
        self.dl_status.setText(f"Downloading {entry['name']} ({entry['size_gb']} GB)...")

        self.download_worker = DownloadWorker(self.manager, key)
        self.download_worker.progress_updated.connect(self._on_progress)
        self.download_worker.finished_ok.connect(self._on_done)
        self.download_worker.failed.connect(self._on_error)
        self.download_worker.start()

    def _on_progress(self, pct, label):
        self.dl_progress.setValue(pct)
        self.dl_status.setText(f"{label}: {pct}%")

    def _on_done(self, path):
        self.btn_download.setEnabled(True)
        self.dl_status.setText(f"✅ Downloaded: {os.path.basename(path)}")
        self.dl_progress.setValue(100)
        self.refresh()

    def _on_error(self, msg):
        self.btn_download.setEnabled(True)
        self.dl_status.setText(f"❌ Download failed: {msg}")
