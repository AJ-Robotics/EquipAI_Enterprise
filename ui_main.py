import sys
import os
import time
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLineEdit, QPushButton, QScrollArea, 
                             QFrame, QLabel, QFileDialog, QComboBox, QSplitter,
                             QInputDialog, QMessageBox, QSplashScreen, QProgressBar, QProgressDialog)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer, QTimer as QTimer2, QPropertyAnimation, QEasingCurve, QPoint
from PyQt6.QtGui import QIcon, QFont, QPixmap, QAction, QActionGroup, QKeySequence

import qtawesome as qta
from ui_components import ChatBubble, SidebarButton, StatusBadge, GroupLabel, DocumentItem, HistoryItem, WelcomeView, ModelManagerDialog, build_splash_pixmap
from rag_engine import DocumentLoader, TextSplitter, VectorStore
from llm_client import LLMClient
from session_manager import SessionManager
from feedback_manager import FeedbackManager
from model_manager import ModelManager
import config
import analytics

VECTOR_STORE_PATH = config.VECTOR_STORE_PATH

class LLMWorker(QThread):
    finished = pyqtSignal(str, list)
    chunk_received = pyqtSignal(str)

    def __init__(self, llm, store, query, role, filters, feedback_manager=None, image_path=None, history=None, mode=None):
        super().__init__()
        self.llm = llm
        self.store = store
        self.query = query
        self.role = role
        self.filters = filters
        self.feedback_manager = feedback_manager
        self.image_path = image_path
        self.history = history or []
        self.mode = mode or config.DEFAULT_RESPONSE_MODE

    def run(self):
        mode_cfg = config.response_mode(self.mode)

        # 0. Learning: search past successes (skipped in Fast mode for latency)
        past_success = []
        if self.feedback_manager and mode_cfg["use_feedback"]:
            past_success = self.feedback_manager.get_relevant_past_success(self.query)

        # 1. Retrieval (hybrid, mode-aware depth)
        results = []
        t0 = analytics.timer_start()
        if not self.image_path:
            results = self.store.search(
                self.query, k=mode_cfg["k"], filter_criteria=self.filters)

        # 2. Generation (Streaming)
        full_response = ""
        for chunk in self.llm.generate_answer_stream(
            self.query, results, role=self.role,
            past_success_context=past_success,
            image_path=self.image_path,
            history=self.history,
            mode=self.mode,
        ):
            full_response += chunk
            self.chunk_received.emit(chunk)

        latency = analytics.timer_ms(t0)
        analytics.log_query(
            self.query, role=self.role,
            latency_ms=latency,
            retrieval_score=(results[0]["score"] if results else None),
            sources=[r["metadata"].get("source") for r in results[:3]],
            code_match=any(r.get("code_match") for r in results[:3]),
            vision=bool(self.image_path),
            mode=self.mode,
        )

        self.finished.emit(full_response, results)

class VisionLoadingWorker(QThread):
    """Background thread to pre-load vision model."""
    def __init__(self, llm):
        super().__init__()
        self.llm = llm

    def run(self):
        self.llm._ensure_vision_loaded()

class IngestWorker(QThread):
    """Background thread for heavy technical data ingestion."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(int, str) # count, machine_name
    error = pyqtSignal(str)

    def __init__(self, loader, splitter, store, llm, files, machine, subsystem):
        super().__init__()
        self.loader = loader
        self.splitter = splitter
        self.store = store
        self.llm = llm
        self.files = files
        self.machine = machine
        self.subsystem = subsystem

    def run(self):
        try:
            docs = []
            total = len(self.files)
            for i, f in enumerate(self.files):
                fname = os.path.basename(f)
                self.progress.emit(int((i / total) * 80), f"Processing {fname}...")
                
                d = self.loader.load_file(f)
                if d:
                    is_img = any(f.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp'])
                    if is_img:
                        self.progress.emit(int((i / total) * 80), f"Analyzing visual: {fname}...")
                        img_desc = self.llm.get_image_description(f, "Identify equipment part and explain function.")
                        d['content'] = f"TECHNICAL IMAGE REFERENCE: {fname}\nFUNCTION: {img_desc}"
                    
                    d['machine'] = self.machine
                    d['subsystem'] = self.subsystem
                    docs.append(d)
            
            if docs:
                self.progress.emit(90, "Updating Neural Engine Index...")
                self.store.add_documents(docs, self.splitter)
                self.store.save(config.VECTOR_STORE_PATH)
                self.finished.emit(len(self.files), self.machine)
            else:
                self.finished.emit(0, self.machine)
                
        except Exception as e:
            self.error.emit(str(e))

class OptimizedInput(QLineEdit):
    """Enhanced input with Drag-and-Drop + Clipboard Paste support."""
    image_dropped = pyqtSignal(str)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                self.image_dropped.emit(file_path)
                event.acceptProposedAction()
        else:
            super().dropEvent(event)

    def keyPressEvent(self, event):
        # Handle CTRL+V for images
        if event.key() == Qt.Key.Key_V and (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            clipboard = QApplication.clipboard()
            mime_data = clipboard.mimeData()
            if mime_data.hasImage():
                image = clipboard.image()
                temp_path = os.path.join(config.TEMP_UPLOAD_DIR, "temp_paste.png")
                os.makedirs(config.TEMP_UPLOAD_DIR, exist_ok=True)
                image.save(temp_path)
                self.image_dropped.emit(temp_path)
                return
        super().keyPressEvent(event)

class EqpAIApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EquipAI Enterprise | Industrial Assistant")
        self.resize(1100, 800)
        
        # Initialize Backend
        self.store = VectorStore()
        if os.path.exists(VECTOR_STORE_PATH) or os.path.exists(VECTOR_STORE_PATH + ".faiss"):
            self.store.load(VECTOR_STORE_PATH)

        self.llm = LLMClient()
        self.session_manager = SessionManager()
        self.feedback_manager = FeedbackManager(embedding_model=self.store.model)
        self.model_manager = ModelManager()

        self.current_session_id = self.session_manager.create_new_session()
        self.messages = []
        self.attached_image_path = None
        self.vision_loader = None

        analytics.log_system(
            model=self.llm.model_info["name"],
            device=self.store.get_stats()["device"],
            threads=self.llm.model_info["threads"],
            ctx=self.llm.model_info["ctx"],
            chunks=self.store.get_stats()["chunks"],
        )

        self.setup_ui()
        self.setAcceptDrops(True)
        self._build_status_bar()
        self._build_menus()
        self.update_status()
        self._update_status_bar_info()
        self.refresh_history_list()

    def setup_ui(self):
        # Apply global application-level stylesheet on central_widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        central_widget.setStyleSheet("""
            /* Scrollbars */
            QScrollBar:vertical {
                background-color: #0d131f;
                width: 10px;
                margin: 0px 0px 0px 0px;
                border: none;
            }
            QScrollBar::handle:vertical {
                background-color: #1e293b;
                min-height: 20px;
                border-radius: 5px;
                border: none;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #334155;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
                border: none;
                background: none;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }

            QScrollBar:horizontal {
                background-color: #0d131f;
                height: 10px;
                margin: 0px 0px 0px 0px;
                border: none;
            }
            QScrollBar::handle:horizontal {
                background-color: #1e293b;
                min-width: 20px;
                border-radius: 5px;
                border: none;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #334155;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
                border: none;
                background: none;
            }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background: none;
            }

            /* Scroll Area */
            QScrollArea {
                border: none;
                background-color: transparent;
            }

            /* Splitter */
            QSplitter::handle {
                background-color: #1e293b;
            }
        """)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ---------------- SIDEBAR ----------------
        self.sidebar = QFrame()
        self.sidebar.setMinimumWidth(220)
        self.sidebar.setMaximumWidth(440)
        self.sidebar.setStyleSheet("background-color: #090d16; border-right: 1px solid #1e293b;")
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(12, 15, 12, 15)
        sidebar_layout.setSpacing(4)

        # Status Badge
        self.status_badge = StatusBadge("System Ready", active=True)
        sidebar_layout.addWidget(self.status_badge)
        sidebar_layout.addSpacing(6)

        # Compact System Info Panel
        self.sys_info = QFrame()
        self.sys_info.setStyleSheet("""
            QFrame { background-color: #0d1826; border: 1px solid #1e293b; border-radius: 8px; }
            QLabel { color: #94a3b8; font-size: 11px; background: transparent; border: none; }
        """)
        sys_layout = QVBoxLayout(self.sys_info)
        sys_layout.setContentsMargins(10, 8, 10, 8)
        sys_layout.setSpacing(2)
        self.lbl_sys_model = QLabel("Engine: —")
        self.lbl_sys_device = QLabel("Device: —")
        self.lbl_sys_vision = QLabel("Vision: —")
        self.lbl_sys_mode = QLabel(f"Mode: {self.mode_selector.currentData().upper() if hasattr(self, 'mode_selector') else 'BALANCED'}")
        for lbl in (self.lbl_sys_model, self.lbl_sys_device, self.lbl_sys_vision, self.lbl_sys_mode):
            sys_layout.addWidget(lbl)
        sidebar_layout.addWidget(self.sys_info)
        sidebar_layout.addSpacing(6)

        # New Chat Button
        btn_new_chat = SidebarButton("New chat", "fa5s.edit")
        btn_new_chat.setFixedHeight(40)
        btn_new_chat.setStyleSheet("""
            SidebarButton { 
                color: #cbd5e1; 
                border-radius: 8px; 
                padding-left: 10px; 
                font-weight: 500; 
                border: 1px solid #1e293b;
                background-color: #0f172a;
            }
            SidebarButton:hover { 
                background: #1e293b; 
                color: #f8fafc;
                border-color: #334155;
            }
        """)
        btn_new_chat.clicked.connect(self.new_session)
        sidebar_layout.addWidget(btn_new_chat)
        sidebar_layout.addSpacing(10)

        # Role Selector
        sidebar_layout.addWidget(GroupLabel("User Role"))
        self.role_selector = QComboBox()
        self.role_selector.addItems(["Operator", "Technician", "Engineer"])
        self.role_selector.setCurrentText("Technician")
        _combo_style = """
            QComboBox {
                background: #0f172a; color: #cbd5e1;
                border: 1px solid #1e293b; border-radius: 8px;
                padding: 6px 10px; font-size: 13px;
            }
            QComboBox:hover { border-color: #06b6d4; }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #0f172a;
                color: #cbd5e1;
                selection-background-color: #1e293b;
                selection-color: #06b6d4;
                border: 1px solid #1e293b;
            }
        """
        self.role_selector.setStyleSheet(_combo_style)
        sidebar_layout.addWidget(self.role_selector)
        sidebar_layout.addSpacing(8)

        # Response Mode Selector (Fast / Balanced / Research)
        sidebar_layout.addWidget(GroupLabel("Response Mode"))
        self.mode_selector = QComboBox()
        for key in ["fast", "balanced", "research"]:
            m = config.RESPONSE_MODES[key]
            self.mode_selector.addItem(f"{m['icon']} {m['label']}", userData=key)
        default_mode = config.DEFAULT_RESPONSE_MODE
        idx = self.mode_selector.findData(default_mode)
        if idx >= 0:
            self.mode_selector.setCurrentIndex(idx)
        self.mode_selector.setStyleSheet(_combo_style)
        self.mode_selector.setToolTip(config.RESPONSE_MODES[config.DEFAULT_RESPONSE_MODE]["usage"])
        self.mode_selector.currentIndexChanged.connect(self._on_mode_combo_changed)
        sidebar_layout.addWidget(self.mode_selector)
        sidebar_layout.addSpacing(8)

        # Machine / Subsystem Selectors
        sidebar_layout.addWidget(GroupLabel("Context Filter"))
        self.machine_selector = QComboBox()
        self.machine_selector.addItem("All Machines")
        self.machine_selector.setStyleSheet(_combo_style)
        sidebar_layout.addWidget(self.machine_selector)

        self.subsystem_selector = QComboBox()
        self.subsystem_selector.addItem("All Subsystems")
        self.subsystem_selector.setStyleSheet(_combo_style)
        sidebar_layout.addWidget(self.subsystem_selector)
        sidebar_layout.addSpacing(10)

        # Session History
        sidebar_layout.addWidget(GroupLabel("Recent"))
        self.history_scroll = QScrollArea()
        self.history_scroll.setWidgetResizable(True)
        self.history_scroll.setStyleSheet("background: transparent; border: none;")
        self.history_container = QWidget()
        self.history_layout = QVBoxLayout(self.history_container)
        self.history_layout.setContentsMargins(0, 0, 0, 0)
        self.history_layout.setSpacing(2)
        self.history_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.history_scroll.setWidget(self.history_container)
        sidebar_layout.addWidget(self.history_scroll, 1)
        sidebar_layout.addSpacing(8)

        # Document Management
        sidebar_layout.addWidget(GroupLabel("Knowledge Base"))
        self.doc_scroll = QScrollArea()
        self.doc_scroll.setWidgetResizable(True)
        self.doc_scroll.setStyleSheet("background: transparent; border: none;")
        self.doc_scroll.setFixedHeight(110)
        self.doc_container = QWidget()
        self.doc_layout = QVBoxLayout(self.doc_container)
        self.doc_layout.setContentsMargins(0, 0, 0, 0)
        self.doc_layout.setSpacing(2)
        self.doc_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.doc_scroll.setWidget(self.doc_container)
        sidebar_layout.addWidget(self.doc_scroll)

        btn_upload = SidebarButton("Import Data", "fa5s.database")
        btn_upload.clicked.connect(self.upload_manuals)
        sidebar_layout.addWidget(btn_upload)

        btn_models = SidebarButton("Model Management", "fa5s.microchip")
        btn_models.setCheckable(False)
        btn_models.clicked.connect(self.open_model_dialog)
        sidebar_layout.addWidget(btn_models)

        btn_analytics = SidebarButton("Export Analytics", "fa5s.chart-line")
        btn_analytics.setCheckable(False)
        btn_analytics.clicked.connect(self.action_export_analytics)
        sidebar_layout.addWidget(btn_analytics)

        self.refresh_selectors()
        self.update_document_list()

        # ---------------- CHAT AREA ----------------
        self.chat_container = QWidget()
        self.chat_container.setStyleSheet("background-color: #0d131f;")
        chat_layout = QVBoxLayout(self.chat_container)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        chat_layout.setSpacing(0)

        # TOP HEADER
        header = QFrame()
        header.setFixedHeight(50)
        header.setStyleSheet("background-color: #0d131f; border-bottom: 1px solid #1e293b;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 0, 20, 0)

        title_btn = QPushButton("FactoryMind AI  ")
        title_btn.setIcon(qta.icon("fa5s.chevron-down", color="#94a3b8"))
        title_btn.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        title_btn.setStyleSheet("QPushButton { border: none; font-size: 18px; font-weight: 600; color: #f8fafc; background: transparent; }")
        header_layout.addWidget(title_btn, 0, Qt.AlignmentFlag.AlignLeft)
        header_layout.addStretch()

        self.online_badge = QPushButton("OFFLINE RAG ACTIVE")
        self.online_badge.setFixedSize(190, 26)
        self.online_badge.setStyleSheet("""
            QPushButton {
                background-color: rgba(16, 185, 129, 0.1); 
                color: #10b981;
                border-radius: 13px; 
                font-size: 10px; 
                font-weight: bold; 
                border: 1px solid rgba(16, 185, 129, 0.3);
                font-family: 'Consolas', monospace;
            }
        """)
        header_layout.addWidget(self.online_badge)
        header_layout.addStretch()

        self.btn_share = QPushButton()
        self.btn_share.setIcon(qta.icon("fa5s.share-alt", color="#cbd5e1"))
        self.btn_share.setToolTip("Export conversation (.md)")
        self.btn_share.setFixedSize(32, 32)
        self.btn_share.setStyleSheet("QPushButton { border: none; background: transparent; }")
        self.btn_share.clicked.connect(self.export_chat)
        header_layout.addWidget(self.btn_share)
        chat_layout.addWidget(header)

        # Messages Scroll Area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("border: none; background-color: transparent;")

        self.scroll_content = QWidget()
        self.messages_layout = QVBoxLayout(self.scroll_content)
        self.messages_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.messages_layout.setSpacing(2)
        self.messages_layout.setContentsMargins(0, 10, 0, 10)

        self.welcome_widget = WelcomeView()
        self.messages_layout.addWidget(self.welcome_widget)
        self.scroll_area.setWidget(self.scroll_content)
        chat_layout.addWidget(self.scroll_area, 1)

        # ---- FLOATING BOTTOM (preview + input pill) ----
        # IMPORTANT: preview_container must be created FIRST
        self.preview_container = QWidget()
        self.preview_container.setVisible(False)
        self.preview_layout = QHBoxLayout(self.preview_container)
        self.preview_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        bottom_area = QWidget()
        bottom_vbox = QVBoxLayout(bottom_area)
        bottom_vbox.setContentsMargins(0, 0, 0, 18)
        bottom_vbox.setSpacing(8)
        bottom_vbox.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bottom_vbox.addWidget(self.preview_container)

        pill_row = QHBoxLayout()
        pill_row.addStretch()

        input_container = QFrame()
        input_container.setMinimumWidth(500)
        input_container.setMaximumWidth(860)
        input_container.setSizePolicy(
            input_container.sizePolicy().horizontalPolicy(),
            input_container.sizePolicy().verticalPolicy()
        )
        input_container.setStyleSheet("""
            QFrame {
                background-color: #0f172a;
                border-radius: 24px;
                border: 1px solid #1e293b;
            }
            QFrame:focus-within {
                border: 1px solid #06b6d4;
            }
        """)
        input_layout = QHBoxLayout(input_container)
        input_layout.setContentsMargins(15, 8, 15, 8)
        input_layout.setSpacing(10)

        self.btn_attach = QPushButton()
        self.btn_attach.setFixedSize(30, 30)
        self.btn_attach.setIcon(qta.icon("fa5s.plus", color="#cbd5e1"))
        self.btn_attach.setToolTip("Attach image")
        self.btn_attach.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 15px; } QPushButton:hover { background: #1e293b; }")
        self.btn_attach.clicked.connect(self.attach_image)
        input_layout.addWidget(self.btn_attach)

        self.chat_input = OptimizedInput()
        self.chat_input.setPlaceholderText("Ask anything")
        self.chat_input.setFixedHeight(36)
        self.chat_input.setStyleSheet("border: none; font-size: 14px; background: transparent; color: #f8fafc;")
        self.chat_input.returnPressed.connect(self.send_message)
        self.chat_input.image_dropped.connect(self.handle_image_input)
        input_layout.addWidget(self.chat_input)

        self.btn_voice = QPushButton()
        self.btn_voice.setFixedSize(30, 30)
        self.btn_voice.setIcon(qta.icon("fa5s.microphone", color="#94a3b8"))
        self.btn_voice.setToolTip("Voice input")
        self.btn_voice.setStyleSheet("QPushButton { border: none; background: transparent; border-radius: 15px; } QPushButton:hover { background: #1e293b; }")
        input_layout.addWidget(self.btn_voice)

        self.btn_send = QPushButton()
        self.btn_send.setFixedSize(28, 28)
        self.btn_send.setIcon(qta.icon("fa5s.arrow-up", color="white"))
        self.btn_send.setToolTip("Send message")
        self.btn_send.setStyleSheet("QPushButton { background-color: #06b6d4; border-radius: 14px; border: none; } QPushButton:hover { background-color: #0891b2; }")
        self.btn_send.clicked.connect(self.send_message)
        input_layout.addWidget(self.btn_send)

        pill_row.addWidget(input_container, 1)
        pill_row.addStretch()
        bottom_vbox.addLayout(pill_row)
        chat_layout.addWidget(bottom_area)

        # ---------------- SPLITTER (resizable sidebar) ----------------
        self.content_splitter = QSplitter()
        self.content_splitter.setHandleWidth(2)
        self.content_splitter.setChildrenCollapsible(False)
        self.content_splitter.addWidget(self.sidebar)
        self.content_splitter.addWidget(self.chat_container)
        self.content_splitter.setStretchFactor(0, 0)
        self.content_splitter.setStretchFactor(1, 1)
        self.content_splitter.setSizes([270, 830])
        main_layout.addWidget(self.content_splitter)

    # ---------------- MENU BAR & STATUS BAR ----------------
    def _build_menus(self):
        mbar = self.menuBar()
        mbar.setNativeMenuBar(False)

        file_menu = mbar.addMenu("&File")
        act_new = QAction("New Session", self)
        act_new.setShortcut(QKeySequence("Ctrl+N"))
        act_new.triggered.connect(self.new_session)
        file_menu.addAction(act_new)

        act_import = QAction("Import Manuals…", self)
        act_import.setShortcut(QKeySequence("Ctrl+I"))
        act_import.triggered.connect(self.upload_manuals)
        file_menu.addAction(act_import)

        file_menu.addSeparator()
        act_export = QAction("Export Session (TXT)", self)
        act_export.setShortcut(QKeySequence("Ctrl+E"))
        act_export.triggered.connect(self.export_chat)
        file_menu.addAction(act_export)

        act_export_an = QAction("Export Analytics…", self)
        act_export_an.setShortcut(QKeySequence("Ctrl+D"))
        act_export_an.triggered.connect(self.action_export_analytics)
        file_menu.addAction(act_export_an)

        file_menu.addSeparator()
        act_exit = QAction("Exit", self)
        act_exit.setShortcut(QKeySequence("Ctrl+Q"))
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        tools_menu = mbar.addMenu("&Tools")
        act_models = QAction("Model Management…", self)
        act_models.setShortcut(QKeySequence("Ctrl+M"))
        act_models.triggered.connect(self.open_model_dialog)
        tools_menu.addAction(act_models)

        mode_menu = tools_menu.addMenu("Response Mode")
        self._mode_actions = QActionGroup(self)
        self._mode_actions.setExclusive(True)
        for _key, _cfg in config.RESPONSE_MODES.items():
            _a = QAction(f"{_cfg['icon']} {_cfg['label']}", self)
            _a.setCheckable(True)
            _a.setData(_key)
            if _key == self.mode_selector.currentData():
                _a.setChecked(True)
            self._mode_actions.addAction(_a)
            mode_menu.addAction(_a)
            _a.triggered.connect(lambda checked=False, k=_key: self._set_mode(k))
        tools_menu.addSeparator()

        help_menu = mbar.addMenu("&Help")
        act_about = QAction("About EquipAI", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    def _set_mode(self, key):
        idx = self.mode_selector.findData(key)
        if idx >= 0:
            self.mode_selector.setCurrentIndex(idx)
        self._update_status_bar_info()
        self._sync_mode_action(key)

    def _on_mode_combo_changed(self, index):
        key = self.mode_selector.itemData(index)
        if key:
            self._sync_mode_action(key)
        self._update_status_bar_info()

    def _sync_mode_action(self, key):
        for _a in self._mode_actions.actions():
            _a.setChecked(_a.data() == key)

    def _show_about(self):
        m_info = self.llm.model_info
        QMessageBox.about(
            self, "EquipAI Enterprise",
            "EquipAI Enterprise — Industrial Maintenance & Troubleshooting Assistant\n\n"
            f"Engine: {m_info['name']}\n"
            f"Device: {self.store.get_stats()['device']}\n"
            f"Chunks: {self.store.get_stats()['chunks']} knowledge units\n"
            "Modes: Fast / Balanced / Research\n\n"
            "CPU-optimized on-device RAG with hybrid BM25 retrieval.")

    def _build_status_bar(self):
        bar = self.statusBar()
        bar.setStyleSheet("QStatusBar { background: #0d1826; color: #94a3b8; border-top: 1px solid #1e293b; font-size: 11px; }")
        bar.setSizeGripEnabled(False)
        self.sb_text = QLabel("Ready")
        bar.addWidget(self.sb_text)
        self.sb_chunks = QLabel("")
        self.sb_docs = QLabel("")
        self.sb_faults = QLabel("")
        self.sb_latency = QLabel("")
        for lbl in (self.sb_chunks, self.sb_docs, self.sb_faults, self.sb_latency):
            lbl.setStyleSheet("color: #64748b; padding-left: 12px;")
            bar.addPermanentWidget(lbl)

    def _update_status_bar_info(self):
        if not hasattr(self, "sb_text"):
            return
        stats = self.store.get_stats()
        self.sb_chunks.setText(f"Chunks {stats['chunks']}")
        self.sb_docs.setText(f"Docs {stats['documents']}")
        self.sb_faults.setText(f"Fault-Codes {stats.get('fault_codes_indexed', 0)}")
        try:
            _al = analytics.get_stats()["avg_latency_ms"]
            _lat = f"{_al:.1f}s" if _al is not None else "—"
        except Exception:
            _lat = "—"
        self.sb_latency.setText(f"Avg Latency {_lat}")
        if hasattr(self, "lbl_sys_model"):
            self.lbl_sys_model.setText(f"Engine: {self.llm.model_info['name'][:38]}")
            self.lbl_sys_device.setText(f"Device: {stats['device']}")
            self.lbl_sys_vision.setText(f"Vision: {'ON' if self.llm.model_info.get('vision') else 'OFF'}")
        if hasattr(self, "lbl_sys_mode"):
            self.lbl_sys_mode.setText(f"Mode: {self.mode_selector.currentData().upper()}")

    def add_message(self, text, sender="user", image_path=None):
        bubble = ChatBubble(text, sender, image_path=image_path)
        
        # Connect feedback for assistant
        if sender == "assistant":
            # For feedback logic, we need to know what query triggered this
            # We'll use the last user message as the query
            user_query = ""
            for m in reversed(self.messages):
                if m['role'] == 'user':
                    user_query = m['content']
                    break
            
            bubble.feedback_clicked.connect(
                lambda s, q=user_query, r=text: self.on_feedback_submitted(s, q, r, bubble)
            )

        self.messages_layout.addWidget(bubble)
        # Scroll to bottom
        QTimer.singleShot(50, lambda: self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        ))
        return bubble

    def on_feedback_submitted(self, score, query, response, bubble):
        """User clicked Like/Dislike"""
        self.feedback_manager.add_feedback(query, response, score, context=[])
        if score > 0:
            QMessageBox.information(self, "Success", "Knowledge updated.")
        else:
            QMessageBox.warning(self, "Noted", f"Feedback received.")

    def send_message(self):
        query = self.chat_input.text().strip()
        if not query: return
        
        # Hide welcome on first message
        if self.welcome_widget:
            self.welcome_widget.setParent(None)
            self.welcome_widget = None

        # Pass attached image
        img_to_send = self.attached_image_path
        self.remove_attached_image()

        self.add_message(query, "user", image_path=img_to_send)
        self.messages.append({"role": "user", "content": query, "image": img_to_send})

        # Create Assistant Bubble
        self.current_assistant_bubble = self.add_message("...", "assistant")
        self.current_response_text = ""

        # Run Backend Logic
        role = self.role_selector.currentText()
        filters = {}
        m_filter = self.machine_selector.currentText()
        s_filter = self.subsystem_selector.currentText()
        if m_filter != "All Machines": filters['machine'] = m_filter
        if s_filter != "All Subsystems": filters['subsystem'] = s_filter
        
        self.worker = LLMWorker(self.llm, self.store, query, role, filters, 
                               self.feedback_manager, image_path=img_to_send,
                               history=self.messages, mode=self.mode_selector.currentData())
        self.worker.chunk_received.connect(self.update_response_chunk)
        self.worker.finished.connect(self.finalize_response)
        self.worker.start()

        # Update header mode badge for a professional system overview
        mode_label = self.mode_selector.currentText()
        self.online_badge.setText(f"OFFLINE RAG · {mode_label}")
        mode_cfg = config.response_mode(self.mode_selector.currentData())
        self.online_badge.setToolTip(mode_cfg["usage"])

        # Clear input after sending
        self.chat_input.clear()

    def update_response_chunk(self, chunk):
        if self.current_response_text == "":
            self.current_assistant_bubble.label.clear()
        self.current_response_text += chunk
        self.current_assistant_bubble.label.setText(self.current_response_text)

    def finalize_response(self, full_text, results):
        self.messages.append({"role": "assistant", "content": full_text})
        
        # Check for images in results and show if relevant
        all_images = []
        for r in results:
            if 'metadata' in r and 'images' in r['metadata']:
                all_images.extend(r['metadata']['images'])
                
        if all_images:
            # Show the first relevant image
            img = all_images[0]
            if os.path.exists(img['path']):
                img_label = QLabel()
                pixmap = QPixmap(img['path']).scaled(500, 400, Qt.AspectRatioMode.KeepAspectRatio)
                img_label.setPixmap(pixmap)
                img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                img_label.setStyleSheet("border: 1px solid #e2e8f0; border-radius: 12px; margin: 10px 0;")
                self.messages_layout.addWidget(img_label)
                
                caption = QLabel(f"📄 Schematic Reference: {img['name']} (Page {img['page']+1})")
                caption.setStyleSheet("color: #64748b; font-size: 11px; margin-bottom: 20px;")
                caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.messages_layout.addWidget(caption)

        # Save session
        self.session_manager.save_session(self.current_session_id, self.messages, self.messages[0]["content"][:20])
        self._update_status_bar_info()
        
        # Auto-scroll to bottom
        self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        )

    def new_session(self):
        self.current_session_id = self.session_manager.create_new_session()
        self.messages = []
        # Clear UI
        for i in reversed(range(self.messages_layout.count())): 
            self.messages_layout.itemAt(i).widget().setParent(None)
            
        # Re-add welcome view
        self.welcome_widget = WelcomeView()
        self.messages_layout.addWidget(self.welcome_widget)

    def open_model_dialog(self):
        """Opens the Model Management dialog (switch, delete, download models)."""
        dlg = ModelManagerDialog(self.model_manager, current_path=self.llm.model_path, parent=self)
        dlg.exec()

    def export_chat(self):
        """Exports the current conversation as a Markdown troubleshooting report."""
        if not self.messages:
            QMessageBox.information(self, "Export", "Nothing to export yet.")
            return

        lines = ["# EquipAI Troubleshooting Report", ""]
        lines.append(f"- **Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"- **Role:** {self.role_selector.currentText()}")
        lines.append(f"- **Machine:** {self.machine_selector.currentText()}")
        lines.append(f"- **Subsystem:** {self.subsystem_selector.currentText()}")
        lines.append("")
        for msg in self.messages:
            name = "**Operator**" if msg["role"] == "user" else "**EquipAI**"
            lines.append(f"{name}: {msg['content']}")
            lines.append("")

        md = "\n".join(lines)
        default_name = f"equipai_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
        path, _ = QFileDialog.getSaveFileName(self, "Export Report", default_name, "Markdown (*.md);;Text (*.txt)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(md)
            self.add_message(f"📤 Report exported to {os.path.basename(path)}.", "assistant")
            QMessageBox.information(self, "Export", f"Report saved to:\n{path}")

    def action_export_analytics(self):
        """Exports usage analytics to CSV for management reporting."""
        default_name = f"equipai_analytics_{datetime.now().strftime('%Y%m%d')}.csv"
        path, _ = QFileDialog.getSaveFileName(self, "Export Analytics", default_name, "CSV (*.csv)")
        if path:
            try:
                analytics.export_csv(path)
                QMessageBox.information(self, "Analytics", f"Analytics saved to:\n{path}")
            except Exception as e:
                QMessageBox.warning(self, "Analytics", f"Export failed:\n{e}")

    def upload_manuals(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Technical Data", "", "Data (*.pdf *.docx *.txt *.csv *.log *.png *.jpg *.jpeg *.bmp)")
        if files:
            # Ask for Machine Name
            machine, ok = QInputDialog.getText(self, "Machine Information", "Enter Machine Name (e.g. CNC-01):")
            if not ok or not machine: machine = "Unknown"
            
            subsystem, ok = QInputDialog.getText(self, "Machine Information", "Enter Subsystem (e.g. Electrical):")
            if not ok or not subsystem: subsystem = "General"

            # Initialize Background Worker
            loader = DocumentLoader()
            splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
            
            # Setup Progress Dialog
            self.pd = QProgressDialog("Initializing Data Ingestion...", "Cancel", 0, 100, self)
            self.pd.setWindowModality(Qt.WindowModality.WindowModal)
            self.pd.setMinimumDuration(0)
            self.pd.setWindowTitle("Neural Data Hub")
            self.pd.show()

            self.ingest_worker = IngestWorker(loader, splitter, self.store, self.llm, files, machine, subsystem)
            self.ingest_worker.progress.connect(lambda val, txt: self.update_ingest_progress(val, txt))
            self.ingest_worker.finished.connect(self.on_ingest_finished)
            self.ingest_worker.error.connect(self.on_ingest_error)
            self.ingest_worker.start()

    def update_ingest_progress(self, val, txt):
        if hasattr(self, 'pd'):
            self.pd.setValue(val)
            self.pd.setLabelText(txt)

    def on_ingest_finished(self, count, machine):
        if hasattr(self, 'pd'):
            self.pd.close()
        self.refresh_selectors()
        self.update_document_list()
        if count > 0:
            self.add_message(f"✅ Data Hub Updated: Successfully ingested {count} items for {machine}.", "assistant")
        else:
            self.add_message(f"⚠️ Ingestion complete, but no new data chunks were added (likely duplicates).", "assistant")

    def on_ingest_error(self, err_msg):
        if hasattr(self, 'pd'):
            self.pd.close()
        QMessageBox.critical(self, "Ingestion Error", f"An error occurred during data processing:\n{err_msg}")

    def refresh_selectors(self):
        """Updates the dropdowns with unique values from the vector store."""
        machines, subsystems = self.store.get_unique_metadata_values()
        
        # Sync Machine Selector
        current_m = self.machine_selector.currentText()
        self.machine_selector.clear()
        self.machine_selector.addItem("All Machines")
        self.machine_selector.addItems(machines)
        idx = self.machine_selector.findText(current_m)
        if idx >= 0: self.machine_selector.setCurrentIndex(idx)
        
        # Sync Subsystem Selector
        current_s = self.subsystem_selector.currentText()
        self.subsystem_selector.clear()
        self.subsystem_selector.addItem("All Subsystems")
        self.subsystem_selector.addItems(subsystems)
        idx = self.subsystem_selector.findText(current_s)
        if idx >= 0: self.subsystem_selector.setCurrentIndex(idx)

    def refresh_history_list(self):
        """Updates the sidebar history list."""
        for i in reversed(range(self.history_layout.count())):
            self.history_layout.itemAt(i).widget().setParent(None)
            
        sessions = self.session_manager.list_sessions()
        for sess in sessions:
            date_obj = datetime.fromisoformat(sess['last_updated'])
            date_str = date_obj.strftime("%b %d, %H:%M")
            item = HistoryItem(sess['title'], date_str, sess['session_id'])
            item.clicked.connect(self.load_session)
            item.delete_requested.connect(self.request_delete_session)
            self.history_layout.addWidget(item)

    def request_delete_session(self, session_id):
        """Asks user for confirmation before deleting a session."""
        reply = QMessageBox.question(self, 'Delete History', 
                                    "Are you sure you want to delete this session history?\nThis action cannot be undone.",
                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            self.session_manager.delete_session(session_id)
            if self.current_session_id == session_id:
                self.new_session()
            self.refresh_history_list()
            self.add_message("🗑️ Session history deleted.", "assistant")

    def load_session(self, session_id):
        """Loads a past session into the UI."""
        data = self.session_manager.load_session(session_id)
        if not data: return
        
        self.current_session_id = session_id
        self.messages = data.get('messages', [])
        
        # Clear UI
        for i in reversed(range(self.messages_layout.count())):
            self.messages_layout.itemAt(i).widget().setParent(None)
        
        self.welcome_widget = None
        
        for msg in self.messages:
            self.add_message(msg['content'], msg['role'], image_path=msg.get('image'))

    def handle_image_input(self, file_path):
        """Unified entry point for D&D, Clipboard, and File Dialog."""
        self.attach_image(file_path)

    def attach_image(self, file_path=None):
        """Open file dialog (if no path) and show preview + trigger pre-load."""
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        
        if file_path:
            self.attached_image_path = file_path
            
            # PERFORMANCE: Pre-load vision model in background
            if not self.llm.vision_llm:
                self.vision_loader = VisionLoadingWorker(self.llm)
                self.vision_loader.start()

            # Update UI
            from ui_components import ImagePreviewItem
            for i in reversed(range(self.preview_layout.count())):
                item = self.preview_layout.itemAt(i)
                if item and item.widget():
                    item.widget().setParent(None)
                else:
                    self.preview_layout.removeItem(item)
                
            preview = ImagePreviewItem(file_path)
            preview.remove_requested.connect(self.remove_attached_image)
            self.preview_layout.addWidget(preview)
            self.preview_layout.addStretch()
            self.preview_container.setVisible(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                self.handle_image_input(file_path)
        else:
            super().dropEvent(event)

    def remove_attached_image(self):
        """Clears the attachment."""
        self.attached_image_path = None
        for i in reversed(range(self.preview_layout.count())):
            item = self.preview_layout.itemAt(i)
            if item and item.widget():
                item.widget().setParent(None)
            else:
                self.preview_layout.removeItem(item)
        self.preview_container.setVisible(False)

    def update_document_list(self):
        """Refreshes the sidebar list of ingested documents."""
        # Clear existing
        for i in reversed(range(self.doc_layout.count())): 
            self.doc_layout.itemAt(i).widget().setParent(None)
            
        docs = self.store.get_ingested_documents()
        for doc_name in docs:
            item = DocumentItem(doc_name)
            item.delete_requested.connect(self.request_delete_document)
            self.doc_layout.addWidget(item)

    def request_delete_document(self, filename):
        reply = QMessageBox.question(self, 'Delete Document', 
                                    f"Are you sure you want to delete '{filename}'?\nThis will remove it from the knowledge base.",
                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            self.store.delete_document(filename)
            self.store.save(config.VECTOR_STORE_PATH)
            self.update_document_list()
            self.refresh_selectors()
            self.add_message(f"🗑️ Removed '{filename}' from the system.", "assistant")
            self._update_status_bar_info()

    def update_status(self):
        is_ready = self.llm.check_health()
        self.status_badge.label.setText("System Active" if is_ready else "Model Missing")
        if is_ready:
            info = self.llm.model_info
            self.status_badge.setToolTip(
                f"Model: {info['name']}\nThreads: {info['threads']} (CPU)\nContext: {info['ctx']}\n"
                f"Vision: {'Loaded' if info['vision_loaded'] else ('Unavailable' if not info['vision_available'] else 'Standby')}"
            )
        self._update_status_bar_info()

class AppSplashScreen(QSplashScreen):
    def __init__(self, pixmap):
        super().__init__(pixmap)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)
        self.setWindowOpacity(0.0) # Start transparent for Fade-In
        
        # 1. Fade-In Animation
        self.fade_anim = QPropertyAnimation(self, b"windowOpacity")
        self.fade_anim.setDuration(800)
        self.fade_anim.setStartValue(0.0)
        self.fade_anim.setEndValue(1.0)
        self.fade_anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self.fade_anim.start()

        # 2. Classic Style Components
        self.progress = QProgressBar(self)
        self.progress.setGeometry(50, pixmap.height() - 50, pixmap.width() - 100, 10)
        self.progress.setStyleSheet("""
            QProgressBar {
                background-color: rgba(15, 23, 42, 0.6);
                border: 1px solid #334155;
                border-radius: 5px;
                text-align: center;
                color: transparent;
            }
            QProgressBar::chunk {
                background-color: #06b6d4;
                border-radius: 4px;
            }
        """)
        self.progress.setValue(0)
        
        # retro terminal-style font
        self.label = QLabel("> INITIALIZING INDUSTRIAL SYSTEM...", self)
        self.label.move(50, pixmap.height() - 85)
        self.label.setFixedWidth(pixmap.width() - 100)
        self.label.setStyleSheet("""
            color: #06b6d4; 
            font-size: 13px; 
            font-family: 'Consolas', 'Courier New', monospace; 
            font-weight: 600;
        """)

        # 3. Dynamic Scanline Animation
        self.scanline = QFrame(self)
        self.scanline.setGeometry(0, 0, pixmap.width(), 2)
        self.scanline.setStyleSheet("background-color: rgba(6, 182, 212, 0.4); border: none;")
        
        self.scan_anim = QPropertyAnimation(self.scanline, b"pos")
        self.scan_anim.setDuration(2500)
        self.scan_anim.setStartValue(QPoint(0, 50))
        self.scan_anim.setEndValue(QPoint(0, pixmap.height() - 100))
        self.scan_anim.setLoopCount(-1) # Infinite loop
        self.scan_anim.setEasingCurve(QEasingCurve.Type.Linear)
        self.scan_anim.start()

    def set_progress(self, val, text=None):
        self.progress.setValue(val)
        if text: 
            # Add classic prefix
            formatted_text = f"> {text.upper()}"
            self.label.setText(formatted_text)
        QApplication.processEvents()

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)

        # 1. Splash Screen (programmatic, portable)
        splash = AppSplashScreen(build_splash_pixmap(800, 500))
        splash.show()
        splash.set_progress(20, "Loading Neural Engine...")

        app.setWindowIcon(qta.icon("fa5s.cog", color="#06b6d4"))

        # Apply global font (strictly positive to fix the QFont::setPointSize: Point size <= 0 warning)
        font = QFont("Inter", 10)
        app.setFont(font)

        splash.set_progress(50, "Indexing Technical Manuals...")
        window = EqpAIApp()

        splash.set_progress(100, "System Ready")
        time.sleep(0.5)
        splash.finish(window)

        window.show()
        sys.exit(app.exec())
    except Exception:
        # Write startup errors to a log file next to the executable so a frozen
        # (windowed) build can be diagnosed - no console is available.
        import traceback
        try:
            log_path = os.path.join(config.BASE_DIR, "startup_error.log")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())
        except Exception:
            pass
        raise
