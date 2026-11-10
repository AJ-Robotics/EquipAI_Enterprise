import sys
import os
from PyQt6.QtWidgets import QApplication, QSplashScreen
from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtCore import Qt
import qtawesome as qta
import time

# Internal Imports
from ui_main import EqpAIApp, AppSplashScreen
from ui_components import build_splash_pixmap

def launch_gui():
    """Official Production Launcher for EqpAI Desktop."""
    app = QApplication(sys.argv)
    
    # 1. Aesthetics initialization
    app.setApplicationName("FactoryMind AI Enterprise")
    app.setWindowIcon(qta.icon("fa5s.cog", color="#06b6d4"))
    
    # Global Font (Classic Modern)
    font = QFont("Inter", 10)
    app.setFont(font)

    # 2. Splash Screen (programmatic, no external asset required)
    splash = AppSplashScreen(build_splash_pixmap())
    splash.show()
    splash.set_progress(20, "Initializing Industrial Intelligence...")
    time.sleep(0.5)
    splash.set_progress(50, "Loading Neural Engine...")

    # 3. Main Application Launch
    window = EqpAIApp()

    splash.set_progress(100, "System Ready")
    splash.finish(window)

    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    # In production, we prioritize the GUI. 
    # CLI mode can be added back here if specific flags like --cli are passed.
    launch_gui()
