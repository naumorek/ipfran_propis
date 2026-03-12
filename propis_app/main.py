"""
Propis IPFRAN — Crystal growth data processing application.

Main entry point. Launches the PyQt6 GUI application.

Usage:
    python -m propis_app.main
    python propis_app/main.py
"""

import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Propis IPFRAN")
    app.setOrganizationName("IPF RAN")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    # Ensure the package is importable
    app_dir = Path(__file__).parent
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))
    if str(app_dir.parent) not in sys.path:
        sys.path.insert(0, str(app_dir.parent))

    main()
