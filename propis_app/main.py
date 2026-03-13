"""
Propis IPFRAN — Crystal growth data processing application.

Main entry point. Launches the PyQt6 GUI application.

Usage:
    python -m propis_app.main
    python propis_app/main.py
"""

import sys
from pathlib import Path


def main():
    # Ensure package is importable
    app_dir = Path(__file__).parent
    project_root = app_dir.parent
    for p in [str(project_root), str(app_dir)]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from PyQt6.QtWidgets import QApplication

    from propis_app.gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Propis IPFRAN")
    app.setOrganizationName("IPF RAN")
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
