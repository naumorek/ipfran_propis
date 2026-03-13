"""
Results view: parameter table and export.

Displays the Mat table (like Mathcad output) and allows CSV export.
"""

from typing import Optional
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QFileDialog, QLabel, QHeaderView,
)
from PyQt6.QtCore import Qt

from ..core.pipeline import PipelineResult


MAT_ROWS = [
    ("Salt", "Salt", "Тип соли (1=KDP, 2=DKDP)"),
    ("Acid", "Acid", "Кислотность (0=нейтр.)"),
    ("Face", "Face", "Грань (0=призма, 1=пирамида)"),
    ("te", "te", "Температура при isat (°C)"),
    ("tn", "tn", "Температура насыщения (°C)"),
    ("Td", "Td", "Мёртвая зона (°C)"),
    ("Sigm", "Sigm", "Перенасыщение при им. зоны (%)"),
    ("s0", "s0", "Кинетический коэффициент"),
    ("s1", "s1", "Мёртвая зона (% перенасыщения)"),
    ("s2", "s2", "Параметр формы (-Q1/Q2)"),
    ("Sig035", "Sig035", "σ при LOESS(R) > 0.35 (%)"),
    ("N экстр.", "n_extrema", "Число экстремумов"),
    ("Режим", "mode", "classic / modern"),
]


class ResultsView(QWidget):
    """Results table and export."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._result: Optional[PipelineResult] = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        self._lbl_file = QLabel("Файл: —")
        layout.addWidget(self._lbl_file)

        # Parameters table
        self._table = QTableWidget()
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(["Параметр", "Значение", "Описание"])
        self._table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        self._table.setRowCount(len(MAT_ROWS))

        for i, (name, _, desc) in enumerate(MAT_ROWS):
            self._table.setItem(i, 0, QTableWidgetItem(name))
            self._table.setItem(i, 1, QTableWidgetItem("—"))
            self._table.setItem(i, 2, QTableWidgetItem(desc))

        layout.addWidget(self._table)

        # Export buttons
        btn_layout = QHBoxLayout()
        self._btn_export_csv = QPushButton("Экспорт CSV")
        self._btn_export_csv.clicked.connect(self._export_csv)
        btn_layout.addWidget(self._btn_export_csv)

        self._btn_export_mat = QPushButton("Экспорт Mat (вектор)")
        self._btn_export_mat.clicked.connect(self._export_mat)
        btn_layout.addWidget(self._btn_export_mat)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def set_result(self, result: PipelineResult):
        """Display single result."""
        self._result = result
        self._lbl_file.setText(f"Файл: {result.filename}")

        def fmt(val, f=".4f"):
            if val is None or val == 0.0:
                return "—"
            return f"{val:{f}}"

        values = {
            "Salt": str(result.Salt),
            "Acid": str(result.Acid),
            "Face": str(result.Face),
            "te": f"{result.te:.2f}",
            "tn": f"{result.tn:.2f}",
            "Td": f"{result.Td:.3f}",
            "Sigm": f"{result.Sigm:.4f}",
            "s0": fmt(result.s0),
            "s1": fmt(result.s1),
            "s2": fmt(result.s2),
            "Sig035": f"{result.Sig035:.2f}",
            "n_extrema": str(result.n_extrema),
            "mode": result.mode,
        }

        for i, (_, key, _) in enumerate(MAT_ROWS):
            item = QTableWidgetItem(values.get(key, "—"))
            self._table.setItem(i, 1, item)

    def _export_csv(self):
        if self._result is None:
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Сохранить CSV", "", "CSV files (*.csv)")
        if not filepath:
            return
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("Параметр,Значение,Описание\n")
            for i in range(self._table.rowCount()):
                param = self._table.item(i, 0).text()
                value = self._table.item(i, 1).text()
                desc = self._table.item(i, 2).text()
                f.write(f"{param},{value},{desc}\n")

    def _export_mat(self):
        """Export in Mat format (Mathcad-compatible vector)."""
        if self._result is None:
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Сохранить Mat", "", "Text files (*.txt)")
        if not filepath:
            return

        r = self._result
        mat = r.mat_vector()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# Propis IPFRAN — Mat vector\n")
            f.write(f"# File: {r.filename}\n")
            f.write(f"# Mode: {r.mode}\n")
            labels = ["Salt", "Acid", "te", "te1", "tn",
                      "s1", "Sigm", "s2", "Sig035", "Td",
                      "n1", "n2", "ww", "d", "dtau", "l",
                      "im1", "isat1", "im", "isat"]
            for label, val in zip(labels, mat):
                f.write(f"{label}={val}\n")
