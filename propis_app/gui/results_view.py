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

from ..core.batch import ProcessingResult


# Mat table column definitions
MAT_COLUMNS = [
    ("Параметр", "param"),
    ("Значение", "value"),
    ("Описание", "description"),
]

MAT_ROWS = [
    ("Salt", "salt", "Тип соли (1=KDP, 2=DKDP)"),
    ("Acid", "acid", "Кислотность (0=нейтр.)"),
    ("te", "te", "Равновесная температура (°C)"),
    ("tn", "tn", "Температура насыщения (°C)"),
    ("Td", "td", "Мёртвая зона (°C)"),
    ("s1", "s1", "Мёртвая зона (% перенасыщения)"),
    ("s0", "s0", "Кинетический коэффициент"),
    ("s2", "s2", "Параметр формы"),
    ("Sig035", "sig035", "σ при F>0.35 (%)"),
    ("w", "w", "Показатель степени"),
]


class ResultsView(QWidget):
    """Results table and export."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._result: Optional[ProcessingResult] = None
        self._results_list: list[ProcessingResult] = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # File info
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

        self._btn_export_mat = QPushButton("Экспорт Mat (таблица)")
        self._btn_export_mat.clicked.connect(self._export_mat)
        btn_layout.addWidget(self._btn_export_mat)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def set_result(self, result: ProcessingResult):
        """Display single result."""
        self._result = result
        self._lbl_file.setText(f"Файл: {result.filepath.name}")

        values = {
            "salt": str(result.salt),
            "acid": str(result.acid),
            "te": f"{result.te:.2f}",
            "tn": f"{result.tn:.2f}",
            "td": f"{result.td:.2f}",
            "s1": f"{result.s1:.4f}",
            "s0": f"{result.s0:.4f}",
            "s2": f"{result.s2:.4f}",
            "sig035": f"{result.sig035:.2f}",
            "w": f"{result.w:.2f}",
        }

        for i, (_, key, _) in enumerate(MAT_ROWS):
            item = QTableWidgetItem(values.get(key, "—"))
            self._table.setItem(i, 1, item)

    def set_results_list(self, results: list[ProcessingResult]):
        """Store multiple results for batch export."""
        self._results_list = results

    def _export_csv(self):
        if self._result is None:
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Сохранить CSV", "", "CSV files (*.csv)"
        )
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
        """Export in Mat format (like Mathcad output vector)."""
        if self._result is None:
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Сохранить Mat", "", "Text files (*.txt);;CSV files (*.csv)"
        )
        if not filepath:
            return

        r = self._result
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# Propis IPFRAN — результаты обработки\n")
            f.write(f"# Файл: {r.filepath.name}\n")
            f.write(f"Salt={r.salt}\n")
            f.write(f"Acid={r.acid}\n")
            f.write(f"te={r.te:.2f}\n")
            f.write(f"tn={r.tn:.2f}\n")
            f.write(f"Td={r.td:.2f}\n")
            f.write(f"s1={r.s1:.4f}\n")
            f.write(f"s0={r.s0:.4f}\n")
            f.write(f"s2={r.s2:.4f}\n")
            f.write(f"Sig035={r.sig035:.2f}\n")
            f.write(f"w={r.w:.2f}\n")
