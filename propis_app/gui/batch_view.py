"""
Batch processing view.

Allows selecting a directory, processing all PRN files,
and displaying a summary table.
"""

from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QTableWidget, QTableWidgetItem, QHeaderView,
    QProgressBar, QMessageBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from ..core.batch import (
    ProcessingParams, ProcessingResult,
    process_batch, results_to_mat_table,
)
from ..core.prn_reader import read_prn


class BatchWorker(QThread):
    """Worker thread for batch processing."""
    progress = pyqtSignal(int, int)  # current, total
    finished = pyqtSignal(list)      # results
    error = pyqtSignal(str)

    def __init__(self, directory: Path, params: ProcessingParams):
        super().__init__()
        self._directory = directory
        self._params = params

    def run(self):
        try:
            prn_files = sorted(self._directory.glob("*.prn"))
            results = []
            for i, prn_file in enumerate(prn_files):
                self.progress.emit(i + 1, len(prn_files))
                try:
                    data = read_prn(prn_file)
                    from ..core.batch import process_single
                    result = process_single(data, self._params)
                except Exception as e:
                    result = ProcessingResult(
                        filepath=prn_file, params=self._params,
                        success=False, error=str(e),
                    )
                results.append(result)
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class BatchView(QWidget):
    """Batch processing interface."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._results: list[ProcessingResult] = []
        self._worker: Optional[BatchWorker] = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Controls
        ctrl = QHBoxLayout()

        self._btn_select = QPushButton("Выбрать папку...")
        self._btn_select.clicked.connect(self._select_directory)
        ctrl.addWidget(self._btn_select)

        self._lbl_dir = QLabel("Папка: не выбрана")
        ctrl.addWidget(self._lbl_dir)

        self._btn_process = QPushButton("Обработать все")
        self._btn_process.setEnabled(False)
        self._btn_process.clicked.connect(self._start_processing)
        ctrl.addWidget(self._btn_process)

        self._btn_export = QPushButton("Экспорт CSV")
        self._btn_export.setEnabled(False)
        self._btn_export.clicked.connect(self._export_csv)
        ctrl.addWidget(self._btn_export)

        ctrl.addStretch()
        layout.addLayout(ctrl)

        # Progress
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        # Results table
        self._table = QTableWidget()
        self._table.setColumnCount(12)
        self._table.setHorizontalHeaderLabels([
            "Файл", "Salt", "Acid", "te", "tn", "Td",
            "s1", "s0", "s2", "Sig035", "Статус", "Ошибка"
        ])
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        layout.addWidget(self._table)

        self._directory: Optional[Path] = None

    def _select_directory(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, "Выбрать папку с PRN файлами"
        )
        if dir_path:
            self._directory = Path(dir_path)
            prn_count = len(list(self._directory.glob("*.prn")))
            self._lbl_dir.setText(f"Папка: {self._directory.name} ({prn_count} PRN)")
            self._btn_process.setEnabled(prn_count > 0)

    def _start_processing(self):
        if self._directory is None:
            return

        params = ProcessingParams()  # default parameters
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._btn_process.setEnabled(False)

        self._worker = BatchWorker(self._directory, params)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, current: int, total: int):
        self._progress.setMaximum(total)
        self._progress.setValue(current)

    def _on_finished(self, results: list):
        self._results = results
        self._progress.setVisible(False)
        self._btn_process.setEnabled(True)
        self._btn_export.setEnabled(True)
        self._populate_table()

    def _on_error(self, error: str):
        self._progress.setVisible(False)
        self._btn_process.setEnabled(True)
        QMessageBox.critical(self, "Ошибка", error)

    def _populate_table(self):
        rows = results_to_mat_table(self._results)
        self._table.setRowCount(len(rows))

        for i, row in enumerate(rows):
            cols = [
                row["file"], str(row["Salt"]), str(row["Acid"]),
                str(row["te"]), str(row["tn"]), str(row["Td"]),
                str(row["s1"]), str(row["s0"]), str(row["s2"]),
                str(row["Sig035"]),
                "OK" if row["success"] else "Ошибка",
                row["error"],
            ]
            for j, val in enumerate(cols):
                item = QTableWidgetItem(val)
                if not row["success"]:
                    item.setBackground(Qt.GlobalColor.red)
                self._table.setItem(i, j, item)

    def _export_csv(self):
        if not self._results:
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Сохранить CSV", "batch_results.csv",
            "CSV files (*.csv)"
        )
        if not filepath:
            return

        rows = results_to_mat_table(self._results)
        with open(filepath, "w", encoding="utf-8") as f:
            headers = list(rows[0].keys())
            f.write(",".join(headers) + "\n")
            for row in rows:
                f.write(",".join(str(row[h]) for h in headers) + "\n")
