"""
Main application window.

Layout:
  - Top toolbar: file loading, mode selection, preprocessing toggle
  - Left panel: parameter controls (sliders, inputs)
  - Central area: tabbed views (signal, preprocessing, kinetics, results, comparison)
  - Bottom: status bar with current parameters
"""

from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QToolBar, QStatusBar, QFileDialog,
    QVBoxLayout, QHBoxLayout, QWidget, QLabel, QComboBox,
    QPushButton, QCheckBox, QMessageBox, QSplitter,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QAction

from ..core.prn_reader import PrnData, read_prn
from ..core.batch import ProcessingParams, ProcessingResult, process_single
from ..core.solubility import get_solubility_set

from .signal_view import SignalView
from .preprocessing_view import PreprocessingView
from .kinetic_view import KineticView
from .results_view import ResultsView
from .comparison_view import ComparisonView
from .batch_view import BatchView


class MainWindow(QMainWindow):
    """Main application window for crystal growth data processing."""

    data_loaded = pyqtSignal(object)       # PrnData
    result_ready = pyqtSignal(object)      # ProcessingResult

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Propis IPFRAN — Обработка прописей KDP/DKDP")
        self.setMinimumSize(1200, 800)

        self._prn_data: Optional[PrnData] = None
        self._params = ProcessingParams()
        self._result_classic: Optional[ProcessingResult] = None
        self._result_modern: Optional[ProcessingResult] = None

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Top controls
        top = QHBoxLayout()

        self._btn_open = QPushButton("Открыть PRN...")
        top.addWidget(self._btn_open)

        top.addWidget(QLabel("Соль:"))
        self._combo_salt = QComboBox()
        self._combo_salt.addItems(["KDP", "DKDP"])
        top.addWidget(self._combo_salt)

        top.addWidget(QLabel("Кислота:"))
        self._combo_acid = QComboBox()
        self._combo_acid.addItems(["Нейтр.", "9.83%", "12.5%/12.67%", "18.5%"])
        top.addWidget(self._combo_acid)

        top.addWidget(QLabel("Грань:"))
        self._combo_face = QComboBox()
        self._combo_face.addItems(["Призма {100}", "Пирамида {101}"])
        top.addWidget(self._combo_face)

        top.addWidget(QLabel("Режим:"))
        self._combo_mode = QComboBox()
        self._combo_mode.addItems(["Классический", "Современный", "Оба (сравнение)"])
        top.addWidget(self._combo_mode)

        self._chk_preprocess = QCheckBox("Предобработка")
        top.addWidget(self._chk_preprocess)

        self._btn_process = QPushButton("Обработать")
        self._btn_process.setEnabled(False)
        top.addWidget(self._btn_process)

        top.addStretch()
        layout.addLayout(top)

        # Tab widget
        self._tabs = QTabWidget()

        self._signal_view = SignalView()
        self._tabs.addTab(self._signal_view, "Сигнал")

        self._preprocess_view = PreprocessingView()
        self._tabs.addTab(self._preprocess_view, "Предобработка")

        self._kinetic_view = KineticView()
        self._tabs.addTab(self._kinetic_view, "Кинетика")

        self._results_view = ResultsView()
        self._tabs.addTab(self._results_view, "Результаты")

        self._comparison_view = ComparisonView()
        self._tabs.addTab(self._comparison_view, "Сравнение")

        self._batch_view = BatchView()
        self._tabs.addTab(self._batch_view, "Пакетная")

        layout.addWidget(self._tabs)

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Готово. Откройте PRN файл для начала работы.")

    def _connect_signals(self):
        self._btn_open.clicked.connect(self._open_file)
        self._btn_process.clicked.connect(self._process)

    def _open_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Открыть PRN файл", "",
            "PRN files (*.prn);;All files (*)",
        )
        if not filepath:
            return

        try:
            self._prn_data = read_prn(filepath)
            self._btn_process.setEnabled(True)
            self._status.showMessage(
                f"Загружен: {Path(filepath).name} — "
                f"{self._prn_data.n_samples} отсчётов, "
                f"T: {self._prn_data.temp_c.min():.1f}–{self._prn_data.temp_c.max():.1f}°C"
            )
            self._signal_view.set_data(self._prn_data)
            self.data_loaded.emit(self._prn_data)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить файл:\n{e}")

    def _get_params(self) -> ProcessingParams:
        """Collect parameters from UI controls."""
        params = ProcessingParams()
        params.salt = self._combo_salt.currentIndex() + 1
        params.acid = self._combo_acid.currentIndex()
        params.face = self._combo_face.currentIndex()
        params.use_preprocessing = self._chk_preprocess.isChecked()

        mode_idx = self._combo_mode.currentIndex()
        params.mode = ["classic", "modern", "both"][mode_idx]

        # Get boundaries from signal view
        n1, n2, im, isat = self._signal_view.get_boundaries()
        params.n1 = n1
        params.n2 = n2
        params.im = im
        params.isat = isat

        return params

    def _process(self):
        if self._prn_data is None:
            return

        params = self._get_params()
        self._status.showMessage("Обработка...")

        try:
            # Classic mode
            if params.mode in ("classic", "both"):
                p = ProcessingParams(**{
                    k: v for k, v in params.__dict__.items()
                })
                p.mode = "classic"
                self._result_classic = process_single(self._prn_data, p)

            # Modern mode
            if params.mode in ("modern", "both"):
                p = ProcessingParams(**{
                    k: v for k, v in params.__dict__.items()
                })
                p.mode = "modern"
                p.use_preprocessing = True  # modern mode needs preprocessing
                self._result_modern = process_single(self._prn_data, p)

            # Update views
            result = self._result_classic or self._result_modern
            if result and result.success:
                self._kinetic_view.set_result(result)
                self._results_view.set_result(result)
                self._status.showMessage(
                    f"Готово: tn={result.tn:.2f}°C, "
                    f"Td={result.td:.2f}°C, "
                    f"s1={result.s1:.4f}"
                )
                self.result_ready.emit(result)
            elif result:
                self._status.showMessage(f"Ошибка: {result.error}")

            # Comparison view
            if params.mode == "both" and self._result_classic and self._result_modern:
                self._comparison_view.set_results(
                    self._result_classic, self._result_modern
                )

        except Exception as e:
            self._status.showMessage(f"Ошибка обработки: {e}")
            QMessageBox.critical(self, "Ошибка", str(e))
