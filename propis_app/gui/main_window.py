"""
Main application window.

Layout:
  - Top toolbar: file loading, mode/salt/acid/face selection
  - Central area: tabbed views (signal, preprocessing, kinetics, results, comparison)
  - Bottom: status bar
"""

from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QMainWindow, QTabWidget, QStatusBar, QFileDialog,
    QVBoxLayout, QHBoxLayout, QWidget, QLabel, QComboBox,
    QPushButton, QCheckBox, QMessageBox,
)
from PyQt6.QtCore import Qt, pyqtSignal

from ..core.prn_reader import PrnData, read_prn
from ..core.pipeline import (
    CycleParams, PipelineResult, run_classic, run_modern,
)

from .signal_view import SignalView
from .preprocessing_view import PreprocessingView
from .kinetic_view import KineticView
from .results_view import ResultsView
from .comparison_view import ComparisonView
from .batch_view import BatchView


class MainWindow(QMainWindow):
    """Main application window for crystal growth data processing."""

    data_loaded = pyqtSignal(object)
    result_ready = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Propis IPFRAN — Обработка прописей KDP/DKDP")
        self.setMinimumSize(1200, 800)

        self._prn_data: Optional[PrnData] = None
        self._result_classic: Optional[PipelineResult] = None
        self._result_modern: Optional[PipelineResult] = None

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Top controls row
        top = QHBoxLayout()

        self._btn_open = QPushButton("Открыть PRN...")
        top.addWidget(self._btn_open)

        top.addWidget(QLabel("Соль:"))
        self._combo_salt = QComboBox()
        self._combo_salt.addItems(["KDP", "DKDP"])
        top.addWidget(self._combo_salt)

        top.addWidget(QLabel("Кислота:"))
        self._combo_acid = QComboBox()
        self._combo_acid.addItems(["Нейтр.", "9.83%", "12.5%", "18.5%"])
        top.addWidget(self._combo_acid)

        top.addWidget(QLabel("Грань:"))
        self._combo_face = QComboBox()
        self._combo_face.addItems(["Призма {100}", "Пирамида {101}"])
        top.addWidget(self._combo_face)

        top.addWidget(QLabel("Режим:"))
        self._combo_mode = QComboBox()
        self._combo_mode.addItems(["Классический", "Современный", "Оба"])
        top.addWidget(self._combo_mode)

        self._btn_process = QPushButton("Обработать")
        self._btn_process.setEnabled(False)
        self._btn_process.setStyleSheet("font-weight: bold;")
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
                f"T: {self._prn_data.temp_c.min():.1f}–"
                f"{self._prn_data.temp_c.max():.1f}°C"
            )
            self._signal_view.set_data(self._prn_data)
            self.data_loaded.emit(self._prn_data)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить:\n{e}")

    def _get_cycle_params(self) -> CycleParams:
        """Build CycleParams from signal view boundaries."""
        b = self._signal_view.get_boundaries()
        return CycleParams(
            n1=b["n1"], n2=b["n2"],
            im=b["im"], isat=b["isat"],
            im1=b["im1"], isat1=b["isat1"],
        )

    def _process(self):
        if self._prn_data is None:
            return

        salt = self._combo_salt.currentIndex() + 1   # 1=KDP, 2=DKDP
        acid = self._combo_acid.currentIndex()        # 0=нейтр
        face = self._combo_face.currentIndex()        # 0=призма
        mode_idx = self._combo_mode.currentIndex()    # 0=classic,1=modern,2=both
        tn_manual = self._signal_view.get_tn_manual()

        params = self._get_cycle_params()

        self._status.showMessage("Обработка...")
        self._result_classic = None
        self._result_modern = None

        try:
            if mode_idx in (0, 2):  # classic or both
                self._result_classic = run_classic(
                    self._prn_data, params,
                    salt=salt, acid=acid, face=face,
                    channel=1, tn_manual=tn_manual,
                )

            if mode_idx in (1, 2):  # modern or both
                self._result_modern = run_modern(
                    self._prn_data, params,
                    salt=salt, acid=acid, face=face,
                    channel=1,
                )

            # Show primary result
            result = self._result_classic or self._result_modern
            if result:
                self._kinetic_view.set_result(result)
                self._results_view.set_result(result)
                self._tabs.setCurrentWidget(self._kinetic_view)
                self._status.showMessage(
                    f"Готово [{result.mode}]: "
                    f"te={result.te:.2f}°C  tn={result.tn:.2f}°C  "
                    f"Td={result.Td:.2f}°C  Sigm={result.Sigm:.3f}%  "
                    f"s2={result.s2:.3f}  Sig035={result.Sig035:.2f}  "
                    f"N={result.n_extrema}"
                )
                self.result_ready.emit(result)

            # Comparison view
            if mode_idx == 2 and self._result_classic and self._result_modern:
                self._comparison_view.set_results(
                    self._result_classic, self._result_modern
                )

        except Exception as e:
            self._status.showMessage(f"Ошибка: {e}")
            QMessageBox.critical(self, "Ошибка обработки", str(e))
