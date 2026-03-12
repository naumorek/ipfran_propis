"""
Preprocessing visualization view.

Shows:
  - Raw signal vs preprocessed signal (overlay)
  - Envelope overlay
  - Filter frequency response
  - Parameter controls for preprocessing pipeline
"""

from typing import Optional

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QCheckBox, QDoubleSpinBox, QSpinBox, QLabel, QPushButton,
    QComboBox,
)

import pyqtgraph as pg

from ..core.preprocessing import (
    PreprocessingParams, PreprocessingResult, preprocess,
)


class PreprocessingView(QWidget):
    """Preprocessing visualization and parameter adjustment."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._signal: Optional[np.ndarray] = None
        self._fs: float = 1.0
        self._result: Optional[PreprocessingResult] = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Controls
        ctrl_group = QGroupBox("Параметры предобработки")
        ctrl_layout = QHBoxLayout(ctrl_group)

        # Detrend
        self._chk_detrend = QCheckBox("Удаление тренда")
        self._chk_detrend.setChecked(True)
        ctrl_layout.addWidget(self._chk_detrend)

        # Bandpass
        self._chk_bandpass = QCheckBox("Полосовой фильтр")
        self._chk_bandpass.setChecked(True)
        ctrl_layout.addWidget(self._chk_bandpass)

        ctrl_layout.addWidget(QLabel("f_low:"))
        self._spin_flow = QDoubleSpinBox()
        self._spin_flow.setRange(0.0001, 1.0)
        self._spin_flow.setValue(0.005)
        self._spin_flow.setDecimals(4)
        self._spin_flow.setSingleStep(0.001)
        ctrl_layout.addWidget(self._spin_flow)

        ctrl_layout.addWidget(QLabel("f_high:"))
        self._spin_fhigh = QDoubleSpinBox()
        self._spin_fhigh.setRange(0.001, 10.0)
        self._spin_fhigh.setValue(0.1)
        self._spin_fhigh.setDecimals(3)
        self._spin_fhigh.setSingleStep(0.01)
        ctrl_layout.addWidget(self._spin_fhigh)

        # Normalize
        self._chk_normalize = QCheckBox("Нормализация")
        self._chk_normalize.setChecked(True)
        ctrl_layout.addWidget(self._chk_normalize)

        # Wavelet
        self._chk_wavelet = QCheckBox("Вейвлет")
        self._chk_wavelet.setChecked(False)
        ctrl_layout.addWidget(self._chk_wavelet)

        # Apply button
        self._btn_apply = QPushButton("Применить")
        self._btn_apply.clicked.connect(self._apply_preprocessing)
        ctrl_layout.addWidget(self._btn_apply)

        layout.addWidget(ctrl_group)

        # Plots
        # Signal before/after
        self._plot_signal = pg.PlotWidget(title="Сигнал: до и после предобработки")
        self._plot_signal.setLabel("bottom", "Отсчёт")
        self._plot_signal.setLabel("left", "Амплитуда")
        self._plot_signal.addLegend()
        self._plot_signal.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self._plot_signal, stretch=2)

        # Envelope
        self._plot_envelope = pg.PlotWidget(title="Огибающая")
        self._plot_envelope.setLabel("bottom", "Отсчёт")
        self._plot_envelope.setLabel("left", "Амплитуда")
        self._plot_envelope.showGrid(x=True, y=True, alpha=0.3)
        self._plot_envelope.setXLink(self._plot_signal)
        layout.addWidget(self._plot_envelope, stretch=1)

    def set_signal(self, signal: np.ndarray, fs: float = 1.0):
        """Set raw signal for preprocessing."""
        self._signal = signal
        self._fs = fs
        self._plot_raw()

    def _plot_raw(self):
        if self._signal is None:
            return
        self._plot_signal.clear()
        x = np.arange(len(self._signal))
        self._plot_signal.plot(x, self._signal,
                                pen=pg.mkPen("c", width=1),
                                name="Сырой")

    def _get_params(self) -> PreprocessingParams:
        return PreprocessingParams(
            detrend=self._chk_detrend.isChecked(),
            bandpass=self._chk_bandpass.isChecked(),
            f_low=self._spin_flow.value(),
            f_high=self._spin_fhigh.value(),
            normalize_amplitude=self._chk_normalize.isChecked(),
            wavelet_denoise=self._chk_wavelet.isChecked(),
        )

    def _apply_preprocessing(self):
        if self._signal is None:
            return

        params = self._get_params()
        self._result = preprocess(self._signal, fs=self._fs, params=params)
        self._update_plots()

    def _update_plots(self):
        if self._result is None:
            return

        x = np.arange(len(self._result.signal_raw))

        # Signal plot
        self._plot_signal.clear()
        self._plot_signal.plot(x, self._result.signal_raw,
                                pen=pg.mkPen("c", width=1, alpha=0.5),
                                name="Сырой")
        self._plot_signal.plot(x, self._result.signal_processed,
                                pen=pg.mkPen("y", width=1),
                                name="Обработанный")

        # Envelope plot
        self._plot_envelope.clear()
        if self._result.envelope is not None:
            self._plot_envelope.plot(x, self._result.envelope,
                                      pen=pg.mkPen("m", width=2),
                                      name="Огибающая")
