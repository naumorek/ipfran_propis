"""
Signal visualization view.

Shows:
  - Raw interferometric signal (LED1/LED2 ratio) and individual LEDs
  - Temperature curve
  - Interactive boundary markers (n1, n2, im, isat, im1, isat1) as draggable lines
  - Zoom and pan
"""

from typing import Optional

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox,
    QGroupBox, QCheckBox, QDoubleSpinBox,
)
from PyQt6.QtCore import Qt

import pyqtgraph as pg

from ..core.prn_reader import PrnData


class SignalView(QWidget):
    """Interactive signal visualization with boundary controls."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: Optional[PrnData] = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Controls row 1: display toggles
        ctrl_layout = QHBoxLayout()

        self._chk_ratio = QCheckBox("LED1/LED2")
        self._chk_ratio.setChecked(True)
        self._chk_ratio.stateChanged.connect(self._update_plot)
        ctrl_layout.addWidget(self._chk_ratio)

        self._chk_led1 = QCheckBox("LED1")
        self._chk_led1.setChecked(False)
        self._chk_led1.stateChanged.connect(self._update_plot)
        ctrl_layout.addWidget(self._chk_led1)

        self._chk_led2 = QCheckBox("LED2")
        self._chk_led2.setChecked(False)
        self._chk_led2.stateChanged.connect(self._update_plot)
        ctrl_layout.addWidget(self._chk_led2)

        self._chk_temp = QCheckBox("Температура")
        self._chk_temp.setChecked(True)
        self._chk_temp.stateChanged.connect(self._update_plot)
        ctrl_layout.addWidget(self._chk_temp)

        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)

        # Boundary controls: 6 boundaries + tn
        bounds_group = QGroupBox("Границы цикла (относительно n1)")
        bounds_layout = QHBoxLayout(bounds_group)

        self._spin_n1 = QSpinBox()
        self._spin_n1.setRange(0, 100000)
        self._spin_n1.setValue(1)
        self._spin_n1.setPrefix("n1: ")
        self._spin_n1.valueChanged.connect(self._update_boundaries)
        bounds_layout.addWidget(self._spin_n1)

        self._spin_n2 = QSpinBox()
        self._spin_n2.setRange(0, 100000)
        self._spin_n2.setValue(10000)
        self._spin_n2.setPrefix("n2: ")
        self._spin_n2.valueChanged.connect(self._update_boundaries)
        bounds_layout.addWidget(self._spin_n2)

        self._spin_im1 = QSpinBox()
        self._spin_im1.setRange(0, 100000)
        self._spin_im1.setValue(5000)
        self._spin_im1.setPrefix("im1: ")
        self._spin_im1.setToolTip("Грубая граница мёртвой зоны (конец зоны роста)")
        self._spin_im1.valueChanged.connect(self._update_boundaries)
        bounds_layout.addWidget(self._spin_im1)

        self._spin_im = QSpinBox()
        self._spin_im.setRange(0, 100000)
        self._spin_im.setValue(6000)
        self._spin_im.setPrefix("im: ")
        self._spin_im.setToolTip("Точная граница мёртвой зоны")
        self._spin_im.valueChanged.connect(self._update_boundaries)
        bounds_layout.addWidget(self._spin_im)

        self._spin_isat = QSpinBox()
        self._spin_isat.setRange(0, 100000)
        self._spin_isat.setValue(7000)
        self._spin_isat.setPrefix("isat: ")
        self._spin_isat.setToolTip("Точная граница растворения")
        self._spin_isat.valueChanged.connect(self._update_boundaries)
        bounds_layout.addWidget(self._spin_isat)

        self._spin_isat1 = QSpinBox()
        self._spin_isat1.setRange(0, 100000)
        self._spin_isat1.setValue(8000)
        self._spin_isat1.setPrefix("isat1: ")
        self._spin_isat1.setToolTip("Грубая граница растворения")
        self._spin_isat1.valueChanged.connect(self._update_boundaries)
        bounds_layout.addWidget(self._spin_isat1)

        layout.addWidget(bounds_group)

        # tn manual input
        tn_layout = QHBoxLayout()
        tn_layout.addWidget(QLabel("tn (°C):"))
        self._spin_tn = QDoubleSpinBox()
        self._spin_tn.setRange(0.0, 100.0)
        self._spin_tn.setDecimals(2)
        self._spin_tn.setValue(0.0)
        self._spin_tn.setSpecialValueText("авто (=te)")
        self._spin_tn.setToolTip("Температура насыщения. 0 = использовать te")
        tn_layout.addWidget(self._spin_tn)
        tn_layout.addStretch()
        layout.addLayout(tn_layout)

        # Plot widget: signal
        self._plot_widget = pg.PlotWidget(title="Интерференционный сигнал")
        self._plot_widget.setLabel("bottom", "Отсчёт")
        self._plot_widget.setLabel("left", "Сигнал")
        self._plot_widget.addLegend()
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self._plot_widget, stretch=3)

        # Temperature plot (linked X axis)
        self._temp_widget = pg.PlotWidget(title="Температура")
        self._temp_widget.setLabel("bottom", "Отсчёт")
        self._temp_widget.setLabel("left", "T (°C)")
        self._temp_widget.showGrid(x=True, y=True, alpha=0.3)
        self._temp_widget.setXLink(self._plot_widget)
        layout.addWidget(self._temp_widget, stretch=1)

        # Boundary lines on signal plot
        self._lines = {}
        line_defs = {
            "im1": ("m", "im1 (рост)"),
            "im": ("r", "im (мз)"),
            "isat": ("b", "isat (раств.)"),
            "isat1": ("c", "isat1 (грубая)"),
        }
        for name, (color, label) in line_defs.items():
            line = pg.InfiniteLine(
                pos=0, angle=90,
                pen=pg.mkPen(color, width=2, style=Qt.PenStyle.DashLine),
                movable=True, label=label,
                labelOpts={"position": 0.95, "color": color},
            )
            self._plot_widget.addItem(line)
            self._lines[name] = line

        # Connect line drag to spinboxes
        self._lines["im1"].sigPositionChanged.connect(
            lambda l: self._spin_im1.setValue(int(l.value())))
        self._lines["im"].sigPositionChanged.connect(
            lambda l: self._spin_im.setValue(int(l.value())))
        self._lines["isat"].sigPositionChanged.connect(
            lambda l: self._spin_isat.setValue(int(l.value())))
        self._lines["isat1"].sigPositionChanged.connect(
            lambda l: self._spin_isat1.setValue(int(l.value())))

    def set_data(self, data: PrnData):
        """Load new PRN data and display it."""
        self._data = data
        n = data.n_samples

        for spin in [self._spin_n1, self._spin_n2, self._spin_im1,
                     self._spin_im, self._spin_isat, self._spin_isat1]:
            spin.setRange(0, n)

        # Auto-set boundaries (rough estimates, user will adjust)
        self._spin_n1.setValue(1)
        self._spin_n2.setValue(min(n, 10000))
        self._spin_im1.setValue(int(n * 0.35))
        self._spin_im.setValue(int(n * 0.45))
        self._spin_isat.setValue(int(n * 0.55))
        self._spin_isat1.setValue(int(n * 0.65))

        self._update_plot()

    def _update_plot(self):
        if self._data is None:
            return

        self._plot_widget.clear()
        x = np.arange(self._data.n_samples)

        if self._chk_ratio.isChecked():
            led2_safe = np.where(
                np.abs(self._data.led2) < 1e-10, 1e-10, self._data.led2)
            ratio = self._data.led1 / led2_safe
            self._plot_widget.plot(x, ratio,
                                   pen=pg.mkPen("w", width=1),
                                   name="LED1/LED2")

        if self._chk_led1.isChecked():
            self._plot_widget.plot(x, self._data.led1,
                                   pen=pg.mkPen("c", width=1),
                                   name="LED1")

        if self._chk_led2.isChecked():
            self._plot_widget.plot(x, self._data.led2,
                                   pen=pg.mkPen("y", width=1),
                                   name="LED2")

        # Re-add boundary lines
        for line in self._lines.values():
            self._plot_widget.addItem(line)

        # Temperature
        self._temp_widget.clear()
        if self._chk_temp.isChecked():
            self._temp_widget.plot(x, self._data.temp_c,
                                    pen=pg.mkPen("r", width=1))

        self._update_boundaries()

    def _update_boundaries(self):
        self._lines["im1"].setValue(self._spin_im1.value())
        self._lines["im"].setValue(self._spin_im.value())
        self._lines["isat"].setValue(self._spin_isat.value())
        self._lines["isat1"].setValue(self._spin_isat1.value())

    def get_boundaries(self) -> dict:
        """Return current boundary values."""
        return {
            "n1": self._spin_n1.value(),
            "n2": self._spin_n2.value(),
            "im": self._spin_im.value(),
            "isat": self._spin_isat.value(),
            "im1": self._spin_im1.value(),
            "isat1": self._spin_isat1.value(),
        }

    def get_tn_manual(self) -> float | None:
        """Return manual tn or None if auto."""
        val = self._spin_tn.value()
        return val if val > 0 else None
