"""
Signal visualization view.

Shows:
  - Raw interferometric signal (LED1 and LED2)
  - Temperature curve
  - Interactive boundary markers (n1, n2, im, isat) as draggable lines
  - Zoom and pan
"""

from typing import Optional

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox,
    QGroupBox, QCheckBox,
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

        # Controls
        ctrl_layout = QHBoxLayout()

        self._chk_led1 = QCheckBox("LED1 (470nm)")
        self._chk_led1.setChecked(True)
        self._chk_led1.stateChanged.connect(self._update_plot)
        ctrl_layout.addWidget(self._chk_led1)

        self._chk_led2 = QCheckBox("LED2 (590nm)")
        self._chk_led2.setChecked(True)
        self._chk_led2.stateChanged.connect(self._update_plot)
        ctrl_layout.addWidget(self._chk_led2)

        self._chk_temp = QCheckBox("Температура")
        self._chk_temp.setChecked(True)
        self._chk_temp.stateChanged.connect(self._update_plot)
        ctrl_layout.addWidget(self._chk_temp)

        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)

        # Boundary controls
        bounds_group = QGroupBox("Границы обработки")
        bounds_layout = QHBoxLayout(bounds_group)

        self._spin_n1 = QSpinBox()
        self._spin_n1.setRange(0, 100000)
        self._spin_n1.setValue(100)
        self._spin_n1.setPrefix("n1: ")
        self._spin_n1.valueChanged.connect(self._update_boundaries)
        bounds_layout.addWidget(self._spin_n1)

        self._spin_n2 = QSpinBox()
        self._spin_n2.setRange(0, 100000)
        self._spin_n2.setValue(10000)
        self._spin_n2.setPrefix("n2: ")
        self._spin_n2.valueChanged.connect(self._update_boundaries)
        bounds_layout.addWidget(self._spin_n2)

        self._spin_im = QSpinBox()
        self._spin_im.setRange(0, 100000)
        self._spin_im.setValue(6500)
        self._spin_im.setPrefix("im: ")
        self._spin_im.valueChanged.connect(self._update_boundaries)
        bounds_layout.addWidget(self._spin_im)

        self._spin_isat = QSpinBox()
        self._spin_isat.setRange(0, 100000)
        self._spin_isat.setValue(9000)
        self._spin_isat.setPrefix("isat: ")
        self._spin_isat.valueChanged.connect(self._update_boundaries)
        bounds_layout.addWidget(self._spin_isat)

        self._spin_parasitic = QSpinBox()
        self._spin_parasitic.setRange(0, 50)
        self._spin_parasitic.setValue(0)
        self._spin_parasitic.setPrefix("Паразит: ")
        bounds_layout.addWidget(self._spin_parasitic)

        layout.addWidget(bounds_group)

        # Plot widget
        self._plot_widget = pg.PlotWidget(title="Интерференционный сигнал")
        self._plot_widget.setLabel("bottom", "Отсчёт")
        self._plot_widget.setLabel("left", "Сигнал (В)")
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

        # Boundary lines
        self._lines = {}
        colors = {"n1": "g", "n2": "g", "im": "r", "isat": "b"}
        for name, color in colors.items():
            line = pg.InfiniteLine(
                pos=0, angle=90, pen=pg.mkPen(color, width=2, style=Qt.PenStyle.DashLine),
                movable=True, label=name,
                labelOpts={"position": 0.95, "color": color},
            )
            self._plot_widget.addItem(line)
            self._lines[name] = line

        # Connect line drag to spinboxes
        self._lines["n1"].sigPositionChanged.connect(
            lambda l: self._spin_n1.setValue(int(l.value())))
        self._lines["n2"].sigPositionChanged.connect(
            lambda l: self._spin_n2.setValue(int(l.value())))
        self._lines["im"].sigPositionChanged.connect(
            lambda l: self._spin_im.setValue(int(l.value())))
        self._lines["isat"].sigPositionChanged.connect(
            lambda l: self._spin_isat.setValue(int(l.value())))

    def set_data(self, data: PrnData):
        """Load new PRN data and display it."""
        self._data = data

        # Update spinbox ranges
        n = data.n_samples
        for spin in [self._spin_n1, self._spin_n2, self._spin_im, self._spin_isat]:
            spin.setRange(0, n)

        # Auto-set boundaries
        self._spin_n1.setValue(100)
        self._spin_n2.setValue(min(n, 10000))
        self._spin_im.setValue(int(n * 0.4))
        self._spin_isat.setValue(int(n * 0.6))

        self._update_plot()

    def _update_plot(self):
        if self._data is None:
            return

        self._plot_widget.clear()
        x = np.arange(self._data.n_samples)

        if self._chk_led1.isChecked():
            self._plot_widget.plot(x, self._data.led1,
                                   pen=pg.mkPen("c", width=1),
                                   name="LED1 (470nm)")

        if self._chk_led2.isChecked():
            self._plot_widget.plot(x, self._data.led2,
                                   pen=pg.mkPen("y", width=1),
                                   name="LED2 (590nm)")

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
        self._lines["n1"].setValue(self._spin_n1.value())
        self._lines["n2"].setValue(self._spin_n2.value())
        self._lines["im"].setValue(self._spin_im.value())
        self._lines["isat"].setValue(self._spin_isat.value())

    def get_boundaries(self) -> tuple[int, int, int, int]:
        """Return current boundary values (n1, n2, im, isat)."""
        return (
            self._spin_n1.value(),
            self._spin_n2.value(),
            self._spin_im.value(),
            self._spin_isat.value(),
        )
