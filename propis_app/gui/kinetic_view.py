"""
Kinetic curve visualization view.

Shows:
  - R(ΔT) — growth rate vs supercooling (mm/day)
  - R(σ) — growth rate vs supersaturation (μm/min)
  - Reference curves (Cfe=0, Cfe=16ppm)
  - Fitted curve overlay
  - Dead zone markers
"""

from typing import Optional

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel,
    QGroupBox, QPushButton,
)

import pyqtgraph as pg

from ..core.batch import ProcessingResult
from ..core.reference_curves import ReferenceManager, get_placeholder_references


class KineticView(QWidget):
    """Kinetic curve visualization."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._result: Optional[ProcessingResult] = None
        self._ref_manager = ReferenceManager()
        self._ref_manager.load_defaults()
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Controls
        ctrl = QHBoxLayout()

        self._chk_ch1 = QCheckBox("Канал 1 (470nm)")
        self._chk_ch1.setChecked(True)
        self._chk_ch1.stateChanged.connect(self._update_plots)
        ctrl.addWidget(self._chk_ch1)

        self._chk_ch2 = QCheckBox("Канал 2 (590nm)")
        self._chk_ch2.setChecked(True)
        self._chk_ch2.stateChanged.connect(self._update_plots)
        ctrl.addWidget(self._chk_ch2)

        self._chk_fit = QCheckBox("Фиттинг")
        self._chk_fit.setChecked(True)
        self._chk_fit.stateChanged.connect(self._update_plots)
        ctrl.addWidget(self._chk_fit)

        self._chk_ref = QCheckBox("Эталоны")
        self._chk_ref.setChecked(True)
        self._chk_ref.stateChanged.connect(self._update_plots)
        ctrl.addWidget(self._chk_ref)

        ctrl.addStretch()
        layout.addLayout(ctrl)

        # Plot R(ΔT)
        self._plot_dt = pg.PlotWidget(
            title="Кинетическая кривая R(ΔT)"
        )
        self._plot_dt.setLabel("bottom", "Переохлаждение ΔT (°C)")
        self._plot_dt.setLabel("left", "Скорость роста R (мм/день)")
        self._plot_dt.addLegend()
        self._plot_dt.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self._plot_dt, stretch=1)

        # Plot R(σ)
        self._plot_sigma = pg.PlotWidget(
            title="Кинетическая кривая R(σ)"
        )
        self._plot_sigma.setLabel("bottom", "Перенасыщение σ (%)")
        self._plot_sigma.setLabel("left", "Скорость роста R (мкм/мин)")
        self._plot_sigma.addLegend()
        self._plot_sigma.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self._plot_sigma, stretch=1)

        # Parameters display
        self._lbl_params = QLabel("Параметры: —")
        layout.addWidget(self._lbl_params)

    def set_result(self, result: ProcessingResult):
        """Display processing result."""
        self._result = result
        self._update_plots()
        self._update_params_label()

    def _update_params_label(self):
        if self._result is None:
            return
        r = self._result
        self._lbl_params.setText(
            f"te={r.te:.2f}°C  tn={r.tn:.2f}°C  "
            f"Td={r.td:.2f}°C  s1={r.s1:.4f}  "
            f"s0={r.s0:.4f}  s2={r.s2:.4f}  "
            f"Sig035={r.sig035:.2f}"
        )

    def _update_plots(self):
        self._plot_dt.clear()
        self._plot_sigma.clear()

        if self._result is None:
            return

        r = self._result

        # Plot data points — R(ΔT)
        if self._chk_ch1.isChecked() and r.fit_ch1 is not None:
            fit = r.fit_ch1
            # Convert σ to ΔT approximately (σ ≈ const * ΔT for small ΔT)
            # For display, use measured supercooling if available
            self._plot_sigma.plot(
                fit.sigma_percent, fit.rate_measured,
                pen=None, symbol="o", symbolSize=5,
                symbolBrush="c", name="CH1 данные",
            )
            if self._chk_fit.isChecked():
                # Fitted curve
                sigma_fit = np.linspace(
                    max(fit.sigma_percent.min(), 0),
                    fit.sigma_percent.max(), 200
                )
                from ..core.kinetics.power_law import power_law_model
                rate_fit = power_law_model(sigma_fit, fit.s0, fit.s1, fit.w)
                self._plot_sigma.plot(
                    sigma_fit, rate_fit,
                    pen=pg.mkPen("c", width=2), name="CH1 фит",
                )

        if self._chk_ch2.isChecked() and r.fit_ch2 is not None:
            fit = r.fit_ch2
            self._plot_sigma.plot(
                fit.sigma_percent, fit.rate_measured,
                pen=None, symbol="s", symbolSize=5,
                symbolBrush="y", name="CH2 данные",
            )
            if self._chk_fit.isChecked():
                sigma_fit = np.linspace(
                    max(fit.sigma_percent.min(), 0),
                    fit.sigma_percent.max(), 200
                )
                from ..core.kinetics.power_law import power_law_model
                rate_fit = power_law_model(sigma_fit, fit.s0, fit.s1, fit.w)
                self._plot_sigma.plot(
                    sigma_fit, rate_fit,
                    pen=pg.mkPen("y", width=2), name="CH2 фит",
                )

        # Reference curves on R(ΔT) plot
        if self._chk_ref.isChecked():
            clean = self._ref_manager.get_curve("Cfe=0")
            contam = self._ref_manager.get_curve("Cfe=16ppm")

            if clean is not None:
                self._plot_dt.plot(
                    clean.supercooling, clean.rate_mm_day,
                    pen=pg.mkPen("g", width=2, style=pg.QtCore.Qt.PenStyle.DashLine),
                    name="Cfe=0 (чистый)",
                )
            if contam is not None:
                self._plot_dt.plot(
                    contam.supercooling, contam.rate_mm_day,
                    pen=pg.mkPen("r", width=2, style=pg.QtCore.Qt.PenStyle.DashLine),
                    name="Cfe=16ppm",
                )

        # Dead zone marker
        if r.td > 0:
            td_line = pg.InfiniteLine(
                pos=r.td, angle=90,
                pen=pg.mkPen("m", width=1, style=pg.QtCore.Qt.PenStyle.DotLine),
                label=f"Td={r.td:.2f}°C",
            )
            self._plot_dt.addItem(td_line)
