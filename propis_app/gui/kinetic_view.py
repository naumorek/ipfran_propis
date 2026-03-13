"""
Kinetic curve visualization view.

Shows:
  - R(ΔT) — growth rate vs supercooling (mm/day)
  - R(σ) — growth rate vs supersaturation (mm/day)
  - Reference curves (Cfe=0, Cfe=16ppm)
  - Fitted curve overlay
  - Dead zone markers
"""

from typing import Optional

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel,
)

import pyqtgraph as pg

from PyQt6.QtCore import Qt

from ..core.pipeline import PipelineResult
from ..core.reference_curves import ReferenceManager


class KineticView(QWidget):
    """Kinetic curve visualization."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._result: Optional[PipelineResult] = None
        self._ref_manager = ReferenceManager()
        self._ref_manager.load_defaults()
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Controls
        ctrl = QHBoxLayout()

        self._chk_data = QCheckBox("Данные")
        self._chk_data.setChecked(True)
        self._chk_data.stateChanged.connect(self._update_plots)
        ctrl.addWidget(self._chk_data)

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
        self._plot_dt = pg.PlotWidget(title="R(ΔT) — скорость роста")
        self._plot_dt.setLabel("bottom", "Переохлаждение ΔT (°C)")
        self._plot_dt.setLabel("left", "R (мм/день)")
        self._plot_dt.addLegend()
        self._plot_dt.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self._plot_dt, stretch=1)

        # Plot R(σ)
        self._plot_sigma = pg.PlotWidget(title="R(σ) — кинетическая кривая")
        self._plot_sigma.setLabel("bottom", "Перенасыщение σ (%)")
        self._plot_sigma.setLabel("left", "R (мм/день)")
        self._plot_sigma.addLegend()
        self._plot_sigma.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self._plot_sigma, stretch=1)

        # Parameters display
        self._lbl_params = QLabel("Параметры: —")
        layout.addWidget(self._lbl_params)

    def set_result(self, result: PipelineResult):
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
            f"Td={r.Td:.2f}°C  Sigm={r.Sigm:.3f}%  "
            f"s0={r.s0:.4f}  s1={r.s1:.4f}  s2={r.s2:.4f}  "
            f"Sig035={r.Sig035:.2f}  [{r.mode}]"
        )

    def _update_plots(self):
        self._plot_dt.clear()
        self._plot_sigma.clear()

        if self._result is None:
            return

        r = self._result
        gr = r.growth_rate
        fit = r.fit_result

        if gr is None or fit is None:
            return

        # Data points
        if self._chk_data.isChecked():
            sigma = fit.sigma_percent
            rate = fit.rate_measured

            # R(σ) plot
            self._plot_sigma.plot(
                sigma, rate,
                pen=None, symbol="o", symbolSize=5,
                symbolBrush="c", name="Данные",
            )

            # R(ΔT) plot — use supercooling from growth_rate
            dt = gr.supercooling
            rate_dt = gr.rate_mm_day
            mask = dt > 0
            if np.any(mask):
                self._plot_dt.plot(
                    dt[mask], rate_dt[mask],
                    pen=None, symbol="o", symbolSize=5,
                    symbolBrush="c", name="Данные",
                )

        # Fitted curve
        if self._chk_fit.isChecked() and fit is not None:
            from ..core.kinetics.power_law import power_law_model
            sigma_range = np.linspace(
                max(fit.sigma_percent.min(), 0),
                fit.sigma_percent.max(), 200
            )
            rate_fit = power_law_model(sigma_range, fit.s0, fit.s1, fit.w)
            self._plot_sigma.plot(
                sigma_range, rate_fit,
                pen=pg.mkPen("g", width=2), name="Фит",
            )

        # Reference curves
        if self._chk_ref.isChecked():
            clean = self._ref_manager.get_curve("Cfe=0")
            contam = self._ref_manager.get_curve("Cfe=16ppm")
            if clean is not None:
                self._plot_dt.plot(
                    clean.supercooling, clean.rate_mm_day,
                    pen=pg.mkPen("g", width=2, style=Qt.PenStyle.DashLine),
                    name="Cfe=0",
                )
            if contam is not None:
                self._plot_dt.plot(
                    contam.supercooling, contam.rate_mm_day,
                    pen=pg.mkPen("r", width=2, style=Qt.PenStyle.DashLine),
                    name="Cfe=16ppm",
                )

        # Dead zone marker on R(ΔT)
        if r.Td > 0:
            td_line = pg.InfiniteLine(
                pos=r.Td, angle=90,
                pen=pg.mkPen("m", width=1, style=Qt.PenStyle.DotLine),
                label=f"Td={r.Td:.2f}°C",
            )
            self._plot_dt.addItem(td_line)
