"""
Comparison view: classic vs modern mode side-by-side.

Shows:
  - Two kinetic curves side by side
  - Numerical comparison of parameters
  - Residual comparison
"""

from typing import Optional

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox,
)

import pyqtgraph as pg

from ..core.batch import ProcessingResult


class ComparisonView(QWidget):
    """Side-by-side comparison of classic and modern processing."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._classic: Optional[ProcessingResult] = None
        self._modern: Optional[ProcessingResult] = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Plots side by side
        plots_layout = QHBoxLayout()

        # Classic R(σ)
        self._plot_classic = pg.PlotWidget(title="Классический режим R(σ)")
        self._plot_classic.setLabel("bottom", "σ (%)")
        self._plot_classic.setLabel("left", "R (мкм/мин)")
        self._plot_classic.showGrid(x=True, y=True, alpha=0.3)
        plots_layout.addWidget(self._plot_classic)

        # Modern R(σ)
        self._plot_modern = pg.PlotWidget(title="Современный режим R(σ)")
        self._plot_modern.setLabel("bottom", "σ (%)")
        self._plot_modern.setLabel("left", "R (мкм/мин)")
        self._plot_modern.showGrid(x=True, y=True, alpha=0.3)
        plots_layout.addWidget(self._plot_modern)

        layout.addLayout(plots_layout, stretch=2)

        # Parameter comparison table
        group = QGroupBox("Сравнение параметров")
        group_layout = QVBoxLayout(group)

        self._table = QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels([
            "Параметр", "Классический", "Современный", "Δ"
        ])
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

        params = ["tn (°C)", "Td (°C)", "s1 (%)", "s0", "s2", "Sig035 (%)"]
        self._table.setRowCount(len(params))
        for i, p in enumerate(params):
            self._table.setItem(i, 0, QTableWidgetItem(p))
            for j in range(1, 4):
                self._table.setItem(i, j, QTableWidgetItem("—"))

        group_layout.addWidget(self._table)
        layout.addWidget(group, stretch=1)

    def set_results(self, classic: ProcessingResult, modern: ProcessingResult):
        """Set both results for comparison."""
        self._classic = classic
        self._modern = modern
        self._update_plots()
        self._update_table()

    def _update_plots(self):
        self._plot_classic.clear()
        self._plot_modern.clear()

        if self._classic is not None and self._classic.fit_ch1 is not None:
            fit = self._classic.fit_ch1
            self._plot_classic.plot(
                fit.sigma_percent, fit.rate_measured,
                pen=None, symbol="o", symbolSize=5, symbolBrush="c",
            )
            if len(fit.rate_fitted) > 0:
                sigma_fit = np.linspace(
                    max(fit.sigma_percent.min(), 0),
                    fit.sigma_percent.max(), 200
                )
                from ..core.kinetics.power_law import power_law_model
                rate_fit = power_law_model(sigma_fit, fit.s0, fit.s1, fit.w)
                self._plot_classic.plot(
                    sigma_fit, rate_fit, pen=pg.mkPen("c", width=2),
                )

        if self._modern is not None and self._modern.fit_ch1 is not None:
            fit = self._modern.fit_ch1
            self._plot_modern.plot(
                fit.sigma_percent, fit.rate_measured,
                pen=None, symbol="o", symbolSize=5, symbolBrush="y",
            )
            if len(fit.rate_fitted) > 0:
                sigma_fit = np.linspace(
                    max(fit.sigma_percent.min(), 0),
                    fit.sigma_percent.max(), 200
                )
                from ..core.kinetics.power_law import power_law_model
                rate_fit = power_law_model(sigma_fit, fit.s0, fit.s1, fit.w)
                self._plot_modern.plot(
                    sigma_fit, rate_fit, pen=pg.mkPen("y", width=2),
                )

    def _update_table(self):
        if self._classic is None or self._modern is None:
            return

        c = self._classic
        m = self._modern

        params = [
            ("tn (°C)", c.tn, m.tn),
            ("Td (°C)", c.td, m.td),
            ("s1 (%)", c.s1, m.s1),
            ("s0", c.s0, m.s0),
            ("s2", c.s2, m.s2),
            ("Sig035 (%)", c.sig035, m.sig035),
        ]

        for i, (name, cv, mv) in enumerate(params):
            self._table.item(i, 0).setText(name)
            self._table.item(i, 1).setText(f"{cv:.4f}")
            self._table.item(i, 2).setText(f"{mv:.4f}")
            delta = mv - cv
            self._table.item(i, 3).setText(f"{delta:+.4f}")
