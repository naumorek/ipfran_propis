"""
Comparison view: classic vs modern mode side-by-side.

Shows:
  - Two kinetic curves side by side
  - Numerical comparison of parameters
"""

from typing import Optional

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox,
)
from PyQt6.QtCore import Qt

import pyqtgraph as pg

from ..core.pipeline import PipelineResult


class ComparisonView(QWidget):
    """Side-by-side comparison of classic and modern processing."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._classic: Optional[PipelineResult] = None
        self._modern: Optional[PipelineResult] = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Plots side by side
        plots_layout = QHBoxLayout()

        self._plot_classic = pg.PlotWidget(title="Классический R(σ)")
        self._plot_classic.setLabel("bottom", "σ (%)")
        self._plot_classic.setLabel("left", "R (мм/день)")
        self._plot_classic.showGrid(x=True, y=True, alpha=0.3)
        plots_layout.addWidget(self._plot_classic)

        self._plot_modern = pg.PlotWidget(title="Современный R(σ)")
        self._plot_modern.setLabel("bottom", "σ (%)")
        self._plot_modern.setLabel("left", "R (мм/день)")
        self._plot_modern.showGrid(x=True, y=True, alpha=0.3)
        plots_layout.addWidget(self._plot_modern)

        layout.addLayout(plots_layout, stretch=2)

        # Parameter comparison table
        group = QGroupBox("Сравнение параметров")
        group_layout = QVBoxLayout(group)

        self._param_names = [
            "te (°C)", "tn (°C)", "Td (°C)", "Sigm (%)",
            "s0", "s1 (%)", "s2", "Sig035 (%)", "N экстр.",
        ]
        self._param_keys = [
            "te", "tn", "Td", "Sigm",
            "s0", "s1", "s2", "Sig035", "n_extrema",
        ]

        self._table = QTableWidget()
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels([
            "Параметр", "Классический", "Современный", "Δ"
        ])
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._table.setRowCount(len(self._param_names))
        for i, p in enumerate(self._param_names):
            self._table.setItem(i, 0, QTableWidgetItem(p))
            for j in range(1, 4):
                self._table.setItem(i, j, QTableWidgetItem("—"))

        group_layout.addWidget(self._table)
        layout.addWidget(group, stretch=1)

    def set_results(self, classic: PipelineResult, modern: PipelineResult):
        """Set both results for comparison."""
        self._classic = classic
        self._modern = modern
        self._update_plots()
        self._update_table()

    def _plot_result(self, plot_widget, result: PipelineResult, color: str):
        """Plot one result's kinetic curve."""
        fit = result.fit_result
        if fit is None:
            return

        plot_widget.plot(
            fit.sigma_percent, fit.rate_measured,
            pen=None, symbol="o", symbolSize=5, symbolBrush=color,
        )
        if len(fit.rate_fitted) > 0:
            sigma_range = np.linspace(
                max(fit.sigma_percent.min(), 0),
                fit.sigma_percent.max(), 200
            )
            from ..core.kinetics.power_law import power_law_model
            rate_fit = power_law_model(sigma_range, fit.s0, fit.s1, fit.w)
            plot_widget.plot(sigma_range, rate_fit, pen=pg.mkPen(color, width=2))

    def _update_plots(self):
        self._plot_classic.clear()
        self._plot_modern.clear()

        if self._classic is not None:
            self._plot_result(self._plot_classic, self._classic, "c")
        if self._modern is not None:
            self._plot_result(self._plot_modern, self._modern, "y")

    def _update_table(self):
        if self._classic is None or self._modern is None:
            return

        for i, key in enumerate(self._param_keys):
            cv = getattr(self._classic, key, 0)
            mv = getattr(self._modern, key, 0)
            fmt = ".4f" if key not in ("n_extrema",) else "d"

            if fmt == "d":
                self._table.item(i, 1).setText(str(int(cv)))
                self._table.item(i, 2).setText(str(int(mv)))
                self._table.item(i, 3).setText(str(int(mv - cv)))
            else:
                self._table.item(i, 1).setText(f"{cv:{fmt}}")
                self._table.item(i, 2).setText(f"{mv:{fmt}}")
                delta = mv - cv
                self._table.item(i, 3).setText(f"{delta:+{fmt}}")
