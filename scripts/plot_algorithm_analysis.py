#!/usr/bin/env python3
"""
Аналитическая визуализация алгоритма mathcad_exact.py.

Показывает: что реально измеряется, а что дорисовывается алгоритмом.
Генерирует аннотированный JPG с 4 графиками.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.mathcad_exact import run_mathcad


def main():
    # Запуск алгоритма
    r = run_mathcad(
        '020326 KDP нейтр XXXIII п.15/__020326_2.prn',
        n1=10, n2=8000, im=5963, isat=6692, im1=5120, isat1=7580,
        Salt=1, Acid=0, Face=0, d=3.3, ww=1, k1=0.15, span1=0.2, l=0.01,
        tn_manual=48.60,
    )
    if r is None:
        print("Ошибка расчёта!")
        return

    # Извлечение данных
    signal_raw = r['signal_raw']
    S1 = r['S1']
    e = r['e']           # 17 экстремумов (реальные данные!)
    y = r['y']            # d-step grid: 2297 точек
    L_arr = r['L_arr']    # кумулятивная толщина
    T_smooth = r['T_smooth']  # LOESS(L)
    z = r['z']            # скорости (2296 точек)
    VX, VY = r['VX'], r['VY']
    F1 = r['F1']
    Si, Si1 = r['Si'], r['Si1']
    te, tn, Td = r['te'], r['tn'], r['Td']
    Sig035 = r['Sig035']
    im1, im, isat, isat1 = r['im1'], r['im'], r['isat'], r['isat1']
    y0 = r['y0']
    m = r['m']
    co2, co3, co4 = r['co2'], r['co3'], r['co4']
    Cn = co2 + co3 * tn + co4 * tn ** 2

    n_ext = len(e)
    n_dstep = len(y)
    n_rates = len(z)

    # ========================================================================
    # РИСУЕМ
    # ========================================================================
    fig = plt.figure(figsize=(20, 22))
    fig.patch.set_facecolor('white')

    # Цвета
    C_RAW = '#888888'      # сырые данные
    C_REAL = '#D62728'     # реальные точки (экстремумы)
    C_INTERP = '#1F77B4'   # интерполированные данные
    C_SMOOTH = '#2CA02C'   # LOESS сглаженное
    C_ZONE = '#FFE0B2'     # зона dead zone

    # ====== SUBPLOT 1: Сырой сигнал + экстремумы ======
    ax1 = fig.add_subplot(3, 2, 1)
    x_axis = np.arange(m)
    ax1.plot(x_axis, signal_raw, color=C_RAW, linewidth=0.3, alpha=0.7, label=f'Сигнал ({m} отсчётов)')
    ax1.axhline(y=y0, color='green', linewidth=0.8, linestyle='--', alpha=0.5, label=f'Baseline y0={y0:.3f}')

    # Зоны
    ax1.axvspan(im1, im, alpha=0.15, color='orange', label='Dead zone')
    ax1.axvspan(im, isat, alpha=0.10, color='red')
    ax1.axvspan(isat, isat1, alpha=0.08, color='purple')

    # Экстремумы — РЕАЛЬНЫЕ точки
    ax1.scatter(e[:, 0], e[:, 1], color=C_REAL, s=60, zorder=5, edgecolors='black',
                linewidths=0.5, label=f'{n_ext} экстремумов (РЕАЛЬНЫЕ)')

    ax1.set_xlabel('Позиция (отсчёты)')
    ax1.set_ylabel('Сигнал (LED1/LED2)')
    ax1.set_title('1. СЫРОЙ СИГНАЛ + ЭКСТРЕМУМЫ', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=7, loc='upper right')

    # Аннотация
    ax1.text(0.02, 0.02,
             f'{m} отсчётов × 1 сек = {m/3600:.1f} час\n'
             f'{n_ext} экстремумов = {n_ext} реальных\n'
             f'точек роста кристалла\n'
             f'(каждая = λ/4 ≈ 0.12 мкм толщины)',
             transform=ax1.transAxes, fontsize=8, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
             verticalalignment='bottom')

    # ====== SUBPLOT 2: Фаза + L ======
    ax2 = fig.add_subplot(3, 2, 2)

    # Фаза (левая ось)
    color_phase = C_INTERP
    ax2.plot(y[:, 0], y[:, 2], color=color_phase, linewidth=0.5, alpha=0.7,
             label=f'Фаза ({n_dstep} точек, arcsin интерп.)')
    # Отметить позиции экстремумов на фазе
    for j, ext in enumerate(e):
        pos = ext[0]
        didx = int(pos / 3.3)
        if 0 <= didx < len(y):
            ax2.plot(pos, y[didx, 2], 'ro', markersize=4, zorder=5)

    ax2.set_xlabel('Позиция (отсчёты)')
    ax2.set_ylabel('Фаза (рад)', color=color_phase)
    ax2.tick_params(axis='y', labelcolor=color_phase)

    # L (правая ось)
    ax2r = ax2.twinx()
    ax2r.plot(y[:, 0], L_arr, color='gray', linewidth=0.3, alpha=0.5, label='L (raw)')
    ax2r.plot(y[:, 0], T_smooth, color=C_SMOOTH, linewidth=1.5, label='L_smooth (LOESS)')
    ax2r.set_ylabel('Толщина L (полосы)', color=C_SMOOTH)
    ax2r.tick_params(axis='y', labelcolor=C_SMOOTH)

    ax2.set_title('2. ФАЗА + КУМУЛЯТИВНАЯ ТОЛЩИНА L', fontsize=12, fontweight='bold')

    # Комбинированная легенда
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper left')

    ax2.text(0.02, 0.02,
             f'{n_dstep} точек d-step сетки (шаг {3.3:.1f})\n'
             f'Из них РЕАЛЬНЫХ: {n_ext} экстремумов\n'
             f'Остальное: arcsin-интерполяция\n'
             f'L_smooth: LOESS degree=2, span=0.15',
             transform=ax2.transAxes, fontsize=8, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
             verticalalignment='bottom')

    # ====== SUBPLOT 3: R(σ%) в мкм/мин ======
    ax3 = fig.add_subplot(3, 2, 3)

    # Точки R
    ax3.scatter(z[:, 3], z[:, 1], color=C_RAW, s=1, alpha=0.3,
                label=f'R = dL_smooth/dt ({n_rates} точек)')

    # LOESS F1
    sigma_grid = np.linspace(max(0, VX[0]), VX[-1], 500)
    F1_vals = np.array([float(F1(s)) for s in sigma_grid])
    ax3.plot(sigma_grid, F1_vals, color=C_SMOOTH, linewidth=2,
             label=f'F1 = LOESS(R, σ), span=0.2')

    # Порог Sig035
    ax3.axhline(y=0.35, color='red', linewidth=0.8, linestyle='--', alpha=0.7)
    ax3.axvline(x=Sig035, color='red', linewidth=0.8, linestyle=':', alpha=0.7)
    ax3.text(Sig035 + 0.1, 0.36, f'Sig035={Sig035:.2f}%', fontsize=8, color='red')

    # s2 линия
    if abs(r['s2']) < 10:
        s2_sigma = np.linspace(0, 3, 100)
        # sqrt(R) = Q1 + Q2 * sigma → R = (Q1 + Q2*sigma)^2
        # Примерная линия
        ax3.axvline(x=r['s2'], color='blue', linewidth=0.8, linestyle=':', alpha=0.5)
        ax3.text(r['s2'] + 0.05, 0.01, f's2={r["s2"]:.2f}%', fontsize=7, color='blue')

    ax3.set_xlabel('Пересыщение σ (%)')
    ax3.set_ylabel('Скорость R (мкм/мин)')
    ax3.set_title('3. СКОРОСТЬ РОСТА R(σ)', fontsize=12, fontweight='bold')
    ax3.set_xlim(-0.5, 6)
    ax3.set_ylim(-0.05, max(z[:, 1]) * 1.1)
    ax3.legend(fontsize=7, loc='upper left')

    ax3.text(0.55, 0.02,
             f'{n_rates} точек R = производная\n'
             f'LOESS-сглаженной толщины L\n'
             f'F1 = ещё одно LOESS поверх R(σ)\n'
             f'Из 17 реальных → 2296 точек R\n'
             f'через двойное LOESS сглаживание',
             transform=ax3.transAxes, fontsize=8, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
             verticalalignment='bottom')

    # ====== SUBPLOT 4: R(σ%) в мм/день (практические единицы) ======
    ax4 = fig.add_subplot(3, 2, 4)

    K = 1.441  # мкм/мин → мм/день

    ax4.scatter(z[:, 3], z[:, 1] * K, color=C_RAW, s=1, alpha=0.3,
                label=f'R ({n_rates} точек)')
    ax4.plot(sigma_grid, F1_vals * K, color=C_SMOOTH, linewidth=2,
             label='F1 (LOESS)')

    # Порог Sig035 в мм/день
    ax4.axhline(y=0.35 * K, color='red', linewidth=0.8, linestyle='--', alpha=0.7)
    ax4.axvline(x=Sig035, color='red', linewidth=0.8, linestyle=':', alpha=0.7)
    ax4.text(Sig035 + 0.1, 0.35 * K + 0.02, f'Sig035={Sig035:.2f}%\n({0.35*K:.2f} мм/д)',
             fontsize=8, color='red')

    ax4.set_xlabel('Пересыщение σ (%)')
    ax4.set_ylabel('Скорость R (мм/день)')
    ax4.set_title('4. СКОРОСТЬ РОСТА R(σ) — мм/день', fontsize=12, fontweight='bold')
    ax4.set_xlim(-0.5, 6)
    ax4.set_ylim(-0.1, max(z[:, 1]) * K * 1.1)
    ax4.legend(fontsize=7, loc='upper left')

    ax4.text(0.55, 0.02,
             f'1 мкм/мин × 1.441 = 1 мм/день\n'
             f'R max = {z[:,1].max()*K:.2f} мм/день\n'
             f'Sig035 = {Sig035:.2f}% ({0.35*K:.2f} мм/д)\n'
             f'Типичный рост KDP: 1-3 мм/день',
             transform=ax4.transAxes, fontsize=8, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
             verticalalignment='bottom')

    # ====== SUBPLOT 5: R(dT) финальный ======
    ax5 = fig.add_subplot(3, 2, 5)

    z_dT = tn - z[:, 2]
    ax5.scatter(z_dT, 1.441 * z[:, 1], color=C_RAW, s=1, alpha=0.3,
                label=f'R ({n_rates} точек)')

    # F1 на dT сетке
    dT_grid = np.linspace(0, 4, 500)
    F1_dT = np.zeros(len(dT_grid))
    for idx, dt in enumerate(dT_grid):
        T_val = tn - dt
        Cm = co2 + co3 * T_val + co4 * T_val ** 2
        if Cm > 0 and Cn > 0:
            sigma = 100.0 * np.log(Cn / Cm)
        else:
            sigma = 0
        F1_dT[idx] = float(F1(sigma))
    ax5.plot(dT_grid, 1.441 * F1_dT, color=C_SMOOTH, linewidth=2, label='F1 (LOESS)')

    # Эталоны
    ax5.plot(Si[:, 1], 1.441 * Si[:, 0], 'b-o', markersize=5, linewidth=1,
             label=f'Si (эталон, CFe=0)')
    ax5.plot(Si1[:, 1], 1.441 * Si1[:, 0], 'g-d', markersize=5, linewidth=1,
             label=f'Si1 (эталон, CFe=16ppm)')

    ax5.set_xlabel('Переохлаждение dT (°C)')
    ax5.set_ylabel('R (мм/день)')
    ax5.set_title('5. ИТОГОВЫЙ ГРАФИК R(dT) — мм/день', fontsize=12, fontweight='bold')
    ax5.set_xlim(0, 4)
    ax5.set_ylim(-0.1, 4)
    ax5.legend(fontsize=7, loc='upper left')

    ax5.text(0.55, 0.02,
             f'te={te:.2f}°C  tn={tn:.2f}°C\n'
             f'Td={Td:.2f}°C  s2={r["s2"]:.2f}\n'
             f'Sig035={Sig035:.2f}%\n'
             f'Эталоны Si/Si1: 8 точек каждый\n'
             f'(полиномиальная аппроксимация)',
             transform=ax5.transAxes, fontsize=8, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
             verticalalignment='bottom')

    # ====== ОБЩИЙ ЗАГОЛОВОК ======
    fig.suptitle(
        f'АНАЛИЗ АЛГОРИТМА: 020326_2_1  |  te={te:.2f}°C  tn={tn:.2f}°C  Td={Td:.2f}°C\n'
        f'РЕАЛЬНЫЕ ДАННЫЕ: {m} отсчётов сигнала → {n_ext} экстремумов (интерф. полос)\n'
        f'ИНТЕРПОЛЯЦИЯ: {n_ext} точек → {n_dstep} d-step → LOESS(L) → {n_rates} точек R → LOESS(R) → F1',
        fontsize=13, fontweight='bold', y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out = 'algorithm_analysis.jpg'
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'\nСохранено: {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()
