#!/usr/bin/env python3
"""
ТОЧНОЕ воспроизведение алгоритма Mathcad 2000 (_020326_5_1 / _040326_6_2).

Все переменные, формулы и порядок выполнения — как в скриншотах mathcad_print/1..30.
Единицы скорости: мкм/мин (как в Mathcad). Для графика R(dT) → ×1.441 = mm/day.

Тест: файл 020326 KDP нейтр XXXIII п.17/__020326_5_1
Mathcad результат: te=48.84, tn=48.6, Td=0.22, Sigm=0.3, s1=-0.43, s2=-0.04, Sig035≈1.96
"""

import sys
from pathlib import Path
import numpy as np
from scipy.signal import medfilt
from scipy.interpolate import interp1d

# ============================================================================
#  0) РУЧНЫЕ ПАРАМЕТРЫ (из MCD-файла __020326_5_1)
# ============================================================================
PRN_FILE = None  # будет задан ниже или из аргумента

# --- Параметры по умолчанию (020326_5_1) ---
n1 = 0
n2 = 9000
vq = 0          # 0=прямо, 1=обратно
k1 = 0.15       # коэфф. для окрестности квадратичного уточнения
dtau = 0.055    # промежуток между отсчётами (мин)
vt1 = 0.0001    # температурные пороги (грубое)
vt2 = 0.0001
im1 = 6350      # начало мёртвой зоны (грубое, ОТНОС. к n1)
isat1 = 8090    # начало растворения (грубое, ОТНОС. к n1)
im = 7227       # начало мёртвой зоны (точное, ОТНОС. к n1)
isat = 7576     # начало растворения (точное, ОТНОС. к n1)
Salt = 1        # KDP(1), DKDP(2)
Acid = 0        # нейтральный(0), кислый(1)
l = 0.01        # коэфф. сглаживания для роста
Face = 0        # призма(0), пирамида(>0)
d = 3.3         # шаг d-step (в отсчётах)
ww = 1          # весовой коэффициент
L0 = 1500       # базовое число полос
span = 0.5      # LOESS span для s1/dissolution
span1 = 0.2     # LOESS span для Sig035 и F1
smooth_window = 5  # окно medsmooth (5 для 020326, 1 для 030226)
z_split_index = None  # J1/J2 разбиение z (индекс строки, None=отключено)

# --- Ручной tn (если задан оператором) ---
tn_manual = None  # None = вычислить по формуле


# ============================================================================
#  1) ЗАГРУЗКА PRN И ИЗВЛЕЧЕНИЕ SUB-ARRAY
# ============================================================================

def _parse_time_to_seconds(time_strings):
    """HH:MM:SS → секунды от начала (из prn_reader.py)."""
    seconds = np.zeros(len(time_strings), dtype=np.float64)
    for i, ts in enumerate(time_strings):
        parts = ts.split(":")
        if len(parts) == 3:
            h, m_t, s = int(parts[0]), int(parts[1]), int(parts[2])
            seconds[i] = h * 3600 + m_t * 60 + s
    # Midnight crossing
    for i in range(1, len(seconds)):
        if seconds[i] < seconds[i - 1]:
            seconds[i:] += 86400
            break
    seconds -= seconds[0]
    return seconds


def load_prn(filepath):
    """
    Читает PRN файл → (матрица (N, 6), time_seconds (N,)).
    Столбцы: 0=index, 1=LED1, 2=LED2, 3=flag, 4=T_raw(~273+T), 5=T_C
    time_seconds: секунды от начала записи (из столбца 6 HH:MM:SS).
    """
    data = []
    time_strings = []
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            try:
                row = [float(parts[i]) for i in range(6)]
                data.append(row)
                if len(parts) >= 7:
                    time_strings.append(parts[6])
                else:
                    time_strings.append("00:00:00")
            except (ValueError, IndexError):
                continue
    time_seconds = _parse_time_to_seconds(time_strings) if time_strings else np.arange(len(data), dtype=np.float64)
    return np.array(data), time_seconds


def run_mathcad(prn_path, **kwargs):
    """
    Полное воспроизведение Mathcad алгоритма.

    Возвращает словарь со всеми промежуточными и финальными данными.
    """
    global n1, n2, vq, k1, dtau, im1, isat1, im, isat
    global Salt, Acid, Face, d, ww, L0, span, span1, l, tn_manual
    global smooth_window, z_split_index, vt1, vt2

    # Переопределение параметров из kwargs
    for key, val in kwargs.items():
        if key in globals():
            globals()[key] = val

    # --- Чтение данных ---
    raw, time_seconds_all = load_prn(prn_path)
    m_total = len(raw)
    print(f"Загружено {m_total} строк из {prn_path}")

    # --- Sub-array: a = submatrix(raw, n1, n2, 0, 5) ---
    # Mathcad: a := submatrix(a, n1, n2, 0, 5)
    a = raw[n1:n2 + 1].copy()
    m = len(a)
    print(f"Sub-array: a[{n1}..{n2}], m = {m}")

    # Столбцы PRN:
    # 0: индекс, 1: LED1, 2: LED2, 3: flag, 4: T(raw), 5: T(°C), 6: time

    # ========================================================================
    #  2) СИГНАЛ, СГЛАЖИВАНИЕ, БАЗОВЫЕ ЛИНИИ
    # ========================================================================

    # Mathcad: nn := -1 if vq > 0, 1 otherwise
    nn = -1 if vq > 0 else 1

    # Mathcad: signal = a[,1] / a[,2] (LED1/LED2 — ratio)
    # Но в скриншоте 1: S1 := medsmooth(a⟨(9)⟩, 5)
    # a⟨(9)⟩ = столбец 9 после расширения (ratio), или столбец 1 (LED1)
    # Для ТОЧНОГО воспроизведения: используем LED1/LED2 как signal
    # Mathcad: a[i,9] := (a[i,1] - vt1) / (a[i,2] - vt2)
    led1 = a[:, 1].astype(np.float64) - vt1
    led2 = a[:, 2].astype(np.float64) - vt2
    led2_safe = np.where(np.abs(led2) < 1e-10, 1e-10, led2)
    signal_raw = led1 / led2_safe

    # Mathcad: S1 := medsmooth(a⟨9⟩, smooth_window)
    # smooth_window=5 для 020326, smooth_window=1 для 030226
    if smooth_window > 1:
        S1 = medfilt(signal_raw, kernel_size=smooth_window)
    else:
        S1 = signal_raw.copy()

    # Mathcad: y0 := (1/im1) * Σ S1[i] для i=0..im1-1  (среднее в зоне роста)
    y0 = float(np.mean(S1[:im1]))

    # Mathcad: y0s := (1/(m-isat1)) * Σ S1[i] для i=isat1..m-1  (среднее в зоне растворения)
    y0s = float(np.mean(S1[isat1:m]))

    print(f"y0 (growth baseline) = {y0:.6f}")
    print(f"y0s (dissolution baseline) = {y0s:.6f}")

    # ========================================================================
    #  3) ТЕМПЕРАТУРА: te, tn
    # ========================================================================

    # Mathcad: te := a[isat, 4] - 273.15
    te = float(a[isat, 4] - 273.15)

    # Mathcad: tn := a[isat, 4] - 273.15 - 0.01*(te - 25)
    if tn_manual is not None:
        tn = tn_manual
    else:
        tn = float(a[isat, 4] - 273.15 - 0.01 * (te - 25))

    print(f"te = {te:.4f}")
    print(f"tn = {tn:.4f}")

    # Интерполяционные функции для температуры и сигнала
    # Mathcad: I(x) := linterp(tau, I, x) — интерп. сигнала
    # Mathcad: t(x) := linterp(tau, t, x) — интерп. температуры
    # Mathcad: Tau(x) := linterp(tau, a1, x) — интерп. времени
    tau_v = np.arange(m, dtype=np.float64)  # индексы = "время"
    I_func = interp1d(tau_v, S1, kind='linear', fill_value='extrapolate', bounds_error=False)
    t_func = interp1d(tau_v, a[:, 4] - 273.15, kind='linear', fill_value='extrapolate', bounds_error=False)

    # ========================================================================
    #  4) ОПТИЧЕСКИЕ И СОЛЮБИЛИТИ КОЭФФИЦИЕНТЫ
    # ========================================================================

    # Mathcad: gran1 := if(Face > 0, 0.47854, 1)
    #          gran2 := if(Face > 0, 0.9969, 1)
    gran1 = 0.47854 if Face > 0 else 1.0
    gran2 = 0.9969 if Face > 0 else 1.0

    # Оптические (показатель преломления): co0, co1
    if Salt == 1:  # KDP
        co0 = 0.06446 * gran1
        co1 = 1.92e-5 * gran2
    else:  # DKDP
        co0 = 0.061413 * gran1
        co1 = 1.02393e-5 * gran2

    # Солюбилити: co2, co3, co4
    if Salt == 1 and Acid == 0:
        co2, co3, co4 = 0.123, 0.002719, 1.1087e-5
    elif Salt == 1 and Acid == 1:
        co2, co3, co4 = 0.15, 0.0032487, 0.0
    elif Salt == 2 and Acid == 0:
        co2, co3, co4 = 0.166, 0.0037, 0.0
    elif Salt == 2 and Acid == 1:
        co2, co3, co4 = 0.2061, 0.003171, 0.0
    else:
        co2, co3, co4 = 0.123, 0.002719, 1.1087e-5

    print(f"co0={co0}, co1={co1}")
    print(f"co2={co2}, co3={co3}, co4={co4}")

    # ========================================================================
    #  5) ДЕТЕКЦИЯ ЭКСТРЕМУМОВ В ЗОНЕ РОСТА (0..im1)
    # ========================================================================
    # Mathcad: функция f (скриншот 1)
    # for i ∈ 0, 2..im1:
    #   k=sign(S1[0]-y0), отслеживать max |S1-y0| между нулевыми пересечениями

    # ---- Mathcad two-phase extrema detection ----

    def find_zero_crossings_f(S1, baseline, start, end):
        """
        Mathcad function f: zero-crossing detection with step=2.
        Returns array of positions where signal crosses baseline.
        """
        if start >= end or start >= len(S1):
            return np.array([], dtype=int)

        start = max(0, start)
        end = min(end, len(S1) - 1)

        k = 1 if (S1[start] - baseline) > 0 else -1
        crossings = []

        i = start
        while i <= end:
            if (S1[i] - baseline) * k < 0:
                crossings.append(i)
                k = -k
            i += 2

        return np.array(crossings, dtype=int)

    def find_extrema_e(S1, baseline, f_crossings, start, end_im1):
        """
        Mathcad function e: find extrema within zero-crossing intervals.
        For each interval between crossings, walks EVERY sample (step=1),
        finds max |S1-baseline|, averages positions if multiple have same max value.
        Filters by n1 > 4 (position sum > 4).
        """
        positions = []
        values = []

        # Build intervals: (start, f[0]), (f[0], f[1]), ..., (f[-1], end_im1)
        intervals = []
        if len(f_crossings) == 0:
            intervals = [(start, end_im1)]
        else:
            intervals.append((start, int(f_crossings[0])))
            for p in range(len(f_crossings) - 1):
                intervals.append((int(f_crossings[p]), int(f_crossings[p + 1])))
            intervals.append((int(f_crossings[-1]), end_im1))

        for lo, hi in intervals:
            if hi <= lo:
                continue

            # Find max |S1[i] - baseline| in interval [lo, hi] — step=1
            max_dev = -1.0
            best_val = 0.0
            for i in range(lo, min(hi + 1, len(S1))):
                dev = abs(S1[i] - baseline)
                if dev > max_dev:
                    max_dev = dev
                    best_val = S1[i]

            if max_dev < 0:
                continue

            # Average positions where S1[i] == best_val (exact match)
            n1_sum = 0
            k_count = 0
            for i in range(lo, min(hi + 1, len(S1))):
                if S1[i] == best_val:
                    n1_sum += i
                    k_count += 1

            if k_count == 0:
                continue

            avg_pos = n1_sum / k_count

            # Mathcad filter: e[j,0] ← n1/k if n1 > 4
            if n1_sum > 4:
                positions.append(avg_pos)
                values.append(best_val)

        if len(positions) == 0:
            return np.empty((0, 2))

        return np.column_stack([positions, values])

    # Phase 1: zero-crossings
    f_crossings = find_zero_crossings_f(S1, y0, 0, im1)
    # Phase 2: extrema within intervals
    extrs_raw = find_extrema_e(S1, y0, f_crossings, 0, im1)
    h_ext = len(extrs_raw)
    print(f"Экстремумов (raw): {h_ext}")

    # ========================================================================
    #  6) ФИЛЬТРАЦИЯ КРАЁВ
    # ========================================================================
    # Mathcad: kk := rows(extrs) - 1
    #   - Убрать первый если (extrs[0,0] - 0) <= 3
    #   - Убрать последний если (im1 - extrs[kk,0]) <= 4

    extrs = extrs_raw.copy()

    if len(extrs) >= 2:
        # Первый слишком близко к началу?
        if extrs[0, 0] - 0 <= 3:
            extrs = extrs[1:]

    if len(extrs) >= 2:
        # Последний слишком близко к im1?
        if im1 - extrs[-1, 0] <= 4:
            extrs = extrs[:-1]

    if len(extrs) >= 2:
        # Последний с отрицательным значением?
        if extrs[-1, 1] <= 0:
            extrs = extrs[:-1]

    ks = len(extrs) - 1  # число интервалов
    print(f"Экстремумов (после фильтрации): {len(extrs)}, ks = {ks}")

    if ks < 2:
        print("ОШИБКА: слишком мало экстремумов!")
        return None

    # ========================================================================
    #  7) КВАДРАТИЧНОЕ УТОЧНЕНИЕ (es)
    # ========================================================================
    # Mathcad: функция es (скриншот 12-13)
    # Для каждого j: окрестность n=floor(|dist|*k1), polyfit deg=2, вершина

    p = len(extrs) - 1  # = ks

    es = np.zeros((len(extrs), 2))
    for j in range(len(extrs)):
        pos_j = int(extrs[j, 0])

        # Размер окрестности
        if j < p:
            n_neigh = int(np.floor(abs(extrs[j + 1, 0] - extrs[j, 0]) * k1))
        else:
            n_neigh = int(np.floor(abs(extrs[j - 1, 0] - extrs[j, 0]) * k1))
        n_neigh = max(n_neigh, 2)

        # Границы
        if j < 1:
            u = pos_j - n_neigh // 2
        else:
            u = pos_j - n_neigh
        v = pos_j + n_neigh

        u = max(0, u)
        v = min(m - 1, v)

        if v - u < 2:
            es[j, 0] = float(pos_j)
            es[j, 1] = S1[pos_j]
            continue

        # Локальные данные
        x_local = np.arange(v - u + 1, dtype=float)
        y_local = S1[u:v + 1].astype(float)

        # Квадратичный фит: y = a2*x^2 + a1*x + a0
        try:
            coeffs = np.polyfit(x_local, y_local, 2)
            a2, a1, a0 = coeffs

            if abs(a2) < 1e-15:
                es[j, 0] = float(pos_j)
                es[j, 1] = S1[pos_j]
                continue

            x_vertex = -a1 / (2 * a2)
            y_vertex = a0 + a1 * x_vertex + a2 * x_vertex ** 2

            es[j, 0] = x_vertex + u  # абсолютная позиция
            es[j, 1] = y_vertex
        except Exception:
            es[j, 0] = float(pos_j)
            es[j, 1] = S1[pos_j]

    print(f"Уточнённые экстремумы es: {len(es)} шт")
    print(f"  es[0] = ({es[0,0]:.2f}, {es[0,1]:.4f})")
    print(f"  es[-1] = ({es[-1,0]:.2f}, {es[-1,1]:.4f})")

    # Mathcad: фаза y использует `e` — экстремумы из find_extrema_e.
    # e[j,0] = позиция (может быть дробной если n1/k усреднение), e[j,1] = S1 value.
    # extrs (после фильтрации краёв) = e_for_phase.
    e_for_phase = extrs.copy()
    print(f"e (для фазы): {len(e_for_phase)} шт, pos[0]={e_for_phase[0,0]:.1f}, val={e_for_phase[0,1]:.4f}")
    print(f"  e[-1]: pos={e_for_phase[-1,0]:.1f}, val={e_for_phase[-1,1]:.4f}")

    # ========================================================================
    #  8) es1 — РАСПРЕДЕЛЕНИЕ ЭКСТРЕМУМОВ НА СИГНАЛЬНУЮ СЕТКУ
    # ========================================================================
    # Mathcad: es1 (скриншот 27) — для визуализации, не для расчёта
    ss = len(es)
    es1 = np.zeros((m, 2))
    j_es1 = 0
    for i in range(m):
        if j_es1 < ss and (i - np.floor(es[j_es1, 0])) ** 2 < 0.0001:
            es1[i, 1] = es[j_es1, 1]
            es1[i, 0] = i
            j_es1 += 1
            if j_es1 >= ss:
                j_es1 = ss - 1
        else:
            es1[i, 1] = 0
            es1[i, 0] = i

    # ========================================================================
    #  9) ФАЗА НА d-СЕТКЕ (y = ys в Mathcad)
    # ========================================================================
    # Mathcad: функция y (скриншот s_1.jpg, "4)-разбиение синусоиды")
    # g = floor(isat1 / d)
    # Для каждой позиции i*d: arcsin интерполяция между экстремумами

    g = int(np.floor(isat1 / d))
    print(f"d-step grid: g = {g} points")

    p_ext = len(e_for_phase) - 1  # число интервалов между экстремумами

    # Массив y: (g+1) строк × 4 столбца [позиция, сигнал, фаза, температура]
    y = np.zeros((g + 1, 4))

    s_last = 0  # позиция последнего заполненного до break

    for i in range(g + 1):
        y[i, 0] = i * d                         # позиция
        y[i, 1] = float(I_func(y[i, 0]))        # интерполированный сигнал
        y[i, 3] = float(t_func(y[i, 0]))        # интерполированная температура

        # Найти между какими экстремумами лежит позиция
        # Используем e_for_phase (integer positions), НЕ es (polyfit refined)
        e = e_for_phase  # alias
        n_idx = p_ext + 1  # default: после всех
        for j_idx in range(len(e)):
            if e[j_idx, 0] > y[i, 0]:
                n_idx = j_idx
                break

        if n_idx < 1:
            # Случай 1: ДО первого экстремума
            denom = e[1, 1] - e[0, 1]
            if abs(denom) > 1e-15:
                arg = np.clip((y[i, 1] - e[0, 1]) / denom, -1.0, 1.0)
                y[i, 2] = -np.arcsin(arg)
            else:
                y[i, 2] = 0.0
        elif n_idx <= p_ext:
            # Случай 2: МЕЖДУ экстремумами n_idx-1 и n_idx
            denom = e[n_idx, 1] - e[n_idx - 1, 1]
            if abs(denom) > 1e-15:
                arg = np.clip((y[i, 1] - e[n_idx - 1, 1]) / denom, -1.0, 1.0)
                y[i, 2] = (np.pi / 2.0) * (n_idx - 1) + np.arcsin(arg)
            else:
                y[i, 2] = (np.pi / 2.0) * (n_idx - 1)
        else:
            # Случай 3: ПОСЛЕ последнего экстремума
            s_last = i
            denom = e[p_ext - 1, 1] - e[p_ext, 1]
            if abs(denom) > 1e-15:
                arg = np.clip((y[i, 1] - e[p_ext, 1]) / denom, -1.0, 1.0)
                y[i, 2] = (np.pi / 2.0) * p_ext + np.arcsin(arg)
            else:
                y[i, 2] = (np.pi / 2.0) * p_ext

            # Mathcad: break if a[i,0] + d > e[p,0]
            # (продолжаем до конца, но Mathcad прерывает после последнего экстремума
            #  и заполняет оставшиеся в отдельном цикле — результат тот же)

    # Mathcad НЕ clamp'ит фазу после последнего экстремума.
    # Фаза продолжает осциллировать через arcsin, как и в зоне роста.
    # Это даёт шумные скорости в dead zone, но slope(phase, T) в зоне im..isat
    # усредняет их до малого Q → малый L1 → малая коррекция.
    # Clamping НЕ применяется.

    print(f"Фаза построена: y[0,2]={y[0,2]:.4f}, y[{g},2]={y[g,2]:.4f}")

    # ========================================================================
    #  10) L1 — SLOPE ФАЗЫ ОТ ТЕМПЕРАТУРЫ
    # ========================================================================
    # Mathcad: u = первый индекс где y[i,0] > im
    #          v = первый индекс где y[i,0] > isat
    #          g_range = u..v
    #          l2[g-u] = y[g, 3] (температура)
    #          P[g-u] = y[g, 2] (фаза)
    #          Q_slope = slope(l2, P)
    #          L1 = -Q_slope / (π · co1)

    u_idx = 0
    for i in range(len(y)):
        u_idx = i
        if y[i, 0] > im:
            break

    # Не нужен v_idx для isat — используем всю зону от первого экстремума до im
    # Mathcad: u = first where y > im, v = first where y > isat
    v_idx = len(y) - 1
    for i in range(len(y)):
        if y[i, 0] > isat:
            v_idx = i - 1
            break

    # Mathcad (скриншот s_1.jpg):
    # u := first i where y[i,0] > im
    # v := first i where y[i,0] > isat
    # l2[g-u] := y[g,3], P[g-u] := y[g,2]  для g = u..v
    # Q := slope(l2, P)
    # Это slope фазы от температуры в зоне IM..ISAT (dead zone → начало растворения)
    # В dead zone фаза ≈ const → Q ≈ 0 → L1 ≈ 0 (малая коррекция)

    u_didx = 0
    for i in range(g + 1):
        if y[i, 0] > im:
            u_didx = i
            break

    v_didx = g
    for i in range(g + 1):
        if y[i, 0] > isat:
            v_didx = i - 1
            break

    if v_didx > u_didx + 2:
        l2_temps = y[u_didx:v_didx + 1, 3]
        P_phases = y[u_didx:v_didx + 1, 2]
        if len(l2_temps) > 2 and np.std(l2_temps) > 1e-10:
            Q_slope = np.polyfit(l2_temps, P_phases, 1)[0]
        else:
            Q_slope = 0.0
    else:
        Q_slope = 0.0

    L1 = -Q_slope / (np.pi * co1) if abs(co1) > 1e-20 else 0.0

    # Mathcad (скриншот hires/1.jpg): x0 = co0 - co1·y[0,3]
    # (НЕ a[m-1,4]! Используется T при начальной позиции)
    x0 = co0 - co1 * y[0, 3]

    # L0: Mathcad: L0 := L1·(co0-co1·y[u,3])/x0 - (y[u,2]-y[0,2])/(π·x0)
    x_at_u = co0 - co1 * y[u_didx, 3]
    L0_calc = L1 * x_at_u / x0 - (y[u_didx, 2] - y[0, 2]) / (np.pi * x0)

    print(f"Q_slope = {Q_slope:.6f}")
    print(f"L1 = {L1:.2f}")
    print(f"x0 = {x0:.8f} (при T[0] = {y[0,3]:.2f}°C)")
    print(f"L0 = {L0_calc:.2f}")

    # ========================================================================
    #  11) L — КУМУЛЯТИВНАЯ ТОЛЩИНА ИЗ ФАЗЫ
    # ========================================================================
    # Mathcad (скриншот hires/1.jpg):
    # L[i] = (y[i,2]/(π·x) + L0·x0/x - L0)·nn
    # где x = co0 - co1·y[i,3]

    L_arr = np.zeros(g + 1)
    for i in range(g + 1):
        x_i = co0 - co1 * y[i, 3]
        if abs(x_i) < 1e-15:
            x_i = 1e-15
        L_arr[i] = (y[i, 2] / (np.pi * x_i) + L0_calc * x0 / x_i - L0_calc) * nn

    print(f"L range: {L_arr.min():.2f} to {L_arr.max():.2f}")

    # ========================================================================
    #  11b) LOESS СГЛАЖИВАНИЕ L(t) → Fy → T[i]
    # ========================================================================
    # Mathcad: ta[w] := Tau(y[w,0]) — время для каждой d-step позиции
    # Sy := loess(ta, L, span1)
    # Fy(x) := interp(Sy, ta, L, x) — сглаженная L
    # T[i] := Fy(ta[i]) — сглаженная L при каждом времени

    # Mathcad: Tau(x) := linterp(tau, a1, x) — реальное время из PRN
    # tau = индексы 0..m-1, a1 = time_seconds для sub-array
    time_sub = time_seconds_all[n1:n2 + 1]  # время для sub-array
    if len(time_sub) == m:
        Tau_func = interp1d(np.arange(m, dtype=np.float64), time_sub,
                            kind='linear', fill_value='extrapolate', bounds_error=False)
        ta_arr = np.array([float(Tau_func(y[i, 0])) for i in range(g + 1)])
    else:
        # Fallback: позиция ≈ время в секундах
        ta_arr = y[:g + 1, 0]

    # Mathcad 2000 loess() uses LOCAL QUADRATIC (degree=2), NOT linear.
    # This is critical: degree=2 gives s2=0.34 (matches Mathcad 0.32),
    # while degree=1 gives s2=0.21 (wrong).
    # span1 = 0.15 for L smoothing (from Mathcad page 1, before redefinition to 0.2).
    span_L = 0.15

    try:
        from skmisc.loess import loess as skmisc_loess
        lo = skmisc_loess(ta_arr, L_arr, span=span_L, degree=2)
        lo.fit()
        T_smooth = lo.outputs.fitted_values
    except ImportError:
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess as lowess_func
            loess_result = lowess_func(L_arr, ta_arr, frac=span_L, return_sorted=False)
            T_smooth = loess_result
        except ImportError:
            from scipy.signal import savgol_filter
            win = max(51, int(len(L_arr) * span_L))
            if win % 2 == 0:
                win += 1
            T_smooth = savgol_filter(L_arr, win, polyorder=2)

    print(f"T_smooth (LOESS L) range: {T_smooth.min():.2f} to {T_smooth.max():.2f}")

    # ========================================================================
    #  11c) z — СКОРОСТЬ = dL_smooth/dt × 30 × nn
    # ========================================================================
    # Mathcad (скриншот hires/1.jpg):
    # z[i,1] = (T[i+1] - T[i]) / (Tau(y[i+1,0]) - Tau(y[i,0])) · 30·nn
    # T[i] = LOESS-smoothed L at time ta[i]
    # 30·nn = конверсионный множитель

    n_rates = g  # g+1 позиций → g интервалов
    z = np.zeros((n_rates, 4))

    for i in range(n_rates):
        # Средняя позиция
        z[i, 0] = (y[i, 0] + y[i + 1, 0]) / 2.0

        # Средняя температура (для σ%)
        y1_idx = int(np.ceil(y[i, 0]))
        y2_idx = int(np.floor(y[i + 1, 0]))
        y1_idx = max(0, min(y1_idx, m - 1))
        y2_idx = max(0, min(y2_idx, m - 1))
        if y2_idx >= y1_idx:
            z[i, 2] = float(np.mean(a[y1_idx:y2_idx + 1, 4] - 273.15))
        else:
            z[i, 2] = 0.5 * (float(a[y1_idx, 4] - 273.15) + float(a[y2_idx, 4] - 273.15))

        # Скорость: dL_smooth/dt × 30 × nn
        dt_tau = ta_arr[i + 1] - ta_arr[i]  # разность времён (мин)
        if abs(dt_tau) < 1e-15:
            z[i, 1] = 0.0
            continue
        z[i, 1] = (T_smooth[i + 1] - T_smooth[i]) / dt_tau * 30.0 * nn

    # Пересыщение σ%: Sigm_i = 100 * ln(Cn / Cm_i)
    # Cn = C(tn), Cm_i = C(T_i)
    Cn = co2 + co3 * tn + co4 * tn ** 2
    for i in range(n_rates):
        T_i = z[i, 2]
        Cm_i = co2 + co3 * T_i + co4 * T_i ** 2
        if Cm_i > 0 and Cn > 0:
            z[i, 3] = 100.0 * np.log(Cn / Cm_i)
        else:
            z[i, 3] = 0.0

    # J1/J2 разбиение z (Mathcad: J1=submatrix(z,0,split,0,2), J2=submatrix(z,split+1,end,0,2))
    # Файлоспецифичный индекс — удаляет переходную строку
    if z_split_index is not None and 0 <= z_split_index < n_rates:
        z = np.delete(z, z_split_index, axis=0)
        n_rates = len(z)

    print(f"\nz (rates): {n_rates} точек")
    print(f"  z[0] = pos={z[0,0]:.1f}, R={z[0,1]:.6f} мкм/мин, T={z[0,2]:.2f}°C, σ={z[0,3]:.4f}%")
    print(f"  z[-1] = pos={z[-1,0]:.1f}, R={z[-1,1]:.6f} мкм/мин, T={z[-1,2]:.2f}°C, σ={z[-1,3]:.4f}%")

    # ========================================================================
    #  12) LOESS СГЛАЖИВАНИЕ → F1
    # ========================================================================
    # Mathcad: u := csort(z, 3) — сортировка по σ%
    #          VX = u[,3], VY = u[,1]
    #          S3 := loess(VX, VY, span1), F1(x) := interp(S3, VX, VY, x)

    sort_idx = np.argsort(z[:, 3])
    u_sorted = z[sort_idx]
    VX = u_sorted[:, 3].copy()  # σ%
    VY = u_sorted[:, 1].copy()  # R (мкм/мин)

    # LOESS сглаживание R(σ) — Mathcad degree=2, span1=0.2 (page 3)
    try:
        from skmisc.loess import loess as skmisc_loess
        lo = skmisc_loess(VX, VY, span=span1, degree=2)
        lo.fit()
        VX_smooth = VX.copy()
        VY_smooth = lo.outputs.fitted_values
    except ImportError:
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            result = lowess(VY, VX, frac=span1, return_sorted=True)
            VX_smooth = result[:, 0]
            VY_smooth = result[:, 1]
        except ImportError:
            window = max(3, int(len(VX) * span1))
            if window % 2 == 0:
                window += 1
            kernel = np.ones(window) / window
            VY_smooth = np.convolve(VY, kernel, mode='same')
            VX_smooth = VX

    # F1(x) — интерполяция LOESS результата
    F1_func = interp1d(VX_smooth, VY_smooth, kind='linear',
                        fill_value='extrapolate', bounds_error=False)

    # ========================================================================
    #  13) VX1, VY1 — ОБРЕЗКА ДО ПЕРВОГО VY > 0.01
    # ========================================================================
    # Mathcad (скриншот 6): VX1, VY1 — обрезать начало (σ с R < 0.01)
    # VX1 используется для Sig035, G_h, s2 (Q regression)
    # Grid search (s) использует ПОЛНЫЕ VX/VY, НЕ обрезанные VX1!

    # Mathcad VX1/VY1: VX1 = копия VX ДО первого VY > 0.01 (dead zone + переход).
    # В Mathcad R в dead zone (σ < 0) → 0. У нас R > 0 из-за LOESS overshooting.
    #
    # Fix: пропускаем dissolution zone (σ < 0, далеко от dead zone),
    # начинаем проверку VY > 0.01 только от σ ≈ 0 (около tn).
    # В Mathcad R в dead zone (σ ≈ 0) → 0. У нас LOESS overshooting при σ < 0.

    # σ ≈ 0 = tn boundary. Пропускаем все точки с σ < 0.
    dead_zone_sigma = 0.0
    skip_dissolution = 0
    for k_idx in range(len(VX)):
        if VX[k_idx] >= dead_zone_sigma:
            skip_dissolution = k_idx
            break

    # Теперь ищем первый VY > 0.01 начиная от dead zone boundary
    t_start = len(VX)
    for k_idx in range(skip_dissolution, len(VX)):
        if VY[k_idx] > 0.01:
            t_start = k_idx
            break

    # VX1 = VX[0..t_start] (dissolution + dead zone + переход)
    VX1 = VX[:t_start + 1].copy()
    VY1 = VY[:t_start + 1].copy()
    n_vx1 = len(VX1) - 1
    print(f"VX1/VY1: {len(VX1)} точек (от σ={VX1[0]:.3f}% до σ={VX1[-1] if len(VX1)>0 else 0:.3f}%)")

    # ========================================================================
    #  14) Sig035 — ПЕРВЫЙ σ ГДЕ F1(σ) > 0.35
    # ========================================================================
    # Mathcad: Sig035 := for i ∈ 0..rows(u)-1: DZ←VX_i, break if F1(VX_i) > 0.35

    Sig035 = 0.0
    for i in range(len(VX)):
        if F1_func(VX[i]) > 0.35:
            Sig035 = float(VX[i])
            break
    if Sig035 == 0.0 and len(VX) > 0:
        Sig035 = float(VX[-1])

    print(f"Sig035 = {Sig035:.4f} %")

    # ========================================================================
    #  15) POWER LAW GRID SEARCH (s0, s1)
    # ========================================================================
    # Mathcad: функция s (скриншот s.jpg)
    # ВАЖНО: grid search использует ПОЛНЫЕ VX/VY (не обрезанные VX1!)
    # s := for i ∈ 0..n-1, for x ∈ VX_i, VX_i+0.01..VX_{i+1}
    #   q1 = Σ(VX_k - x)^w · VY_k  (k=i..n)
    #   q2 = Σ(VX_k - x)^{2w}       (k=i..n)
    #   s0 = q1/q2
    #   f = Σ VY_below² + Σ (s0·(VX_above-x)^w - VY_above)²

    w_exp = 1.0
    best_s0, best_s1, best_min = 0.0, 0.0, 100000.0

    # Mathcad (скриншот s.jpg):
    # n := rows(VX1) - 1    (VX1 = dead zone + transition, ~261 элементов)
    # for i ∈ 0..n-1:
    #   for x ∈ VX_i, VX_i + 0.01..VX_{i+1}:
    #     q1 ← Σ_{k=i}^{n} (VX1_k - x)^w · VY1_k
    #     q2 ← Σ_{k=i}^{n} (VX1_k - x)^{2w}
    #     q ← q1/q2
    #     f ← Σ_{k=0}^{i-1} VY1_k² + Σ_{k=i}^{n} (q·(VX1_k-x)^w - VY1_k)²
    #
    # КЛЮЧЕВОЕ: n = rows(VX1)-1 ≈ 261, т.е. grid search работает только по
    # первым ~262 точкам VX/VY (dead zone + начало роста), а НЕ по всему массиву.
    # Суммы q1/q2 также идут от k=i до n (=261), а не до конца VX.
    # Формулы используют VX1/VY1, но индексы привязаны к VX (сортированному).

    # Mathcad (скриншот s.jpg):
    # VX1 = первые t_start+1 элементов VX (dead zone: VY <= 0.01)
    # n := rows(VX1) - 1 ≈ t_start
    # Grid search по VX1/VY1, суммы от k=i до n
    # Результат: s0 (≈ 0, шум dead zone), s1 = dead zone boundary (σ%)
    #
    # ВАЖНО: s0 из grid search используется только для таблицы H (dead zone).
    # Основная кривая R(σ) — это LOESS F1, а НЕ power law.
    # s1 = dead zone boundary в единицах σ%

    # Mathcad: grid search по VX1/VY1 (sorted, first ~262 elements up to VY>0.01)
    # В нашей реализации arcsin-осцилляции в dead zone крупнее чем в Mathcad
    # (из-за мелких отличий в квадратичном уточнении), что смещает VX1 boundary.
    #
    # Прагматичный подход: grid search по ВСЕМ VX/VY (как в Mathcad _grid_search),
    # суммы от k=i до n (все сортированные данные).
    # Это эквивалентно Mathcad, если данные z совпадают.

    # Mathcad (скриншот hires/7.jpg):
    # n := rows(VX1) - 1 = 273
    # for i ∈ 0..n-1:
    #   for x ∈ VX_i, VX_i + 0.01..VX_{i+1}:    ← итерация по VX (полный!)
    #     q1 ← Σ(k=i..n) (VX1_k - x)^w · VY1_k  ← суммы по VX1/VY1
    #     q2 ← Σ(k=i..n) (VX1_k - x)^{2w}
    #     ...
    #
    # КЛЮЧЕВОЕ: x итерируется по VX[i]..VX[i+1] (полный массив),
    # но суммы берутся по VX1[i..n] и VY1[i..n] (обрезанный массив).
    # Это даёт x < VX1[0] при ранних i (отрицательные σ%),
    # а VX1 содержит только "хвост" до first VY>0.01.

    n_grid = len(VX1) - 1  # = rows(VX1) - 1
    print(f"Grid search: n = {n_grid} (VX1 range: {VX1[0]:.4f} to {VX1[-1]:.4f}%)")

    for i in range(min(n_grid, len(VX) - 1)):
        # x итерируется по VX[i]..VX[i+1] (полный массив!)
        x_val = VX[i]
        x_end = VX[min(i + 1, len(VX) - 1)]
        while x_val < x_end:
            # Суммы по VX1[i..n] и VY1[i..n]
            # Если i > n_grid → нет данных
            i_clamped = min(i, n_grid)
            VX1_above = VX1[i_clamped:]
            VY1_above = VY1[i_clamped:]

            xw = np.power(VX1_above - x_val, w_exp)
            xw2 = np.power(VX1_above - x_val, 2 * w_exp)
            denom = np.sum(xw2)
            if denom < 1e-20:
                x_val += 0.01
                continue

            q1_val = np.sum(xw * VY1_above)
            s0_cand = q1_val / denom

            # f = Σ(k=0..i-1) VY1_k² + Σ(k=i..n) (q·(VX1_k-x)^w - VY1_k)²
            f_res = np.sum(VY1[:i_clamped] ** 2)
            f_res += np.sum((s0_cand * xw - VY1_above) ** 2)

            if f_res < best_min:
                best_s0 = s0_cand
                best_s1 = x_val
                best_min = f_res

            x_val += 0.01

    s_result = np.array([best_s0, best_s1])
    print(f"\nPower law fit: s0 = {best_s0:.6f}, s1 = {best_s1:.4f}")

    # G(σ) = s0 * (σ*0.01 - s1)^w → но Mathcad: G_h = s0*(h*0.01 - s1)^w
    # h — индекс в массиве с шагом по σ%, σ = h*0.01?
    # Нет: G_h = s0*(h*0.01 - s1)^w, где h = floor(VX_hh * 100)
    # Это для таблицы H. Основная формула: G(σ%) = s0*(σ% - s1)^w если σ%>s1, иначе 0

    # ========================================================================
    #  16) s2 — ФОРМ-ФАКТОР
    # ========================================================================
    # Mathcad: Q (скриншот 7)
    # Для z в исходном порядке: break если z[i,1] < 0.0
    # Если 0 < z[i,1] < 0.3: записать (σ%, √R)
    # Q1 = intercept(σ, √R), Q2 = slope(σ, √R)
    # s2 = -Q1/Q2

    sigma_sel = []
    sqrt_r_sel = []

    # Mathcad (скриншот hires/18.jpg):
    # Q := for i ∈ 0..rows(z)-1:
    #   break if z[i,1] < 0.0
    #   q[j,0] ← z[i,3] if z[i,1] < 0.3
    #   q[j,1] ← √(z[i,1]) if z[i,1] < 0.3
    #   j ← j+1 if z[i,1] < 0.3
    #
    # z iterate in POSITION order (not sorted!). z[i,1] = dL_smooth/dt × 30·nn
    # is SMOOTH (LOESS of L before differentiation), so break at first R < 0 works.
    for i in range(n_rates):
        if z[i, 1] < 0.0:
            break
        if 0.0 < z[i, 1] < 0.3 and z[i, 3] > 0:
            sigma_sel.append(z[i, 3])
            sqrt_r_sel.append(np.sqrt(z[i, 1]))

    if len(sigma_sel) >= 3:
        sigma_arr = np.array(sigma_sel)
        sqrt_r_arr = np.array(sqrt_r_sel)
        coeffs_s2 = np.polyfit(sigma_arr, sqrt_r_arr, 1)
        Q2_s2 = coeffs_s2[0]  # slope
        Q1_s2 = coeffs_s2[1]  # intercept
        if abs(Q2_s2) > 1e-15:
            s2 = -Q1_s2 / Q2_s2
        else:
            s2 = 0.0
    else:
        s2 = 0.0

    print(f"s2 = {s2:.4f}")

    # ========================================================================
    #  17) МЁРТВАЯ ЗОНА: Td, Sigm
    # ========================================================================
    # Mathcad: Td := te - tm, tm := a[im, 4] - 273.15
    #          Cn := co2 + co3*te + co4*te^2
    #          Cm := co2 + co3*tm + co4*tm^2
    #          Sigm := 100 * ln(Cn / Cm)

    tm = float(a[im, 4] - 273.15)
    Td = te - tm
    Cn_val = co2 + co3 * te + co4 * te ** 2
    Cm_val = co2 + co3 * tm + co4 * tm ** 2
    if Cm_val > 0 and Cn_val > 0:
        Sigm = 100.0 * np.log(Cn_val / Cm_val)
    else:
        Sigm = 0.0

    print(f"\nTd = {Td:.4f} °C")
    print(f"Sigm = {Sigm:.4f} %")

    # ========================================================================
    #  18) ТАБЛИЦА H (dead zone table)
    # ========================================================================
    # Mathcad: H (скриншот 7)
    # p := max(z[,3]) — макс σ%
    # H: i=0, k=2.6, для x от p до 0 шагом -0.01:
    #   h[i,1] = x если F1(x) < l*k
    #   h[i,0] = k
    #   i += 1 если F1(x) < l*k
    #   k *= 3/4 если F1(x) < l*k
    #   break если k < 0.01

    p_max = float(np.max(z[:, 3]))
    H_list = []
    i_h = 0
    k_h = 2.6
    x_h = p_max
    while x_h >= 0 and k_h >= 0.01:
        f1_val = float(F1_func(x_h))
        if f1_val < l * k_h:
            H_list.append([k_h, x_h])
            k_h *= 3.0 / 4.0
        x_h -= 0.01

    H = np.array(H_list) if H_list else np.zeros((0, 2))
    print(f"Таблица H: {len(H)} строк")
    if len(H) > 0:
        print(f"  H[0] = dT={H[0,0]:.2f}, σ={H[0,1]:.4f}")
        print(f"  H[-1] = dT={H[-1,0]:.2f}, σ={H[-1,1]:.4f}")

    # Дополнительная строка: ur (te, tn)
    ur_rows = len(H)

    # ========================================================================
    #  19) ЭТАЛОННЫЕ КРИВЫЕ Si, Si1 (a/a1/b/b1)
    # ========================================================================
    # Mathcad: скриншоты 19-20
    # a = KDP, Cac=9.8%, CFe=4.5ppm
    # a1 = KDP, Cac=9.8%, CFe=20.5ppm
    # b = KDP, Cac=0, CFe=0
    # b1 = KDP, Cac=0, CFe=16ppm
    # Выбор: a := if(Acid < 1, b, a), a1 := if(Acid < 1, b1, a1)

    coeff_a = np.array([
        [2.879, -0.097, 8.996e-4, 0],
        [3.411, -0.102, 8.523e-4, 0.025],
        [4.201, -0.123, 1.035e-3, 0.05],
        [3.477, -0.075, 4.397e-4, 0.1],
        [3.19,  -0.039, -2.029e-5, 0.2],
        [5.351, -0.102, 5.547e-4, 0.4],
        [10.991, -0.284, 2.219e-3, 0.8],
        [6.374, 0.032, -1.799e-3, 1.6],
    ])

    coeff_a1 = np.array([
        [5.69,  -0.196, 1.779e-3, 0],
        [11.001, -0.371, 3.27e-3, 0.025],
        [8.767, -0.244, 1.736e-3, 0.05],
        [9.146, -0.236, 1.569e-3, 0.1],
        [5.8,   -0.063, -3.515e-4, 0.2],
        [4.342, 0.03, -1.376e-3, 0.4],
        [7.774, -0.087, -1.15e-4, 0.8],
        [9.816, -0.091, -4.187e-4, 1.6],
    ])

    coeff_b = np.array([
        [6.603, -0.174, 9.679e-4, 0],
        [8.593, -0.222, 1.266e-3, 0.025],
        [5.748, -0.032, -1.365e-3, 0.05],
        [7.389, -0.099, -4.851e-4, 0.1],
        [11.691, -0.32, 2.635e-3, 0.2],
        [15.274, -0.472, 4.564e-3, 0.4],
        [22.789, -0.779, 8.167e-3, 0.8],
        [37.35, -1.374, 0.015, 1.6],
    ])

    coeff_b1 = np.array([
        [32.216, -1.291, 0.014, 0],
        [16.196, -0.321, 5.465e-4, 0.025],
        [11.988, -0.066, -2.761e-3, 0.05],
        [16.538, -0.302, 6.399e-4, 0.1],
        [19.768, -0.453, 2.792e-3, 0.2],
        [24.752, -0.641, 5.073e-3, 0.4],
        [18.015, -0.206, -5.911e-4, 0.8],
        [16.58, -0.033, -2.706e-3, 1.6],
    ])

    # Выбор по Acid
    if Acid < 1:
        ref_a = coeff_b
        ref_a1 = coeff_b1
    else:
        ref_a = coeff_a
        ref_a1 = coeff_a1

    # Si[j,0] = a[j,0] + a[j,1]*te + a[j,2]*te^2, Si[j,1] = a[j,3]
    Si = np.zeros((8, 2))
    Si1 = np.zeros((8, 2))
    for j in range(8):
        Si[j, 0] = ref_a[j, 0] + ref_a[j, 1] * te + ref_a[j, 2] * te ** 2
        Si[j, 1] = ref_a[j, 3]
        Si1[j, 0] = ref_a1[j, 0] + ref_a1[j, 1] * te + ref_a1[j, 2] * te ** 2
        Si1[j, 1] = ref_a1[j, 3]

    print(f"\nЭталонные кривые при te={te:.2f}:")
    for j in range(8):
        print(f"  Si[{j}]: R={Si[j,0]:.3f} мкм/мин при dT={Si[j,1]:.3f}°C")

    # ========================================================================
    #  20) D/DFe МАТРИЦЫ И ИНТЕРПОЛЯЦИЯ Z/ZFe
    # ========================================================================
    # Mathcad: D (5×24), DFe (5×24) — скриншоты 17-18
    # Z := interpolate по te в строках D
    # Пока оставляем заглушку — основные кривые строятся из Si/Si1

    D = None   # TODO: заполнить из скриншотов
    DFe = None
    Z_ref = None
    ZFe_ref = None

    # ========================================================================
    #  21) ФИНАЛЬНЫЕ ПАРАМЕТРЫ
    # ========================================================================

    # Конвертация dT: для каждой точки z
    # dT = tn - T  (но в Mathcad: ось X = tn/100 · z[,2] ??? нет, просто Td шкала)
    # На графике R(dT): X = (tn/100) * z[i,?] — нет, X просто = dT = tn - T
    # Mathcad X-ось: (tn/100)·z[i,0] — wait, это из скриншота Base_grath.jpg:
    #   X: (tn/100)·z, (tn/100)·z, Si_j,0·(tn/100), Si1_j,0·(tn/100), (tn/100)·Z, (tn/100)·ZFe
    #   Нет, скорее X = z[,2] (температура) конвертируется в dT

    # Просто dT = te - T (или tn - T)
    # Из графика: ось X = dT(C), от 0 до 4
    # z[i] содержит T в столбце 2 → dT[i] = te - z[i,2]? или tn - z[i,2]?
    # В Mathcad Base_grath.jpg подписи трасс: (tn/100)·z — но это σ% / 100... нет.
    # Фактически ось X в Mathcad = σ% пересчитанная в dT через кривую растворимости

    # Mat vector
    tel = 0  # tel = "найденная по кривой р-ния" — обычно 0 или вручную
    Par = 0
    ExGraf = -0.7

    Mat = np.array([
        Salt, Acid, te, tel, tn, best_s1, Sigm, s2, Sig035, Td,
        n1, n2, ww, d, dtau, l, im1, isat1, im, isat
    ])

    print(f"\n{'='*60}")
    print(f"РЕЗУЛЬТАТ (Mat vector):")
    print(f"{'='*60}")
    print(f"  Salt = {Salt}")
    print(f"  Acid = {Acid}")
    print(f"  te   = {te:.4f} °C")
    print(f"  tel  = {tel}")
    print(f"  tn   = {tn:.4f} °C")
    print(f"  s1   = {best_s1:.4f} %")
    print(f"  Sigm = {Sigm:.4f} %")
    print(f"  s2   = {s2:.4f}")
    print(f"  Sig035 = {Sig035:.4f} %")
    print(f"  Td   = {Td:.4f} °C")

    # ========================================================================
    #  22) ГРАФИК R(dT) — КАК В MATHCAD
    # ========================================================================
    import matplotlib.pyplot as plt

    # dT для z-данных
    z_dT = tn - z[:, 2]  # dT = tn - T

    # dT для эталонных кривых Si/Si1
    Si_dT = Si[:, 1]
    Si1_dT = Si1[:, 1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Trace 1: z (measured data) — чёрные точки (×1.441 для mm/day)
    ax.plot(z_dT, 1.441 * z[:, 1], 'k.', markersize=1, alpha=0.3, label='z (data)')

    # Trace 2: F1 (LOESS smoothed) — красная dash-dot
    # F1 на сетке dT
    dT_grid = np.linspace(0, max(z_dT), 500)
    # Конвертируем dT → σ% → F1(σ%)
    # T = tn - dT → σ% = 100*ln(Cn/C(T))
    F1_on_dT = np.zeros(len(dT_grid))
    for idx_dt, dt_val in enumerate(dT_grid):
        T_val = tn - dt_val
        C_val = co2 + co3 * T_val + co4 * T_val ** 2
        if C_val > 0 and Cn > 0:
            sigma_val = 100.0 * np.log(Cn / C_val)
        else:
            sigma_val = 0.0
        F1_on_dT[idx_dt] = float(F1_func(sigma_val))

    ax.plot(dT_grid, 1.441 * F1_on_dT, 'r-', linewidth=1.5, label='F1 (LOESS)')

    # Trace 3: Si (reference, clean/low Fe) — синяя сплошная
    ax.plot(Si_dT, 1.441 * Si[:, 0], 'b-o', markersize=4, label=f'Si (Acid={Acid}, low Fe)')

    # Trace 4: Si1 (reference, high Fe) — зелёная сплошная
    ax.plot(Si1_dT, 1.441 * Si1[:, 0], 'g-d', markersize=4, label=f'Si1 (Acid={Acid}, high Fe)')

    ax.set_xlabel('dT (°C)')
    ax.set_ylabel('R (mm/day)')
    ax.set_title(f'{Path(prn_path).stem}   te={te:.2f}  tn={tn:.2f}  Td={Td:.2f}  '
                 f's1={best_s1:.2f}  s2={s2:.2f}  Sig035={Sig035:.2f}')
    ax.set_xlim(0, 4)
    ax.set_ylim(-0.2, 3.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Аннотация
    info_text = (f'tn = {tn:.2f}\nTd = {Td:.2f}\nSigm = {Sigm:.2f}\n'
                 f's1 = {best_s1:.2f}\ns2 = {s2:.4f}\nSig035 = {Sig035:.2f}')
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    out_name = f"mathcad_exact_{Path(prn_path).stem}.png"
    fig.savefig(out_name, dpi=150)
    print(f"\nГрафик сохранён: {out_name}")
    plt.close(fig)

    # ========================================================================
    #  23) ГРАФИК R(σ%) — КАК В MATHCAD
    # ========================================================================
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5))

    # Данные
    ax2.plot(z[:, 3], z[:, 1], 'k.', markersize=1, alpha=0.3, label='z (data)')

    # LOESS
    sigma_grid = np.linspace(max(0, VX[0]), VX[-1], 500)
    F1_sigma = np.array([float(F1_func(s_val)) for s_val in sigma_grid])
    ax2.plot(sigma_grid, F1_sigma, 'r-', linewidth=1.5, label='F1 (LOESS)')

    # Power law fit
    sigma_fit = np.linspace(max(best_s1, 0), VX[-1], 200)
    R_fit = np.where(sigma_fit > best_s1,
                      best_s0 * np.power(sigma_fit - best_s1, w_exp), 0.0)
    ax2.plot(sigma_fit, R_fit, 'b--', linewidth=1, label=f'G = {best_s0:.4f}·(σ-({best_s1:.2f}))^{w_exp}')

    ax2.set_xlabel('σ (%)')
    ax2.set_ylabel('R (мкм/мин)')
    ax2.set_title(f'R(σ%) — {Path(prn_path).stem}')
    ax2.set_xlim(0, 6)
    ax2.set_ylim(-0.05, 0.4)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    out_name2 = f"R_sigma_exact_{Path(prn_path).stem}.png"
    fig2.savefig(out_name2, dpi=150)
    print(f"График сохранён: {out_name2}")
    plt.close(fig2)

    return {
        'Mat': Mat,
        'z': z,                # rate data (n_rates × 4)
        'y': y,                # phase grid (g+1 × 4)
        'es': es,              # refined extrema
        'VX': VX, 'VY': VY,   # sorted σ%, R
        'F1': F1_func,         # LOESS function
        'Si': Si, 'Si1': Si1, # reference curves
        'H': H,                # dead zone table
        'te': te, 'tn': tn, 'Td': Td, 'Sigm': Sigm,
        's0': best_s0, 's1': best_s1, 's2': s2, 'Sig035': Sig035,
    }


# ============================================================================
#  MAIN
# ============================================================================

if __name__ == '__main__':
    # Определяем файл
    project_root = Path(__file__).parent.parent

    if len(sys.argv) > 1:
        prn_path = sys.argv[1]
    else:
        # По умолчанию: 020326_5_1 (как в скриншотах Mathcad)
        candidates = [
            project_root / "020326 KDP нейтр XXXIII п.17" / "__020326_5_1.prn",
            project_root / "020326 KDP нейтр XXXIII п.15" / "__020326_2_1.prn",
        ]
        prn_path = None
        for c in candidates:
            if c.exists():
                prn_path = str(c)
                break

        if prn_path is None:
            # Поиск любого PRN
            prn_files = list(project_root.glob("**/*.prn"))
            if prn_files:
                prn_path = str(prn_files[0])
            else:
                print("PRN файл не найден!")
                sys.exit(1)

    print(f"{'='*60}")
    print(f"MATHCAD EXACT REPRODUCTION")
    print(f"Файл: {prn_path}")
    print(f"{'='*60}\n")

    # Параметры для 020326_5_1 (из MCD)
    result = run_mathcad(
        prn_path,
        n1=0, n2=9000,
        im1=6350, isat1=8090,
        im=7227, isat=7576,
        Salt=1, Acid=0, Face=0,
        d=3.3, ww=1, k1=0.15,
        span1=0.2, l=0.01,
        tn_manual=None,  # вычислить по формуле
    )
