# Исследование: Нормализация амплитуды для извлечения фазы

**Дата:** 2026-03-14

---

## Проблема

Hilbert-преобразование даёт ненулевой dφ/dt в dead zone потому что шум фазы ∝ 1/amplitude. При малой амплитуде (нет полос роста) шум фазы огромен. Все попытки постфактум убрать этот шум (envelope weighting, noise floor subtraction, dual-channel коррекция) — костыли. Нужно решить проблему **до** вычисления фазы.

---

## Физика сигнала

Интерферометрический сигнал:
```
S(t) = A(t) · cos(φ(t)) + DC(t) + n(t)
```

Где:
- `A(t)` — амплитуда осцилляций (огибающая). Для идеальной оптики ≈ const.
  Реально: медленно меняется из-за дрейфа юстировки, деградации LED, изменения отражательной способности поверхности кристалла.
- `φ(t)` — фаза = то, что нужно измерить. Каждые 2π = λ/(2n) толщины.
- `DC(t)` — средняя линия (baseline). Дрейфует от температуры и т.п.
- `n(t)` — шум электроники, АЦП, фотонный шум. Примерно постоянный ~σ_n.

### Что известно из экстремумов:
- **Максимумы:** cos(φ) = +1 → S_max = A + DC
- **Минимумы:** cos(φ) = -1 → S_min = -A + DC
- **Нулевые пересечения:** cos(φ) = 0 → S = DC

Отсюда:
```
A = (S_max - S_min) / 2     — полуамплитуда
DC = (S_max + S_min) / 2    — средняя линия
```

### Нормализация между экстремумами:
```
cos(φ(t)) = (S(t) - DC) / A
φ(t) = arccos((S(t) - DC) / A)
```

Это **точная** формула фазы — не приближение, не статистика, а аналитическое выражение. Работает между каждой парой соседних экстремумов.

---

## Ключевой инсайт: связь амплитуды и фазы

В нормализованном сигнале cos(φ):
- |cos(φ)| = расстояние от средней линии / полуамплитуда
- В максимуме: |cos| = 1, фаза определена наиболее точно
- У нулевого пересечения: |cos| = 0, фаза меняется быстро (максимальная чувствительность)
- В dead zone: **нет экстремумов** → нет A → нет нормализации → rate = 0 естественно

Это объясняет, почему Mathcad arcsin работает в dead zone: если нет пары экстремумов, фаза просто не вычисляется.

---

## Методы нормализации

### 1. Peak-trough envelope interpolation (рекомендуется)

Найти все максимумы и минимумы сигнала, построить интерполированные огибающие:

```python
# Найти экстремумы
peaks = find_peaks(signal)
troughs = find_peaks(-signal)

# Интерполировать огибающие на весь сигнал
upper_env = interp1d(peaks, signal[peaks], kind='cubic', fill_value='extrapolate')
lower_env = interp1d(troughs, signal[troughs], kind='cubic', fill_value='extrapolate')

# Нормализация
A = (upper_env(t) - lower_env(t)) / 2
DC = (upper_env(t) + lower_env(t)) / 2
normalized = (signal - DC) / np.maximum(A, threshold)
```

**Плюсы:**
- Использует реальные особенности сигнала (пики/впадины)
- В dead zone A → 0 → нормализация не усиливает шум (порог)
- Убирает как линейный, так и циклический дрейф DC
- Убирает медленные изменения амплитуды A

**Минусы:**
- Нужно надёжно находить экстремумы (шумовые пики = ложные экстремумы)
- Экстраполяция на краях может быть нестабильной

### 2. LOESS envelope (робастная версия)

Вместо всех пиков — LOESS-сглаживание верхней и нижней огибающих:

```python
from statsmodels.nonparametric.smoothers_lowess import lowess

# Верхняя огибающая: LOESS по максимумам
peak_env = lowess(signal[peaks], peaks, frac=0.2)

# Нижняя огибающая: LOESS по минимумам
trough_env = lowess(signal[troughs], troughs, frac=0.2)
```

Устойчива к ложным экстремумам.

### 3. EMD (Empirical Mode Decomposition)

EMD раскладывает сигнал на IMF (intrinsic mode functions), каждая из которых нормализована по определению. Первая IMF ≈ интерференционный сигнал с нормализованной амплитудой.

```python
from emd import sift
imfs = sift(signal)
# imfs[0] ≈ интерференционный компонент
```

### 4. Analytic signal normalization (текущий подход — проблемный)

```python
analytic = hilbert(signal)
envelope = abs(analytic)
normalized = signal / envelope  # ← проблема: envelope шумит в dead zone
```

Проблема: Hilbert envelope в dead zone = шум, нормализация усиливает шум.

---

## Вычисление фазы после нормализации

### Вариант A: arccos (прямой)

```python
cos_phi = np.clip(normalized, -1, 1)
phase_raw = np.arccos(cos_phi)
# Проблема: arccos не различает восходящую и нисходящую ветвь
# Нужно отслеживать направление (знак производной)
```

### Вариант B: Hilbert на нормализованном сигнале

```python
analytic = hilbert(normalized)
phase = unwrap(angle(analytic))
# Теперь envelope ≈ 1 везде (где есть полосы) → шум фазы ≈ noise/1 = const
# В dead zone: normalized ≈ noise/threshold → envelope мал → Hilbert по-прежнему шумит
```

### Вариант C: arcsin с трекингом (как Mathcad, но непрерывный)

Между каждой парой экстремумов:
```python
# Mathcad формула:
# phase[i] = π/2·(n-1) + arcsin((S[i] - midline) / half_amplitude)
# Где n = номер полупериода

# Непрерывная версия:
for each half_period (extr_k to extr_{k+1}):
    A_local = abs(extr_{k+1}.value - extr_k.value) / 2
    DC_local = (extr_{k+1}.value + extr_k.value) / 2
    phi_local = arcsin((S - DC_local) / A_local)
    phase[interval] = k*π/2 + phi_local
```

**Это именно то, что делает Mathcad**, но можно сделать непрерывнее (интерполяция A и DC вместо ступенчатых значений).

### Вариант D: Hybrid — нормализация пиками + Hilbert

Лучшее из двух миров:
1. Нормализовать сигнал по пикам/впадинам → amplitude ≈ 1
2. Применить Hilbert к нормализованному → чистая фаза
3. Где нет пиков (dead zone): amplitude = 0 → rate = 0

```python
# Нормализация
A_interp = interpolate_envelope(peaks, troughs, signal)
DC_interp = interpolate_midline(peaks, troughs, signal)
A_safe = np.maximum(A_interp, noise_threshold)
normalized = (signal - DC_interp) / A_safe

# Маска: где A мала (dead zone), не доверяем фазе
signal_mask = A_interp > noise_threshold

# Hilbert на нормализованном
analytic = hilbert(normalized)
phase = unwrap(angle(analytic))
rate = abs(gradient(phase)) / (2π) * 30 / x
rate[~signal_mask] = 0  # dead zone
```

---

## Дрейфы и их коррекция

### Линейный дрейф DC(t)
- Причина: медленное изменение температуры LED, дрейф юстировки
- Коррекция: midline = (upper_env + lower_env) / 2 автоматически убирает
- Текущий detrend (linear) — грубое приближение

### Циклический дрейф DC(t)
- Причина: циклы охлаждения/нагрева термостата, вибрации
- Коррекция: midline из peak/trough interpolation следует за циклом
- Highpass фильтр может обрезать нужные частоты (если цикл ~= частота полос)

### Дрейф амплитуды A(t)
- Причина: изменение отражательной способности, рост кристалла меняет оптический путь
- Коррекция: A из peak/trough interpolation автоматически следует за дрейфом
- Hilbert envelope менее надёжен (шумит при малой амплитуде)

### Соотношение каналов 40/60
- LED1 и LED2 имеют разные A но одинаковый φ
- При peak/trough нормализации каждого канала отдельно: normalized1 ≈ normalized2 = cos(φ)
- Усреднение нормализованных каналов: лучший SNR

---

## Рекомендация

### Метод: Peak-trough normalization + Hilbert (Вариант D)

1. **Найти экстремумы** в каждом канале (LED1, LED2) — уже есть в коде (find_extrema_mathcad)
2. **Интерполировать огибающие** (cubic или LOESS) → A(t), DC(t) для каждого канала
3. **Нормализовать** каждый канал: `norm_i = (LED_i - DC_i) / max(A_i, threshold)`
4. **Усреднить** нормализованные каналы: `norm_avg = (norm1 + norm2) / 2`
5. **Hilbert** на norm_avg → фаза с постоянным шумом
6. **Маска**: rate = 0 где A < threshold (dead zone — естественно)

**Преимущество над текущим подходом:**
- Шум фазы = const (не ∝ 1/A)
- Dead zone обрабатывается автоматически (нет пиков → A=0 → rate=0)
- Дрейфы (линейные, циклические) убираются автоматически
- Не нужны костыли (noise floor, confidence, Td mask)

**Связь с Mathcad:**
Mathcad arcsin делает то же самое — нормализация по парам экстремумов + arcsin фаза. Наш метод — непрерывная версия (интерполированные огибающие вместо ступенчатых + Hilbert вместо arcsin).
