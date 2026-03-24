# Исследование: очистка сигнала и улучшение алгоритма

Дата: 2026-03-13

## 1. Почему осцилляции в Classic d-step методе

Arcsin-интерполяция фазы между экстремумами (`build_phase_dstep` в `classic.py:327`) имеет **сингулярность производной** вблизи экстремумов. Механизм:

- Между min на позиции `ext_pos[n-1]` и max на `ext_pos[n]` сигнал аппроксимируется как полусинусоида
- Фаза восстанавливается через `arcsin((S - val[n-1]) / (val[n] - val[n-1]))`
- **Проблема**: `d(arcsin)/dx → ∞` при `x → ±1` (вблизи экстремумов). Малый шум → большой скачок фазы
- **Проблема**: знаменатель `val[n] - val[n-1]` меняется от интервала к интервалу (амплитудная модуляция)
- **Проблема**: на границе каждого экстремума формула переключается, создавая разрывы

Результат: фаза имеет периодические кинки → при дифференцировании получаем осцилляции R с периодом = расстоянию между экстремумами.

**Это известная проблема** в интерферометрии. Фурье- и Гильберт-методы были созданы именно для её решения.

### Варианты исправления (по приоритету):

1. **Сплайн-сглаживание L(t) с монотонностью** — заменить LOWESS на UnivariateSpline + `np.maximum.accumulate`:
```python
from scipy.interpolate import UnivariateSpline
spline = UnivariateSpline(t_axis, L_raw, s=len(L_raw) * 0.001)
L_smooth = np.maximum.accumulate(spline(t_axis))
rate = np.diff(L_smooth) / d * 30
```

2. **Не дифференцировать arcsin-фазу напрямую** — вычислять rate только по парам экстремумов (один rate на полупериод), без d-step интерполяции

3. **Перейти на Hilbert/wavelet** для извлечения фазы (см. раздел 2)

---

## 2. Лучшие методы извлечения фазы

### 2a. CWT Ridge (вейвлет-гребень) — РЕКОМЕНДУЕТСЯ

Библиотека: `ssqueezepy` (`pip install ssqueezepy`)

Извлекает мгновенную частоту, следуя гребню максимальной энергии в вейвлет-скалограмме. Устойчив к многокомпонентным сигналам.

```python
from ssqueezepy import cwt, extract_ridges, Wavelet

def wavelet_ridge_frequency(signal, fs=1.0):
    wavelet = Wavelet('gmw')  # Generalized Morse Wavelet
    Wx, scales = cwt(signal, wavelet)
    ridge_idxs, *_ = extract_ridges(Wx, scales, penalty=2.0, n_ridges=1, bw=25)
    ridge = ridge_idxs[0]
    freqs_axis = wavelet.center_frequency * fs / scales
    inst_freq = freqs_axis[ridge]
    return inst_freq
```

Вариант с синхросквизингом (ещё точнее):
```python
from ssqueezepy import ssq_cwt, extract_ridges
Tx, Wx, ssq_freqs, scales, *_ = ssq_cwt(signal, wavelet='gmw')
ridge_idxs, *_ = extract_ridges(Tx, ssq_freqs, penalty=2.0, n_ridges=1)
```

**Преимущества**: не требует моноком-понентности, нет краевых артефактов, хорошо работает с затуханием амплитуды (мёртвая зона).

### 2b. EMD + Hilbert (Hilbert-Huang Transform)

Библиотека: `emd` (`pip install emd`)

```python
import emd

def hht_frequency(signal, sample_rate=1.0):
    imf = emd.sift.mask_sift(signal, max_imfs=5)
    IP, IF, IA = emd.spectra.frequency_transform(imf, sample_rate, 'nht')
    energies = np.sum(IA**2, axis=0)
    dominant_idx = np.argmax(energies)
    return IF[:, dominant_idx], IA[:, dominant_idx]
```

**Преимущества**: полностью адаптивный, не нужно задавать полосу частот.
**Недостатки**: проблема смешения мод, медленнее вейвлета.

### 2c. PLL (фазовая автоподстройка частоты)

Программный PLL отслеживает фазу, замыкая NCO (numerically controlled oscillator) на входной сигнал. Выход петлевого фильтра = мгновенная частота.

```python
def pll_demodulate(signal, f_center, fs=1.0, bw=0.001, zeta=0.707):
    N = len(signal)
    wn = 2 * np.pi * bw / (zeta + 1.0 / (4 * zeta))
    Kp = 2 * zeta * wn / fs
    Ki = (wn / fs) ** 2

    phase_acc = 0.0
    integrator = 0.0
    freq_out = np.zeros(N)

    for n in range(N):
        nco_i = np.cos(2 * np.pi * f_center * n / fs + phase_acc)
        nco_q = -np.sin(2 * np.pi * f_center * n / fs + phase_acc)
        phase_error = np.arctan2(signal[n] * nco_q, signal[n] * nco_i)
        integrator += Ki * phase_error
        filtered = Kp * phase_error + integrator
        phase_acc += filtered
        freq_out[n] = f_center + filtered * fs / (2 * np.pi)

    return freq_out
```

**Преимущества**: отличная шумоустойчивость, в мёртвой зоне удерживает последнюю частоту, нет краевых артефактов.
**Недостатки**: нужна начальная оценка центральной частоты, ручная настройка полосы.

### 2d. STFT Ridge (кратковременное Фурье)

```python
from scipy.signal import stft

def stft_ridge_frequency(signal, fs=1.0, nperseg=256, noverlap=250):
    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    ridge_idx = np.argmax(np.abs(Zxx), axis=0)
    inst_freq = f[ridge_idx]
    return np.interp(np.arange(len(signal)) / fs, t, inst_freq)
```

Простой, но плохое время-частотное разрешение.

### Сравнение методов

| Метод | Моноком-понентность | Краевые артефакты | Мёртвая зона | Разрешение | Сложность |
|-------|-------------------|-----------------|-------------|-----------|-----------|
| Raw Hilbert | Требуется | Умеренные | Плохо | Лучшее временное | Низкая |
| CWT Ridge | Не нужна | Минимальные | Хорошо | Хорошее | Средняя |
| Synchrosqueezed CWT | Не нужна | Минимальные | Хорошо | Лучшее частотное | Средняя |
| EMD+Hilbert | Не нужна | Низкие | Среднее | Адаптивное | Средняя |
| PLL | Не нужна | Нет | Отлично | Настраиваемое | Высокая |
| STFT Ridge | Не нужна | Умеренные | Среднее | Худшее | Низкая |

---

## 3. Правильная предобработка для интерферометрических сигналов

### Проблема текущего bandpass

Частота полос **меняется во времени** (пропорциональна скорости роста). Фиксированный bandpass:
- Обрезает высокочастотные полосы в начале (быстрый рост, высокое переохлаждение)
- Обрезает низкочастотные полосы вблизи мёртвой зоны (медленный рост)

### Рекомендации

**1. Только highpass (не bandpass!)** — убирает дрейф, сохраняет все частоты полос:
```python
sos = butter(4, 0.0002, btype='high', fs=1.0, output='sos')
filtered = sosfiltfilt(sos, signal)
```

**2. Вейвлет-шумоподавление** (адаптивное, не режет полосу):
```python
import pywt

def wavelet_denoise(signal, wavelet='db8', level=None):
    if level is None:
        level = min(pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len), 8)
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    denoised = [coeffs[0]]  # приближение не трогаем
    for c in coeffs[1:]:
        denoised.append(pywt.threshold(c, threshold, mode='soft'))
    return pywt.waverec(denoised, wavelet)[:len(signal)]
```

**3. Savitzky-Golay для удаления тренда** (сохраняет всё осцилляторное):
```python
trend = savgol_filter(signal, window_length=1001, polyorder=2)
detrended = signal - trend
```

### Обработка мёртвой зоны

В мёртвой зоне **нет фазовой информации**. Любой метод даст шум.

Рекомендация:
1. Определить границу мёртвой зоны по огибающей (envelope < 15% от max)
2. Установить R=0 в мёртвой зоне
3. Для фазовых методов: зафиксировать фазу на последнем надёжном значении

---

## 4. Стратегия сглаживания: фазу, не скорость

**Правило: сглаживать кумулятивную величину (фазу/L), а НЕ скорость R.**

- Фаза L — интеграл (кумулятивная сумма R), она inherently глаже
- Дифференцирование усиливает шум. Сглаживание ДО дифференцирования всегда лучше
- Сглаживание R(σ) напрямую вносит смещение из-за нелинейности (неравенство Йенсена)

Текущий код уже сглаживает L через LOWESS (`build_phase_dstep:447-457`). Нужно усилить:
- Заменить LOWESS на **монотонный сглаживающий сплайн**
- Или использовать **гауссовский процесс** (GP regression) для сглаживания с оценкой неопределённости

---

## 5. Автокорреляция в оценках скорости

### Проблема
Соседние rate[i] и rate[i+1] коррелированы: в классическом методе делят общий экстремум, в d-step — общие участки сигнала.

### Решения

**5a. Обобщённый МНК (GLS) с AR(1) моделью ошибки:**
```python
from scipy.optimize import least_squares

def fit_with_ar1(sigma, rate, w=1.0):
    def residual(params):
        s0, s1, rho = params[:3]
        pred = s0 * np.maximum(sigma - s1, 0)**w
        resid = rate - pred
        # Prais-Winsten transformation
        transformed = np.empty_like(resid)
        transformed[0] = resid[0] * np.sqrt(1 - rho**2)
        transformed[1:] = resid[1:] - rho * resid[:-1]
        return transformed

    result = least_squares(residual, [0.5, 1.0, 0.5],
                          bounds=([0, -5, -0.99], [100, 10, 0.99]),
                          loss='soft_l1')
    return result.x
```

**5b. Подвыборка через один** (независимые оценки):
```python
rates_odd = rates[0::2]
rates_even = rates[1::2]
# Фит каждой подвыборки, усреднение параметров
```

---

## 6. Robust-фиттинг

### 6a. Robust loss (Huber/soft_l1)

```python
from scipy.optimize import least_squares

def fit_robust(sigma, rate, w=1.0, s0_init=0.5, s1_init=1.0):
    def residual(params):
        s0, s1 = params
        return rate - s0 * np.maximum(sigma - s1, 0)**w

    result = least_squares(residual, [s0_init, s1_init],
                          loss='soft_l1', f_scale=0.1,
                          bounds=([0, -10], [200, np.max(sigma)]))
    return result.x
```

### 6b. Bootstrap доверительные интервалы

```python
def bootstrap_fit(sigma, rate, w=1.0, n_boot=200):
    n = len(sigma)
    s0_samples, s1_samples = [], []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        s0, s1, _ = _grid_search_s1_mathcad(sigma[idx], rate[idx], w)
        s0_samples.append(s0)
        s1_samples.append(s1)
    return np.median(s0_samples), np.median(s1_samples), \
           np.percentile(s0_samples, [2.5, 97.5]), \
           np.percentile(s1_samples, [2.5, 97.5])
```

### 6c. Профильное правдоподобие для s1

s0 определяется аналитически при заданном s1. Фиттинг — одномерная оптимизация по s1. Профильное правдоподобие даёт корректные доверительные интервалы:
- Для каждого s1 вычисляем оптимальное s0 и RSS
- 95% ДИ: все s1 где RSS < RSS_min * (1 + 3.84/(n-2))

---

## Приоритизированный план внедрения

1. **Высший приоритет, легко**: Сплайн + монотонность вместо LOWESS в `build_phase_dstep` → убирает осцилляции в Classic
2. **Высший приоритет, средняя сложность**: CWT Ridge через `ssqueezepy` как альтернатива Hilbert в `run_modern`
3. **Средний приоритет, легко**: Highpass вместо bandpass в предобработке Modern
4. **Средний приоритет, легко**: Robust loss в фиттинге через `least_squares`
5. **Ниже**: PLL для проблемных сигналов, GLS/bootstrap для доверительных интервалов

### Зависимости

```
pip install ssqueezepy emd
```

---

## Источники

- Fringe Denoising Algorithms Review, Optics and Lasers in Engineering, 2020
- Digital Processing Techniques for Fringe Analysis, IntechOpen
- EMD Python package: https://emd.readthedocs.io/
- ssqueezepy: https://github.com/OverLordGoldDragon/ssqueezepy
- scipy.optimize.least_squares: robust loss functions
- Phase recovery from fringe patterns using CWT (Optics & Lasers)
- Wavelet ridge instantaneous frequency extraction (IEEE)
- Dynamic measurement via laser interferometry: crystal growth monitoring (NTU)
- Review on Recent Advances in Signal Processing in Interferometry (PMC, 2025)
