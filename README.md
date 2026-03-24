# Propis IPFRAN — Обработка интерферометрических данных роста кристаллов KDP/DKDP

Автоматизированная обработка данных лазерной интерферометрии для определения кинетических параметров роста кристаллов KDP и DKDP. Разработано в ИПФ РАН (Институт прикладной физики Российской академии наук).

## Назначение

Система обрабатывает записи двухканального интерферометра (LED1 + LED2) и определяет:

- **te** — температура насыщения раствора (°C)
- **tn** — фактическая температура насыщения из кинетических данных (°C)
- **Td** — ширина мёртвой зоны (°C), индикатор загрязнённости раствора
- **R(dT)** — кинетическая кривая: скорость роста от переохлаждения (мм/день)
- **Sig035** — пересыщение при R = 0.35 мкм/мин (σ%)
- **s2** — параметр формы кривой (экстраполяция √R vs σ)

Входные данные: файлы `.prn` (ASCII, ~9000 отсчётов при 1 Гц) с 10+ колонками (LED каналы, температура, время).

## Два baseline

### Baseline 1: Classic (Mathcad 2000)
Точное воспроизведение алгоритма Mathcad 2000 в Python (~99% совпадение):
- arcsin-фаза в 17 экстремумах → толщина L (мкм)
- LOESS(L, degree=2, span=0.15) → dL/dt → R (мкм/мин)
- Grid search tn по эталонным кривым Si/Si1
- LOESS R(σ, degree=2, span=0.2) → кривая F1

### Baseline 2: CV модель (Кабрера-Вермильи)
Физически обоснованная модель кинетики спирального роста с примесями:
```
R(σ) = β · (σ - σ_dead)² / (σ₁ + σ - σ_dead)   при σ > σ_dead
R(σ) = 0                                          при σ ≤ σ_dead
```
3 интерпретируемых параметра: β (кинетический коэфф.), σ_dead (мёртвая зона), σ₁ (BCF-переход).

## Структура проекта

```
propis_app/                     # Основной Python-пакет
├── core/
│   ├── pipeline.py             # Classic + Modern pipeline (run_classic, run_modern)
│   ├── prn_reader.py           # Чтение файлов PRN
│   ├── mcd_reader.py           # Парсинг Mathcad MCD (извлечение параметров)
│   ├── solubility.py           # Кривые растворимости KDP/DKDP (7 наборов коэфф.)
│   ├── reference_curves.py     # Эталонные кривые Si/Si1
│   ├── preprocessing.py        # Предобработка сигнала
│   ├── saturation.py           # Определение te, tn
│   ├── auto_detect.py          # Автодетекция параметров
│   ├── batch.py                # Пакетная обработка
│   ├── signal_processing/
│   │   ├── classic.py          # arcsin d-step фаза (как Mathcad)
│   │   └── modern.py           # Hilbert, CWT Ridge, PLL, STFT
│   └── kinetics/
│       ├── power_law.py        # Степенной фит R(σ), dissolution, Sig035, s2
│       └── bcf_model.py        # BCF/CV модель
├── gui/                        # PyQt GUI
│   ├── main_window.py
│   ├── signal_view.py          # Визуализация сигнала
│   ├── kinetic_view.py         # R(dT), R(σ) графики
│   ├── comparison_view.py      # Сравнение Classic vs Modern
│   ├── results_view.py         # Таблица параметров + экспорт CSV
│   └── batch_view.py           # Пакетная обработка

scripts/
├── mathcad_exact.py            # Standalone Mathcad reproduction (~1300 строк)
├── plot_mathcad_style.py       # Classic vs Modern сравнительные графики
├── plot_algorithm_analysis.py  # 5-панельный анализ алгоритма
└── validate_pipeline.py        # Валидация vs Mathcad reference values

docs/
├── ALGORITHM_GUIDE.md          # Практический гид по алгоритму
├── mathcad_full_extraction.md  # Полная экстракция из 32 скриншотов Mathcad
├── audit_vs_mathcad.md         # Аудит Python vs Mathcad (30 скриншотов)
├── experiment_log.md           # Журнал экспериментов (14 итераций)
├── ANALYSIS_DEAD_ZONE_ARTIFACT.pdf  # Артефакт LOESS в мёртвой зоне
├── CV_MODEL_REVIEW.pdf         # Обзор модели Кабреры-Вермильи
├── CV_LIMITATIONS_REPORT.pdf   # 8 ограничений CV модели
├── research_dead_zone_modern.md    # Dead zone в Modern pipeline
├── research_dual_channel.md        # Двухканальная обработка LED1+LED2
├── research_signal_cleaning.md     # Очистка сигнала: CWT/EMD/PLL
├── research_tn_key_methods.md      # Научный обзор: 5 глав по ключевым методам
├── algorithm_description.md        # Детальное описание алгоритма
├── ershov_dissertation_summary.md  # Конспект диссертации Ершова
└── ...
```

## Быстрый старт

### Установка
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy scikit-misc PyWavelets matplotlib pyqt5
```

### GUI
```bash
cd propis_app
python main.py
```

### Обработка одного файла (Classic — точное воспроизведение Mathcad)
```bash
python scripts/mathcad_exact.py "020326 KDP нейтр XXXIII п.17/__020326_5_1" --method classic
```

### Обработка с Hilbert (Hybrid: arcsin + Hilbert fusion)
```bash
python scripts/mathcad_exact.py "020326 KDP нейтр XXXIII п.17/__020326_5_1" --method hybrid
```

### Сравнительные графики Classic vs Modern
```bash
python scripts/plot_mathcad_style.py
```

### Анализ алгоритма (5-панельная визуализация)
```bash
python scripts/plot_algorithm_analysis.py
```

## Зависимости

- Python ≥ 3.10
- NumPy, SciPy
- scikit-misc (`skmisc.loess` — LOESS degree=2, критично для воспроизведения Mathcad)
- PyWavelets (`pywt` — CWT Ridge)
- Matplotlib
- PyQt5 (GUI)

## Валидация vs Mathcad 2000

| Параметр | Python (Classic) | Mathcad 2000 | Разница |
|----------|-----------------|--------------|---------|
| te       | 48.60           | 48.60        | 0%      |
| tn       | 48.40           | 48.40        | 0%      |
| Td       | 0.44            | 0.44         | 0%      |
| Sigm     | 0.59            | 0.59         | 0%      |
| s2       | 0.33            | 0.32         | 3%      |
| Sig035   | 2.39            | 2.29         | 4%      |

*Тест-файл: 020326_2_1 (KDP, нейтральный раствор, грань {101})*

## Критерии качества раствора

| Параметр | Чистый раствор | Загрязнённый |
|----------|---------------|-------------|
| Td       | ≤ 0.6°C       | > 0.6°C     |
| R(dT) vs Si1 | ≥ эталон (CFe=16 ppm) | < эталон |
| σ_dead (CV) | < 0.3% | > 0.5% |

Маркер загрязнения — Fe³⁺ (элементный анализ). σ_dead ≈ 0.15·√C_Fe (ppm) при T ≈ 50°C.

## Ключевые открытия

1. **LOESS degree=2** — Mathcad использует local quadratic, не linear. `statsmodels.lowess` (degree=1) даёт неправильные результаты → нужен `skmisc.loess`
2. **Dead zone — артефакт экстраполяции** — переход R→0 при σ→σ_dead рисуется LOESS-сглаживанием. Все 17 реальных точек лежат в зоне роста, в зоне 0 < R < 0.1 данных нет
3. **df/dt автодетекция мёртвой зоны** — заменяет ручной параметр im с точностью ±0.02% по σ. Ключ к автоматической пакетной обработке
4. **CV модель** — физическая альтернатива LOESS: R = 0 в мёртвой зоне по определению, 3 интерпретируемых параметра

## Дорожная карта

1. **Улучшение алгоритма** — автодетекция параметров, df/dt определение мёртвой зоны, интеграция CV модели в pipeline
2. **Пакетная обработка** — архив ~100 ГБ (25 000+ файлов PRN) → структурированный датасет (CSV/Parquet)
3. **ML модель** — PRN → {Td, tn, R(dT), оценка качества}

## Документация

- Описание алгоритма: `docs/ALGORITHM_GUIDE.md`
- Полная экстракция Mathcad: `docs/mathcad_full_extraction.md`
- Аудит Python vs Mathcad: `docs/audit_vs_mathcad.md`
- Журнал экспериментов (14 итераций): `docs/experiment_log.md`

## Контекст

Проект ИПФ РАН, лаборатория роста кристаллов. Замена алгоритма из Mathcad 2000 Professional на современный инструмент с автоматизацией и физически обоснованными моделями.
