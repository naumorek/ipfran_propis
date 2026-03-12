# Propis IPFRAN — Обработка прописей KDP/DKDP

Программа для обработки данных интерферометрического стенда ИПФ РАН. Определяет кинетические параметры роста кристаллов KDP/DKDP из интерференционных прописей.

## Что делает

- Читает данные стенда (PRN файлы: два интерференционных канала + температура)
- Определяет **температуру насыщения** (tn) и **мёртвую зону** (Td)
- Строит **кинетические кривые** R(ΔT) и R(σ) для призмы и пирамиды
- Сравнивает с эталонными кривыми (Cfe=0 — чистый раствор, Cfe=16ppm — загрязнённый)
- Фиттинг степенным законом G(σ) = s₀·(σ - s₁)^w и BCF-моделью

## Два режима обработки

| | Классический (как Mathcad) | Современный |
|---|---|---|
| Детекция скорости | Поиск экстремумов синусоид | Преобразование Гильберта → мгновенная частота |
| Фиттинг | Степенной закон (grid search) | BCF-модель (scipy curve_fit) |
| Предобработка | medsmooth | Полный pipeline: detrend → bandpass → нормализация → вейвлет |

Предобработка включается/выключается отдельно. Итого 4 комбинации режимов.

## Структура

```
propis_app/
├── main.py                 — точка входа (PyQt6 GUI)
├── core/                   — ядро вычислений
│   ├── prn_reader.py       — чтение PRN файлов
│   ├── solubility.py       — растворимость C(T), 7 наборов коэффициентов
│   ├── preprocessing.py    — конвейер предобработки сигнала
│   ├── saturation.py       — определение tn и мёртвой зоны
│   ├── reference_curves.py — эталонные кривые
│   ├── batch.py            — пакетная обработка
│   ├── signal_processing/
│   │   ├── classic.py      — детекция экстремумов
│   │   └── modern.py       — Гильберт + мгновенная частота
│   └── kinetics/
│       ├── power_law.py    — G(σ) = s₀·(σ - s₁)^w
│       └── bcf_model.py    — R = β·(σ-σ_d)²/(σ₁+σ-σ_d)
├── gui/                    — PyQt6 интерфейс
│   ├── main_window.py      — главное окно
│   ├── signal_view.py      — сигнал + интерактивные границы
│   ├── preprocessing_view.py
│   ├── kinetic_view.py     — кинетические кривые с эталонами
│   ├── results_view.py     — таблица параметров + экспорт CSV
│   ├── comparison_view.py  — классический vs современный
│   └── batch_view.py       — пакетная обработка
└── data/
    └── reference_coefficients.json
```

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r propis_app/requirements.txt
```

## Запуск

```bash
cd propis_app
python main.py
```

## Зависимости

- Python ≥ 3.10
- NumPy, SciPy, Matplotlib
- PyQt6, pyqtgraph
- PyWavelets

## Контекст

Проект ИПФ РАН, лаборатория роста кристаллов. Замена алгоритма из Mathcad 2000 Professional на современный инструмент с GUI. Подробная документация — в `docs/`.

## Документация

- `docs/algorithm_description.md` — детальное описание алгоритма: этапы, формулы, коэффициенты
- `docs/mathcad2000_reference.md` — справочник по Mathcad 2000
- `docs/ershov_dissertation_summary.md` — конспект диссертации Ершова
- `docs/research_improvements.md` — обзор литературы и предложения по улучшению
