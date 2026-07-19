# Детекция переломов на рентгеновских снимках

Сервис принимает рентгеновский снимок, запускает выбранную YOLO-модель и возвращает
координаты найденных областей вместе с размеченным изображением.

- [Приложение](https://jcvh3mrbarmvekrnoybn6d.streamlit.app/)
- [Swagger API](https://bone-fracture-detector-xpw8.onrender.com/docs)
- [Отчёт по экспериментам](docs/project_report.md)

Backend работает на бесплатном инстансе Render. После простоя первый запрос может занять
до минуты, а обработка снимка на CPU — ещё несколько десятков секунд.

## Возможности

- загрузка JPEG, PNG и WEBP до 10 МБ;
- выбор между YOLO11n/640 и YOLO11s/768;
- настройка confidence threshold;
- дополнительный режим повышенной чувствительности для Fast-модели;
- bounding boxes, confidence и время обработки в интерфейсе;
- отдельный FastAPI backend с документацией OpenAPI.

```text
Streamlit → POST /predict → FastAPI → ModelManager → YOLO
                                      ↓
                       JSON + размеченный PNG
```

## Модели и результаты

Обе модели обучены на одной и той же одноклассовой выборке. Fast использует YOLO11n и
вход 640 px, Accurate — YOLO11s и 768 px. Идея второго профиля состояла в том, чтобы
проверить влияние размера сети и разрешения на небольшие области.

| Профиль | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | CPU, мс | Вес |
|---|---:|---:|---:|---:|---:|---:|
| Fast | 0.154 | 0.043 | 0.298 | 0.198 | 111.1 | 5.19 МБ |
| Accurate | 0.149 | 0.041 | 0.260 | 0.187 | 338.7 | 18.27 МБ |

Увеличение модели не улучшило метрики: на test split Fast оказался немного точнее и
примерно втрое быстрее. Порог `mAP@0.5 = 0.5` из ТЗ не достигнут, поэтому модельная часть
соответствует 1 баллу из 4.

### Повышенная чувствительность

Если обычный Fast-проход не находит объектов, сервис делит снимок на две перекрывающиеся
полосы и запускает модель повторно. Результат принимается при confidence от `0.28` и только
при наличии пересекающегося слабого сигнала на исходном изображении.

На test split при пороге `0.25` этот режим изменил показатели следующим образом:

| Режим | Precision | Recall | F1 |
|---|---:|---:|---:|
| Обычный | 0.344 | 0.115 | 0.172 |
| Повышенная чувствительность | 0.342 | 0.146 | 0.204 |

## Данные

Используется Kaggle-датасет
[Bone Fracture Detection: Computer Vision Project](https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project)
(CC BY 4.0). В исходной разметке семь анатомических классов. Из-за сильного дисбаланса
все области объединены в один класс `fracture`; координаты разметки при этом сохранены.

| Split | Изображения | Объекты | Без объектов |
|---|---:|---:|---:|
| Train | 3 631 | 2 088 | 1 827 |
| Validation | 348 | 204 | 175 |
| Test | 169 | 96 | 86 |

Сегментационные полигоны исходного архива преобразуются в минимальные axis-aligned
bounding boxes скриптом `prepare_detection_dataset.py`. Аудит дополнительно проверяет
изображения, координаты, ID классов и дубликаты между split.

![Примеры разметки](reports/figures/annotation_examples.jpg)

## Структура репозитория

```text
backend/             FastAPI endpoints
frontend/            интерфейс Streamlit
fracture_detector/   инференс, модели, схемы и обработка изображений
notebooks/           обучение и оценка в Google Colab
scripts/             подготовка данных, аудит, обучение и evaluation
models/              веса Fast и Accurate
reports/             метрики, аудит и графики
docs/                отчёт и план видеопрезентации
tests/               тесты API и служебных модулей
```

Датасет и промежуточные training runs не хранятся в Git. Финальные веса и результаты
оценки находятся в `models/` и `reports/`.

## Локальный запуск

Проект рассчитан на Python 3.10–3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-ml.txt
```

Backend:

```bash
uvicorn backend.main:app --reload
```

Frontend запускается в другом терминале:

```bash
source .venv/bin/activate
streamlit run frontend/app.py
```

После запуска доступны:

- Streamlit: <http://localhost:8501>
- Swagger: <http://localhost:8000/docs>
- healthcheck: <http://localhost:8000/health>

То же окружение можно поднять через Docker:

```bash
docker compose up --build
```

## API

| Метод | Путь | Назначение |
|---|---|---|
| `GET` | `/health` | состояние сервиса и число доступных моделей |
| `GET` | `/models` | список моделей и их параметры |
| `POST` | `/predict` | инференс одного изображения |

Поля `POST /predict`:

- `file`: JPEG, PNG или WEBP;
- `model_name`: `fast` или `accurate`;
- `confidence`: число от `0.05` до `0.95`;
- `sensitivity_mode`: второй проход Fast-модели.

Пример запроса:

```bash
curl -X POST http://localhost:8000/predict \
  -F file=@xray.jpg \
  -F model_name=fast \
  -F confidence=0.25 \
  -F sensitivity_mode=true
```

## Подготовка данных и обучение

Весь цикл собран в
[`notebooks/01_colab_train_and_evaluate.ipynb`](notebooks/01_colab_train_and_evaluate.ipynb).
В Colab нужно выбрать T4 GPU и выполнить ячейки сверху вниз. Результаты сохраняются в
`MyDrive/bone-fracture-detector/`.

Те же команды доступны отдельно:

```bash
python scripts/download_dataset.py
python scripts/prepare_detection_dataset.py \
  --source "data/bone-fracture/bone fracture detection.v4-v4.yolov8" \
  --output data/bone-fracture-detect-one-class \
  --single-class
python scripts/audit_dataset.py \
  --data data/bone-fracture-detect-one-class/data.yaml \
  --output reports/data_audit.json
python scripts/train.py \
  --data data/bone-fracture-detect-one-class/data.yaml \
  --profile fast --device 0
python scripts/evaluate.py \
  --data data/bone-fracture-detect-one-class/data.yaml \
  --split test --device 0
```

## Проверка кода

```bash
pip install -r requirements-dev.txt
ruff check .
pytest --cov=fracture_detector --cov=backend
```

CI запускает эти проверки при каждом push и pull request.

## Деплой

Render использует `Dockerfile` и настройки из `render.yaml`. Для Streamlit Community Cloud
entrypoint — `frontend/app.py`; адрес backend задаётся в secrets:

```toml
API_URL = "https://bone-fracture-detector-xpw8.onrender.com"
```

Сценарий для записи итогового видео находится в
[`docs/presentation_script.md`](docs/presentation_script.md).
