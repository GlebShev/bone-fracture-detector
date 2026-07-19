# Веса моделей

Приложение загружает два файла:

- `fast.pt` — YOLO11n, 640 px;
- `accurate.pt` — YOLO11s, 768 px.

Это веса после fine-tuning на одноклассовой разметке `fracture`, а не исходные COCO
checkpoints. Параметры обучения описаны в ноутбуке, результаты test split — в
`reports/metrics.csv` и `reports/metrics.json`.
