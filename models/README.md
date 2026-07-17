# Model weights

Final Colab artifacts are committed using these exact names:

- `fast.pt` — YOLO11n, 640 px;
- `accurate.pt` — YOLO11s, 768 px.

Both files are fine-tuned one-class (`fracture`) checkpoints, not renamed generic COCO
weights. Their reproducible test metrics are stored in `reports/metrics.json` and
`reports/metrics.csv`.
