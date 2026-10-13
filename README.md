# Satellite Segmentation Lab

An inspectable CPU image segmentation training/evaluation baseline on **generated RGB land/water tiles**, with a local frontend and SQLite experiment history.

```sh
python -m pip install -r requirements.txt
python app.py
# http://127.0.0.1:8084
python pipeline.py --seed 42 --output result.json
python -m unittest -v test_e2e.py
```

Train logistic regression on RGB pixels from twelve generated 48x48 tiles. Evaluate four held-out image seeds; export confusion counts, water IoU, accuracy, coefficients and every RGB/truth/prediction tile. Seeds are disjoint at image level, rather than randomly splitting neighboring pixels from the same image. The viewer supports experiment seeds, comparison, saved history, reload and JSON export.

The generated-data baseline is intentionally small and easy to reproduce. It is not a validated deep-learning architecture or satellite sensing result. A follow-up should use licensed real multispectral tiles, region/time heldouts, cloud masks, geospatial metadata and an actual convolutional model. Synthetic holdout metrics do not establish sensor or geographic generalization. No fabricated real imagery, profitability or production deployment claims. Server binds to loopback; authentication is required before public hosting.
