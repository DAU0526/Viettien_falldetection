# fall-detection-yolov8-demo

Project demo phat hien te nga tren 1 video bang YOLOv8n. Muc tieu cua project la train model tren Kaggle, tai file `best.pt` ve may, sau do chay script de ve bounding box va xuat video ket qua.

## Cong nghe su dung

- Python
- YOLOv8n
- OpenCV
- Kaggle

## Cau truc thu muc

```text
fall-detection-yolov8-demo/
├── data/
│   └── data.yaml
├── models/
│   └── best.pt
├── videos/
│   ├── input/
│   │   └── test_fall.mp4
│   └── output/
│       └── result_fall.mp4
├── src/
│   ├── predict_video.py
│   └── check_model.py
├── app/
│   └── demo_app.py
├── requirements.txt
└── README.md
```

## Train model tren Kaggle

1. Tao notebook tren Kaggle va bat GPU.
2. Upload dataset YOLO va file `data.yaml`.
3. Cai thu vien:

```bash
pip install ultralytics
```

4. Train YOLOv8n:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="/kaggle/input/your-dataset/data.yaml", epochs=50, imgsz=640)
```

Sau khi train xong, model tot nhat nam o:

```text
runs/detect/train/weights/best.pt
```

## Tai best.pt ve may

Tren Kaggle, mo tab Output cua notebook, tim file:

```text
runs/detect/train/weights/best.pt
```

Tai file nay ve may va dat vao project tai:

```text
models/best.pt
```

## Dat video demo

Dat video can test vao dung duong dan:

```text
videos/input/test_fall.mp4
```

## Cai dat thu vien tren may

```bash
pip install -r requirements.txt
```

## Chay phat hien te nga

```bash
python src/predict_video.py
```

Neu phat hien class `fall`, video se hien box mau do va text `FALL DETECTED`. Cac class khac duoc ve box mau xanh.

## Ket qua dau ra

Video ket qua duoc luu tai:

```text
videos/output/result_fall.mp4
```
