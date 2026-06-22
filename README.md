# fall-detection-yolov8-demo

Project demo phat hien te nga tren 1 video bang YOLOv8n. Project tap trung vao viec train model YOLO, lay file `best.pt`, sau do chay demo tren 1 video dau vao va xuat ra video ket qua co bounding box.

## Cong nghe su dung

- Python
- YOLOv8n / Ultralytics
- OpenCV
- Google Colab hoac Kaggle

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

## Chuan bi dataset YOLO

Dataset train YOLO can co dang:

```text
yolo_dataset/
├── data.yaml
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

File `data.yaml` vi du:

```yaml
path: /content/yolo_coffee_room
train: images/train
val: images/val

names:
  0: normal
  1: fall
```

## Train YOLOv8n tren Google Colab

### 1. Bat GPU

Vao `Runtime` -> `Change runtime type` -> chon `T4 GPU`.

Kiem tra GPU:

```python
!nvidia-smi
```

### 2. Cai thu vien

```python
!pip install ultralytics opencv-python -q
```

### 3. Upload dataset zip

Upload file dataset YOLO dang zip, vi du `yolo_coffee_room.zip`:

```python
from google.colab import files

uploaded = files.upload()
```

### 4. Giai nen dataset

```python
import zipfile
from pathlib import Path

zip_name = list(uploaded.keys())[0]
extract_root = Path('/content')

with zipfile.ZipFile(zip_name, 'r') as zip_ref:
    zip_ref.extractall(extract_root)

print('Da giai nen:', zip_name)
```

Neu zip co thu muc `yolo_coffee_room/`, file yaml se nam tai:

```text
/content/yolo_coffee_room/data.yaml
```

Kiem tra:

```python
!find /content/yolo_coffee_room -maxdepth 3 -type f | head -20
!cat /content/yolo_coffee_room/data.yaml
```

### 5. Train YOLOv8n

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(
    data='/content/yolo_coffee_room/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    name='fall_detection_yolov8n'
)
```

Sau khi train xong, model tot nhat nam o:

```text
/content/runs/detect/fall_detection_yolov8n/weights/best.pt
```

### 6. Tai best.pt ve may

```python
from google.colab import files

files.download('/content/runs/detect/fall_detection_yolov8n/weights/best.pt')
```

Dat file vua tai ve vao project local:

```text
models/best.pt
```

## Chay demo tren Google Colab

Neu muon chay demo truc tiep tren Colab, upload cac file sau:

- `best.pt`
- `test_fall.mp4`
- `src/predict_video.py`

### 1. Tao cau truc thu muc tren Colab

```python
!mkdir -p /content/fall-demo/models
!mkdir -p /content/fall-demo/videos/input
!mkdir -p /content/fall-demo/videos/output
!mkdir -p /content/fall-demo/src
```

### 2. Upload model, video va file predict

```python
from google.colab import files

uploaded = files.upload()
```

Sau khi upload, copy file vao dung vi tri:

```python
import shutil
from pathlib import Path

root = Path('/content/fall-demo')

shutil.copy('best.pt', root / 'models' / 'best.pt')
shutil.copy('test_fall.mp4', root / 'videos' / 'input' / 'test_fall.mp4')
shutil.copy('predict_video.py', root / 'src' / 'predict_video.py')
```

### 3. Cai thu vien

```python
!pip install ultralytics opencv-python -q
```

### 4. Chay demo

```python
%cd /content/fall-demo
!python src/predict_video.py
```

Ket qua duoc luu tai:

```text
/content/fall-demo/videos/output/result_fall.mp4
```

### 5. Tai video ket qua ve may

```python
from google.colab import files

files.download('/content/fall-demo/videos/output/result_fall.mp4')
```

## Chay demo tren may local

Dat model YOLO vao:

```text
models/best.pt
```

Dat video demo vao:

```text
videos/input/test_fall.mp4
```

Cai thu vien:

```bash
pip install -r requirements.txt
```

Kiem tra model:

```bash
python src/check_model.py
```

Chay demo:

```bash
python src/predict_video.py
```

Video dau ra:

```text
videos/output/result_fall.mp4
```

## Quy tac hien thi trong video

- Neu phat hien class `fall`: ve bounding box mau do va text `FALL DETECTED`.
- Neu class khac: ve bounding box mau xanh.
- Neu fall xuat hien lien tuc tu 5 frame tro len: hien canh bao lon tren man hinh.

## Luu y

- `best.pt` la model YOLO, khac voi `best_model.pth` cua BiLSTM.
- File `best_model.pth` khong dung duoc cho `src/predict_video.py`.
- Neu chay `src/predict_video.py` bi loi khong tim thay `models/best.pt`, can train YOLO va tai file `best.pt` truoc.
