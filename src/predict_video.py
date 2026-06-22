from pathlib import Path

import cv2
from ultralytics import YOLO


# Lay duong dan goc cua project: lezzi_datasets/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Cac duong dan mac dinh theo yeu cau
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
INPUT_VIDEO = PROJECT_ROOT / "videos" / "input" / "test_fall.mp4"
OUTPUT_VIDEO = PROJECT_ROOT / "videos" / "output" / "result_fall.mp4"

# Cau hinh mau ve box theo dinh dang BGR cua OpenCV
RED = (0, 0, 255)
GREEN = (0, 180, 0)
WHITE = (255, 255, 255)

CONF_THRESHOLD = 0.25
FALL_WARNING_FRAMES = 5


def draw_text_with_background(frame, text, position, color, scale=0.7, thickness=2):
    """Ve chu co nen mau de de doc tren video."""
    x, y = position
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)

    cv2.rectangle(
        frame,
        (x, y - text_h - baseline - 8),
        (x + text_w + 10, y + 4),
        color,
        -1,
    )
    cv2.putText(frame, text, (x + 5, y - 4), font, scale, WHITE, thickness, cv2.LINE_AA)


def main():
    # Kiem tra file dau vao truoc khi chay
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Khong tim thay model: {MODEL_PATH}")
    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(f"Khong tim thay video dau vao: {INPUT_VIDEO}")

    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

    # Load model YOLOv8 tu file best.pt
    model = YOLO(str(MODEL_PATH))
    class_names = model.names

    # Mo video dau vao
    cap = cv2.VideoCapture(str(INPUT_VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Khong mo duoc video: {INPUT_VIDEO}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    # Tao video dau ra dang mp4
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Khong tao duoc video dau ra: {OUTPUT_VIDEO}")

    fall_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Chay du doan tren tung frame
        result = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)[0]
        has_fall_in_frame = False

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = str(class_names[class_id])

            # Neu class la fall thi ve mau do, nguoc lai ve mau xanh
            is_fall = class_name.lower() == "fall"
            color = RED if is_fall else GREEN

            if is_fall:
                has_fall_in_frame = True
                label = f"FALL DETECTED {confidence:.2f}"
            else:
                label = f"{class_name} {confidence:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            draw_text_with_background(frame, label, (x1, max(y1, 25)), color)

        # Dem so frame lien tuc co fall
        if has_fall_in_frame:
            fall_frame_count += 1
        else:
            fall_frame_count = 0

        # Canh bao lon neu fall xuat hien lien tuc tu 5 frame tro len
        if fall_frame_count >= FALL_WARNING_FRAMES:
            warning_text = "WARNING: FALL DETECTED!"
            cv2.rectangle(frame, (0, 0), (width, 90), RED, -1)
            cv2.putText(
                frame,
                warning_text,
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                WHITE,
                4,
                cv2.LINE_AA,
            )

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Da luu video ket qua: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
