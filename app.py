"""
Aplikasi Deteksi Emosi Real-Time (HAPPY / ANGRY / SAD)
Menggunakan model Teachable Machine (Keras .h5) + webcam.

Cara pakai:
    1. Install dependencies:  pip install -r requirements.txt
    2. Jalankan:               python app.py
    3. Tekan 'q' untuk keluar.
"""

from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import cv2

np.set_printoptions(suppress=True)

MODEL_PATH = "keras_Model.h5"
LABELS_PATH = "labels.txt"


def load_labels(path):
    with open(path, "r") as f:
        # format tiap baris: "0 HAPPY" -> ambil bagian setelah index
        return [line.strip().split(" ", 1)[1] for line in f.readlines() if line.strip()]


def load_emotion_model(path):
    try:
        return load_model(path, compile=False)
    except TypeError:
        # Fix untuk error umum "Unrecognized keyword arguments: ['groups']"
        # yang muncul kalau versi TensorFlow lebih baru dari saat model dibuat.
        import tensorflow as tf

        class PatchedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
            def __init__(self, *args, **kwargs):
                kwargs.pop("groups", None)
                super().__init__(*args, **kwargs)

        return load_model(
            path,
            compile=False,
            custom_objects={"DepthwiseConv2D": PatchedDepthwiseConv2D},
        )


def preprocess_frame(frame_bgr):
    # OpenCV pakai BGR, model butuh RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image).astype(np.float32)
    normalized = (image_array / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized
    return data


def main():
    print("Memuat model...")
    model = load_emotion_model(MODEL_PATH)
    class_names = load_labels(LABELS_PATH)
    print(f"Model siap. Label: {class_names}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Tidak bisa mengakses webcam. Pastikan webcam terhubung dan tidak dipakai aplikasi lain.")
        return

    print("Tekan 'q' pada jendela video untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari webcam.")
            break

        data = preprocess_frame(frame)
        prediction = model.predict(data, verbose=0)
        index = int(np.argmax(prediction))
        class_name = class_names[index]
        confidence = float(prediction[0][index])

        label_text = f"{class_name}: {confidence * 100:.1f}%"
        cv2.rectangle(frame, (0, 0), (320, 40), (0, 0, 0), -1)
        cv2.putText(
            frame,
            label_text,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Deteksi Emosi - Tekan 'q' untuk keluar", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
