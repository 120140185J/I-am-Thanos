import cv2
import mediapipe as mp
import numpy as np

# Inisialisasi MediaPipe Face Detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
# Fungsi untuk mengubah warna kulit menjadi ungu
def apply_thanos_color(frame, face_landmarks):
    modified_frame = frame.copy()
    height, width, _ = frame.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    # Ambil koordinat landmark wajah
    for landmark in face_landmarks.landmark:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(mask, (x, y), 2, 255, -1)

    # Gunakan blur untuk membuat masker lebih halus
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    modified_frame[mask > 0] = cv2.addWeighted(modified_frame[mask > 0], 0.5, (128, 0, 128), 0.5, 0)
    return modified_frame

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Tidak dapat mengakses kamera.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Tidak dapat membaca frame dari kamera.")
            break

        # Konversi frame ke RGB untuk MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Deteksi landmark wajah
        results = face_mesh.process(rgb_frame)

        # Jika wajah terdeteksi, proses efek
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                frame = apply_thanos_color(frame, face_landmarks)  # Warna wajah
        cv2.imshow('Thanos Face Color', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
