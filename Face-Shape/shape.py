import cv2
import numpy as np
import mediapipe as mp

# Inisialisasi MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Inisialisasi webcam
cap = cv2.VideoCapture(0)

def modify_face_shape(image, face_points):
    h, w = image.shape[:2]
    
    # Titik-titik kunci untuk modifikasi bentuk wajah Thanos
    jaw_points = face_points[152:175]  # Points for jaw line
    chin_point = face_points[152]      # Center of chin
    
    # Pelebaran rahang
    jaw_modifier = 1.2  # Faktor skala untuk rahang
    
    # Membuat mesh deformasi
    output = image.copy()
    
    # Membuat triangulasi untuk area wajah
    rect = (0, 0, w, h)
    subdiv = cv2.Subdiv2D(rect)
    
    for point in face_points:
        subdiv.insert(point)
    
    # Aplikasikan deformasi pada rahang
    for point in jaw_points:
        x, y = point
        center_x = chin_point[0]
        # Geser titik horizontal untuk membuat rahang lebih lebar
        new_x = center_x + (x - center_x) * jaw_modifier
        cv2.circle(output, (int(new_x), y), 1, (0, 255, 0), -1)
    
    return output

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Gagal mengambil frame dari kamera")
        continue

    # Konversi BGR ke RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Deteksi wajah menggunakan MediaPipe
    results = face_mesh.process(image_rgb)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Mendapatkan koordinat wajah
            h, w, _ = image.shape
            face_points = []
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                face_points.append((x, y))
            
            # Modifikasi bentuk wajah
            image = modify_face_shape(image, face_points)
            
            # Visualisasi landmark wajah
            hull = cv2.convexHull(np.array(face_points))
            cv2.polylines(image, [hull], True, (0, 255, 0), 1)

    # Tampilkan hasil
    cv2.imshow('Thanos Face Shape', image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()