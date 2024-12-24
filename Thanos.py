import cv2
import mediapipe as mp
import numpy as np
import pyaudio
import threading
from scipy.signal import butter, lfilter

# Inisialisasi MediaPipe Face Mesh untuk deteksi wajah
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Konfigurasi audio
CHUNK = 1024  # Ukuran chunk dikurangi untuk mengurangi latency
FORMAT = pyaudio.paFloat32  # Menggunakan format float32 untuk kualitas lebih baik
CHANNELS = 1
RATE = 44100
NOISE_THRESHOLD = 0.02

def modify_face_shape(image, landmarks):
    """
    Fungsi untuk memodifikasi bentuk wajah:
    - Memperlebar bagian rahang
    - Membuat wajah lebih kekar
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Konversi landmark ke array numpy untuk memudahkan manipulasi
    points = np.array([(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark])
    
    # Perlebar rahang (20%)
    jaw_points = points[152:175]
    center = np.mean(jaw_points, axis=0)
    jaw_points = (jaw_points - center) * 1.2 + center
    
    # Gambar mask wajah
    cv2.fillConvexPoly(mask, points.astype(int), 255)
    
    return mask

def apply_thanos_color(frame, mask):
    """
    Fungsi untuk memberikan efek warna ungu Thanos:
    - Menerapkan warna ungu pada area wajah
    - Mempertahankan detail dan tekstur wajah
    """
    # Warna ungu Thanos (R,G,B)
    thanos_color = np.array([128, 0, 128], dtype=np.float32)
    
    # Blur mask untuk transisi halus
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    mask = mask / 255.0
    
    # Aplikasikan warna dengan mempertahankan tekstur
    result = frame.copy()
    for c in range(3):
        result[:,:,c] = frame[:,:,c] * (1 - mask * 0.5) + thanos_color[c] * mask * 0.5
        
    return result

def process_audio(data, pitch_factor=0.7):
    """
    Fungsi untuk memproses audio:
    - Menurunkan pitch suara
    - Menambah bass
    - Menghilangkan noise
    """
    # Konversi ke float32 untuk pemrosesan
    audio_data = np.frombuffer(data, dtype=np.float32)
    
    # Noise gate
    if np.max(np.abs(audio_data)) < NOISE_THRESHOLD:
        return np.zeros_like(audio_data)
    
    # Pitch shifting
    indices = np.arange(0, len(audio_data), pitch_factor)
    audio_data = np.interp(indices, np.arange(len(audio_data)), audio_data)
    
    # Low pass filter untuk menambah bass
    nyquist = RATE / 2
    cutoff = 800 / nyquist
    b, a = butter(4, cutoff, btype='low')
    audio_data = lfilter(b, a, audio_data)
    
    return audio_data.astype(np.float32)

def audio_stream():
    """
    Fungsi untuk menjalankan stream audio secara real-time
    """
    p = pyaudio.PyAudio()
    stream_in = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    stream_out = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK)

    while True:
        try:
            data = stream_in.read(CHUNK)
            processed = process_audio(data)
            stream_out.write(processed.tobytes())
        except:
            break

    stream_in.stop_stream()
    stream_out.stop_stream()
    stream_in.close()
    stream_out.close()
    p.terminate()

def main():
    # Mulai audio processing di thread terpisah
    audio_thread = threading.Thread(target=audio_stream)
    audio_thread.daemon = True
    audio_thread.start()
    
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
            
        # Proses frame untuk deteksi wajah
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Terapkan efek bentuk dan warna
                mask = modify_face_shape(frame, face_landmarks)
                frame = apply_thanos_color(frame, mask)
        
        cv2.imshow('Thanos Effect', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()