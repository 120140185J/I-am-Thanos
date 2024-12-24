import cv2
import mediapipe as mp
import numpy as np
import pyaudio
from scipy.signal import butter, lfilter

# =====================================
# Bagian 1: Manipulasi Bentuk Wajah
# =====================================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def modify_face_shape(image, face_landmarks):
    h, w = image.shape[:2]
    jaw_points = [face_landmarks[i] for i in range(152, 175)]  # Landmark rahang
    chin_point = face_landmarks[152]  # Titik dagu tengah
    jaw_modifier = 1.2  # Faktor skala untuk pelebaran rahang
    modified_image = image.copy()
    
    for point in jaw_points:
        x, y = point
        center_x = chin_point[0]
        # Geser horizontal untuk memperlebar wajah
        new_x = center_x + (x - center_x) * jaw_modifier
        cv2.circle(modified_image, (int(new_x), int(y)), 2, (0, 255, 0), -1)
    return modified_image

# =====================================
# Bagian 2: Manipulasi Warna Wajah
# =====================================
def apply_thanos_color(frame, face_landmarks):
    height, width, _ = frame.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    for landmark in face_landmarks:
        x, y = int(landmark[0]), int(landmark[1])
        cv2.circle(mask, (x, y), 2, 255, -1)

    # Blur untuk menghaluskan masker
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    frame[mask > 0] = cv2.addWeighted(frame[mask > 0], 0.5, (128, 0, 128), 0.5, 0)
    return frame

# =====================================
# Bagian 3: Manipulasi Suara
# =====================================
CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
NOISE_THRESHOLD = 1000
TARGET_PEAK = 20000

def low_pass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return lfilter(b, a, data)

def pitch_shift(data, pitch_factor):
    indices = np.arange(0, len(data), pitch_factor)
    indices = indices[indices < len(data)]
    return np.interp(indices, np.arange(len(data)), data).astype(data.dtype)

def normalize_audio(data, target_peak=TARGET_PEAK):
    max_amplitude = np.max(np.abs(data))
    if max_amplitude == 0:
        return data
    scaling_factor = min(1.0, target_peak / max_amplitude)
    return np.clip(data * scaling_factor, -32768, 32767).astype(np.int16)

def noise_gate(data, threshold=NOISE_THRESHOLD):
    max_amplitude = np.max(np.abs(data))
    if max_amplitude < threshold:
        return np.zeros_like(data)
    return data

def thanos_effect(audio_chunk, pitch_factor=0.8, cutoff_freq=800):
    gated_audio = noise_gate(audio_chunk)
    pitched_audio = pitch_shift(gated_audio, pitch_factor)
    filtered_audio = low_pass_filter(pitched_audio, cutoff=cutoff_freq, fs=RATE)
    return normalize_audio(filtered_audio)

def realtime_voice_filter():
    audio_interface = pyaudio.PyAudio()
    stream_input = audio_interface.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    stream_output = audio_interface.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK)

    print("Efek suara Thanos aktif...")
    try:
        while True:
            input_data = stream_input.read(CHUNK, exception_on_overflow=False)
            audio_chunk = np.frombuffer(input_data, dtype=np.int16)
            modified_audio = thanos_effect(audio_chunk)
            stream_output.write(modified_audio.tobytes())
    except KeyboardInterrupt:
        pass
    finally:
        stream_input.stop_stream()
        stream_input.close()
        stream_output.stop_stream()
        stream_output.close()
        audio_interface.terminate()

# =====================================
# Bagian Utama: Menggabungkan Semua Fungsi
# =====================================
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Tidak dapat mengakses kamera.")
        return

    audio_process = None
    try:
        audio_process = realtime_voice_filter  # Proses suara berjalan paralel

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Tidak dapat membaca frame dari kamera.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for landmarks in results.multi_face_landmarks:
                    face_points = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in landmarks.landmark]
                    frame = apply_thanos_color(frame, face_points)
                    frame = modify_face_shape(frame, face_points)

            cv2.imshow("Thanos Transformation", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
