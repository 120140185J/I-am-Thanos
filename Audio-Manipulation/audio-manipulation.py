import pyaudio
import numpy as np

# Konfigurasi audio
CHUNK = 1024  # Ukuran potongan audio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# Fungsi untuk mengubah suara menjadi suara Thanos
def thanos_effect(audio_chunk, pitch_factor=0.6, resonance_factor=0.85):
    # Ubah pitch dengan interpolasi
    resampled = np.interp(
        np.arange(0, len(audio_chunk), pitch_factor),
        np.arange(0, len(audio_chunk)),
        audio_chunk,
    ).astype(audio_chunk.dtype)

    # Terapkan efek resonansi
    resonant_audio = resampled * resonance_factor
    resonant_audio = np.clip(resonant_audio, -32768, 32767).astype(np.int16)

    return resonant_audio

# Fungsi utama untuk memproses audio secara real-time
def realtime_voice_filter():
    audio_interface = pyaudio.PyAudio()

    # Inisialisasi stream input (mikrofon)
    stream_input = audio_interface.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    # Inisialisasi stream output (speaker)
    stream_output = audio_interface.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        output=True,
        frames_per_buffer=CHUNK,
    )

    print("Filter suara Thanos aktif. Bicara ke mikrofon...")
    try:
        while True:
            # Baca data audio dari mikrofon
            input_data = stream_input.read(CHUNK, exception_on_overflow=False)
            audio_chunk = np.frombuffer(input_data, dtype=np.int16)

            # Terapkan efek suara Thanos
            modified_audio = thanos_effect(audio_chunk)

            # Hindari memantulkan suara output ke input
            if np.any(audio_chunk):  # Proses hanya jika ada sinyal dari mikrofon
                stream_output.write(modified_audio.tobytes())
    except KeyboardInterrupt:
        print("\nProgram dihentikan oleh pengguna.")
    finally:
        # Tutup stream dan akhiri sesi
        stream_input.stop_stream()
        stream_input.close()
        stream_output.stop_stream()
        stream_output.close()
        audio_interface.terminate()

if __name__ == "__main__":
    realtime_voice_filter()
