Judul: I am Thanos

Anggota:
Andry Herwansyah
Justin Halim
Dandy Arkandhiya Putra

Deskripsi Proyek:
Proyek ini menerima input berupa :

- video ekspresi marah wajah pengguna. output dari video akan merubah kulit wajah menjadi ungu dan memperlebar wajah agar mirip Thanos.
- Suara pengguna yang akan diubah menjadi lebih berat menyerupai Thanos.

Proyek ini menggunakan library:

- NumPy
- OpenCV
- PyAudio
- MediaPipe
- SciPy.signal

  4.2 Manipulasi Audio
  Pada bagian ini, implementasi manipulasi audio dilakukan dengan fokus pada transformasi suara real-time menggunakan efek yang menyerupai suara karakter Thanos. Proses manipulasi melibatkan beberapa tahap, termasuk noise gate, perubahan pitch, filter low-pass, resonansi, dan normalisasi amplitudo. Seluruh proses dilakukan secara real-time menggunakan mikrofon sebagai input dan speaker sebagai output.

Modul dan Dependensi
Berikut adalah modul-modul Python yang digunakan dalam implementasi ini:

1. pyaudio : Digunakan untuk menangkap input audio dari mikrofon dan menghasilkan output ke speaker.
2. numpy : Digunakan untuk pengolahan data numerik pada sinyal audio.
3. scipy.signal : Berfungsi untuk menerapkan filter digital (low-pass) pada sinyal audio.

Langkah Implementasi

1. Konfigurasi Audio :

   - Format audio: 16-bit PCM mono.
   - Sampling rate: 44,100 Hz.
   - Ukuran buffer: 2,048 sampel.

2. Komponen Utama :

   - Noise Gate : Menghilangkan noise rendah dengan membandingkan amplitudo sinyal terhadap ambang batas.
   - Pitch Shifting : Menggunakan interpolasi untuk meningkatkan atau menurunkan pitch suara.
   - Filter Low-Pass : Mengurangi frekuensi tinggi untuk menghasilkan efek suara lebih berat.
   - Resonansi : Menambahkan efek resonansi dengan penggandaan amplitudo sinyal.
   - Normalisasi Amplitudo : Menyeimbangkan volume suara untuk menghindari distorsi.

3. Alur Program :
   - Input audio diterima dari mikrofon dalam bentuk potongan (chunk) data.
   - Sinyal audio diproses menggunakan fungsi `thanos_effect`, yang menerapkan semua efek suara secara berurutan.
   - Hasil audio yang telah dimanipulasi dikirim kembali ke output (speaker) secara real-time.

Fungsi-Fungsi Utama

1. Noise Gate
   def noise_gate : Fungsi ini menghilangkan sinyal dengan amplitudo rendah untuk meredam noise latar belakang.
2. Pitch Shifting
   def pitch_shift : Fungsi ini mengubah pitch suara dengan interpolasi, memungkinkan suara menjadi lebih tinggi atau lebih rendah. Hasilnya yaitu pitch suara diturunkan untuk menciptakan efek suara lebih berat.
3. Low-Pass Filter
   def low_pass_filter : Filter low-pass digunakan untuk meredam frekuensi tinggi dan menghasilkan suara yang lebih berat.
4. Efek Thanos
   def thanos_effect : Fungsi ini adalah kombinasi dari semua efek yang diterapkan untuk menghasilkan suara khas Thanos.

Kesimpulan
Implementasi berhasil menghasilkan filter suara Thanos real-time dengan efek suara yang mendalam dan dramatis. Sistem ini menunjukkan bahwa manipulasi audio berbasis Python dapat digunakan untuk keperluan kreatif, seperti menghasilkan efek suara unik untuk karakter atau hiburan.
