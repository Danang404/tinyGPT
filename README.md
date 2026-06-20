# 🧠 TinyGPT: Bitcoin & Blockchain Language Model

Sebuah implementasi mini dari model bahasa GPT (Decoder-only Transformer) yang dibangun dari nol menggunakan PyTorch. Proyek ini berfokus pada eksperimen *Language Modeling* menggunakan dataset khusus bertema Bitcoin dan Blockchain. 

Proyek ini dikembangkan sebagai bagian dari eksperimen Data Mining di Universitas Amikom Yogyakarta.

## ✨ Fitur Utama

* **Arsitektur Transformer Kustom**: Mengimplementasikan *Multi-Head Self-Attention* dengan *causal masking* dan *Position-wise Feed-Forward Network* dari awal (`transformer_blocks.py`).
* **Eksperimen Tokenizer**: Membandingkan performa 3 pendekatan tokenisasi berbeda (`tokenizers.py`):
    1.  **Character-Level**: Tokenisasi tingkat karakter (vocab sangat kecil, urutan panjang).
    2.  **Word-Level**: Tokenisasi tingkat kata (bermakna semantis, vocab lebih besar).
    3.  **BPE-Like (Subword)**: *Byte Pair Encoding* mini untuk menangani *Out-Of-Vocabulary* (OOV) dan keseimbangan token.
* **Optimasi PyTorch**: Menggunakan `autocast` (*Mixed Precision*) dan `GradScaler` untuk mempercepat proses *training* menggunakan GPU CUDA.
* **Text Generation**: Menghasilkan teks autoregresif baru berdasarkan *prompt* yang diberikan dengan parameter *temperature* dan *top-k sampling*.

## 📂 Struktur Direktori

```text
📦 TinyGPT
 ┣ 📜 corpus.txt              # Dataset teks (Artikel Bitcoin & Blockchain)
 ┣ 📜 tokenizers.py           # Implementasi 3 jenis tokenizer (Char, Word, BPE)
 ┣ 📜 transformer_blocks.py   # Arsitektur inti TinyGPT (Attention & Decoder)
 ┣ 📜 tinygpt.py              # Script utama untuk pipeline (Dataset, Training, Evaluasi)
 ┗ 📜 README.md               # Dokumentasi proyek
