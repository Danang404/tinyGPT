"""
tokenizers.py
=============
3 Pendekatan Tokenisasi untuk TinyGPT
Topik: Bitcoin & Blockchain
Proyek Data Mining | Universitas Amikom Yogyakarta

Pendekatan:
  1. CharTokenizer  — level karakter
  2. WordTokenizer  — level kata
  3. BPETokenizer   — Byte Pair Encoding (subword)
"""

import re
import collections
from typing import List, Tuple, Dict


# ═══════════════════════════════════════════════════════════════════
# TOKENIZER 1: CHARACTER-LEVEL
# ═══════════════════════════════════════════════════════════════════

class CharTokenizer:
    """
    Tokenisasi Level Karakter.

    Setiap karakter tunggal (huruf, angka, spasi, tanda baca)
    diperlakukan sebagai satu token.

    Kelebihan:
      - Vocab sangat kecil (biasanya < 100)
      - Tidak ada token <unk> untuk bahasa yang dikenal
      - Dapat merepresentasikan kata apa pun

    Kekurangan:
      - Urutan token sangat panjang
      - Model harus belajar struktur kata dari awal
      - Butuh konteks lebih besar

    Contoh:
      "bitcoin" → ['b','i','t','c','o','i','n'] → [5,12,28,6,18,12,17]
    """

    name = "Character-Level"

    def __init__(self):
        self.stoi: Dict[str, int] = {}   # string → index
        self.itos: Dict[int, str] = {}   # index → string

    def fit(self, text: str) -> "CharTokenizer":
        """Bangun vocab dari semua karakter unik dalam teks."""
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        return self

    def encode(self, text: str) -> List[int]:
        """Ubah teks menjadi list indeks token."""
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids: List[int]) -> str:
        """Ubah list indeks token kembali ke teks."""
        return "".join(self.itos.get(i, "?") for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def info(self) -> str:
        return (
            f"[{self.name}]\n"
            f"  Vocab size   : {self.vocab_size}\n"
            f"  Contoh vocab : {list(self.stoi.keys())[:20]}\n"
        )


# ═══════════════════════════════════════════════════════════════════
# TOKENIZER 2: WORD-LEVEL
# ═══════════════════════════════════════════════════════════════════

class WordTokenizer:
    """
    Tokenisasi Level Kata.

    Teks dipecah berdasarkan spasi dan tanda baca.
    Setiap kata unik menjadi satu token.

    Kelebihan:
      - Urutan token jauh lebih pendek dibanding char-level
      - Setiap token membawa makna semantik penuh
      - Mudah dipahami dan di-debug

    Kekurangan:
      - Vocab bisa sangat besar untuk corpus besar
      - Kata OOV (Out-Of-Vocabulary) menjadi <unk>
      - Tidak bisa menangani kata baru / typo

    Contoh:
      "Bitcoin adalah mata uang digital"
      → ['bitcoin','adalah','mata','uang','digital']
      → [23, 5, 87, 102, 44]
    """

    name       = "Word-Level"
    UNK_TOKEN  = "<unk>"
    PAD_TOKEN  = "<pad>"

    def __init__(self, min_freq: int = 1):
        """
        Args:
            min_freq: frekuensi minimum agar kata masuk vocab.
                      Kata di bawah threshold → <unk>
        """
        self.min_freq = min_freq
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}

    def _split(self, text: str) -> List[str]:
        """Pisahkan teks menjadi token kata + tanda baca."""
        return re.findall(r"[\w'-]+|[^\w\s]", text.lower())

    def fit(self, text: str) -> "WordTokenizer":
        """Bangun vocab dari semua kata dengan frekuensi ≥ min_freq."""
        tokens = self._split(text)
        freq   = collections.Counter(tokens)
        # Special tokens di awal
        vocab  = [self.UNK_TOKEN, self.PAD_TOKEN]
        # Tambahkan kata berurutan dari yang paling sering
        vocab += [w for w, c in freq.most_common() if c >= self.min_freq]
        self.stoi = {w: i for i, w in enumerate(vocab)}
        self.itos = {i: w for w, i in self.stoi.items()}
        return self

    def encode(self, text: str) -> List[int]:
        """Ubah teks menjadi list indeks token."""
        unk_id = self.stoi[self.UNK_TOKEN]
        return [self.stoi.get(t, unk_id) for t in self._split(text)]

    def decode(self, ids: List[int]) -> str:
        """Ubah list indeks token kembali ke teks."""
        words = [self.itos.get(i, self.UNK_TOKEN) for i in ids]
        # Rekonstruksi dengan spasi, kecuali tanda baca
        out = ""
        for w in words:
            if not out or w in {".", ",", ";", ":", "!", "?"}:
                out += w
            else:
                out += " " + w
        return out

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def info(self) -> str:
        top10 = list(self.stoi.keys())[2:12]  # skip special tokens
        return (
            f"[{self.name}]\n"
            f"  Vocab size   : {self.vocab_size}\n"
            f"  Min freq     : {self.min_freq}\n"
            f"  Top-10 kata  : {top10}\n"
        )


# ═══════════════════════════════════════════════════════════════════
# TOKENIZER 3: BPE-LIKE (BYTE PAIR ENCODING)
# ═══════════════════════════════════════════════════════════════════

class BPETokenizer:
    """
    Tokenisasi Subword dengan Byte Pair Encoding (BPE).

    Algoritma:
      1. Mulai dari representasi karakter tiap kata
      2. Hitung semua pasangan simbol yang bersebelahan
      3. Gabungkan pasangan yang paling sering menjadi simbol baru
      4. Ulangi sebanyak `num_merges` kali
      5. Vocab akhir = semua subword yang terbentuk

    Kelebihan:
      - Keseimbangan antara char-level dan word-level
      - Menangani kata OOV dengan memecahnya ke subword
      - Vocab size dapat dikontrol via num_merges
      - Digunakan oleh GPT-2, RoBERTa, dll.

    Kekurangan:
      - Proses fit lebih lambat
      - Decode lebih kompleks
      - Hasil tidak selalu intuitif

    Contoh (num_merges=3):
      "bitcoin" → karakter: b i t c o i n </w>
               → setelah merge: bi tc oi n</w>
               → ids: [45, 23, 67, 89]
    """

    name = "BPE-Like (Subword)"

    def __init__(self, num_merges: int = 200):
        """
        Args:
            num_merges: jumlah iterasi penggabungan pasangan.
                        Lebih besar = vocab lebih besar, subword lebih panjang.
        """
        self.num_merges = num_merges
        self.merges: List[Tuple[str, str]] = []  # aturan merge yang dipelajari
        self.vocab: Dict[str, int] = {}
        self.itos:  Dict[int, str] = {}

    # ── Internal helpers ──

    def _get_pair_freqs(self, word_freqs: Dict[str, int]) -> Dict[Tuple, int]:
        """Hitung frekuensi setiap pasangan simbol yang bersebelahan."""
        pairs: Dict[Tuple, int] = collections.defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _apply_merge(self, word_freqs: Dict[str, int],
                     pair: Tuple[str, str]) -> Dict[str, int]:
        """Terapkan satu aturan merge ke semua kata dalam word_freqs."""
        new_freqs: Dict[str, int] = {}
        bigram  = re.escape(" ".join(pair))
        pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        for word, freq in word_freqs.items():
            new_word = pattern.sub("".join(pair), word)
            new_freqs[new_word] = new_freqs.get(new_word, 0) + freq
        return new_freqs

    # ── Public API ──

    def fit(self, text: str) -> "BPETokenizer":
        """Pelajari aturan BPE dari corpus."""
        # Langkah 1: representasi awal — setiap kata = urutan karakter + </w>
        raw_words = re.findall(r"[\w'-]+|[^\w\s]", text.lower())
        word_freqs: Dict[str, int] = collections.defaultdict(int)
        for w in raw_words:
            word_freqs[" ".join(list(w)) + " </w>"] += 1

        # Langkah 2: iterasi merge
        for merge_idx in range(self.num_merges):
            pairs = self._get_pair_freqs(word_freqs)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            self.merges.append(best_pair)
            word_freqs = self._apply_merge(word_freqs, best_pair)

        # Langkah 3: bangun vocab dari semua subword unik
        all_symbols: set = set()
        for word in word_freqs:
            all_symbols.update(word.split())
        sorted_symbols = ["<unk>", "<pad>"] + sorted(all_symbols)
        self.vocab = {sym: i for i, sym in enumerate(sorted_symbols)}
        self.itos  = {i: sym for sym, i in self.vocab.items()}
        return self

    def _apply_merges_to_word(self, word: str) -> List[str]:
        """Terapkan semua aturan merge yang dipelajari ke satu kata."""
        symbols = list(word) + ["</w>"]
        for pair in self.merges:
            new_syms = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                    new_syms.append(symbols[i] + symbols[i + 1])
                    i += 2
                else:
                    new_syms.append(symbols[i])
                    i += 1
            symbols = new_syms
        return symbols

    def encode(self, text: str) -> List[int]:
        """Ubah teks menjadi list indeks subword token."""
        unk_id = self.vocab.get("<unk>", 0)
        words  = re.findall(r"[\w'-]+|[^\w\s]", text.lower())
        ids: List[int] = []
        for w in words:
            for sym in self._apply_merges_to_word(w):
                ids.append(self.vocab.get(sym, unk_id))
        return ids

    def decode(self, ids: List[int]) -> str:
        """Ubah list indeks subword kembali ke teks."""
        tokens = [self.itos.get(i, "<unk>") for i in ids]
        # Gabungkan subword: hapus marker </w> dan sambungkan
        text   = " ".join(tokens)
        text   = text.replace(" </w>", " ").replace("</w>", " ")
        return text.strip()

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def info(self) -> str:
        top_merges = self.merges[:10]
        return (
            f"[{self.name}]\n"
            f"  Vocab size   : {self.vocab_size}\n"
            f"  Num merges   : {self.num_merges} (actual: {len(self.merges)})\n"
            f"  Top-10 merge : {top_merges}\n"
        )


# ═══════════════════════════════════════════════════════════════════
# DEMO / TEST (jalankan file ini langsung)
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    sample = (
        "Bitcoin adalah mata uang digital pertama yang menggunakan teknologi "
        "blockchain. Blockchain menyimpan semua transaksi secara terdesentralisasi. "
        "Smart contract berjalan otomatis di atas jaringan Ethereum. "
        "Proof of Work memerlukan daya komputasi tinggi untuk memvalidasi blok baru."
    )

    print("=" * 60)
    print("  DEMO TOKENIZER — Topik: Bitcoin & Blockchain")
    print("=" * 60)

    tokenizers = [
        CharTokenizer(),
        WordTokenizer(min_freq=1),
        BPETokenizer(num_merges=50),
    ]

    for tok in tokenizers:
        tok.fit(sample)
        ids     = tok.encode(sample)
        decoded = tok.decode(ids)

        print(f"\n{tok.info()}")
        print(f"  Jumlah token : {len(ids)}")
        print(f"  Ratio char/token: {len(sample)/max(len(ids),1):.2f}")
        print(f"  Contoh encode: {ids[:15]} ...")
        print(f"  Decode (50 char): {decoded[:80]}")
        print(f"  Match original  : {decoded[:50] == sample[:50].lower()}")
        print("-" * 60)
