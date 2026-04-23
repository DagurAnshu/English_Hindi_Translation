import zipfile
import random
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import sentencepiece as spm
import math
import os

# ======================
# CONFIG
# ======================

ZIP_PATH = Path(r"C:/Users/dagur/Downloads/parallel.zip")
HINDI_FILE = "parallel-n/IITB.en-hi.hi"
ENGLISH_FILE = "parallel-n/IITB.en-hi.en"

TOKENIZER_PREFIX = "joint"
VOCAB_SIZE = 32000
MAX_LEN = 128
BATCH_SIZE = 64
EPOCHS = 3
RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# LOAD DATA FROM ZIP
# ======================

def load_parallel_data(zip_path):
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as z:

        zip_contents = z.namelist()

        if HINDI_FILE not in zip_contents:
            raise FileNotFoundError(f"{HINDI_FILE} not found inside zip")

        if ENGLISH_FILE not in zip_contents:
            raise FileNotFoundError(f"{ENGLISH_FILE} not found inside zip")

        with z.open(HINDI_FILE) as f:
            hindi = f.read().decode("utf-8", errors="replace").splitlines()

        with z.open(ENGLISH_FILE) as f:
            english = f.read().decode("utf-8", errors="replace").splitlines()

    cleaned_pairs = []
    for en, hi in zip(english, hindi):
        en = en.strip()
        hi = hi.strip()

        if en and hi:
            if len(en.split()) < 100 and len(hi.split()) < 100:
                cleaned_pairs.append((hi, en))  # (hi, en) for translation

    print("Original English lines :", len(english))
    print("Original Hindi lines   :", len(hindi))
    print("Cleaned aligned pairs  :", len(cleaned_pairs))

    return cleaned_pairs

# ======================
# TRAIN TOKENIZER
# ======================

def train_tokenizer(data):
    if os.path.exists(f"{TOKENIZER_PREFIX}.model"):
        print("Tokenizer already exists.")
        return

    print("Training tokenizer...")

    with open("combined.txt", "w", encoding="utf-8") as f:
        for hi, en in data:
            f.write(hi + "\n")
            f.write(en + "\n")

    spm.SentencePieceTrainer.train(
        input="combined.txt",
        model_prefix=TOKENIZER_PREFIX,
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3
    )

# ======================
# ENCODE DATA
# ======================

def encode_data(data, sp):
    PAD = sp.pad_id()
    BOS = sp.bos_id()
    EOS = sp.eos_id()

    def encode_pair(hi, en):
        hi_ids = [BOS] + sp.encode(hi, out_type=int) + [EOS]
        en_ids = sp.encode(en, out_type=int)

        decoder_input = [BOS] + en_ids
        decoder_target = en_ids + [EOS]

        hi_ids = hi_ids[:MAX_LEN]
        decoder_input = decoder_input[:MAX_LEN]
        decoder_target = decoder_target[:MAX_LEN]

        hi_ids += [PAD] * (MAX_LEN - len(hi_ids))
        decoder_input += [PAD] * (MAX_LEN - len(decoder_input))
        decoder_target += [PAD] * (MAX_LEN - len(decoder_target))

        return hi_ids, decoder_input, decoder_target

    encoded = [encode_pair(hi, en) for hi, en in data]

    encoder_data = torch.tensor([x[0] for x in encoded])
    decoder_input_data = torch.tensor([x[1] for x in encoded])
    decoder_target_data = torch.tensor([x[2] for x in encoded])

    return encoder_data, decoder_input_data, decoder_target_data

# ======================
# MODEL
# ======================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class TranslationTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, pad_id):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.size(1)).to(DEVICE)

        src_key_padding_mask = (src == pad_id)
        tgt_key_padding_mask = (tgt == pad_id)

        src = self.pos_encoder(self.embedding(src))
        tgt = self.pos_encoder(self.embedding(tgt))

        output = self.transformer(
            src, tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        return self.fc_out(output)

# ======================
# TRAIN LOOP
# ======================

def train_model(model, encoder_data, decoder_input_data, decoder_target_data, pad_id):
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    model.to(DEVICE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for i in range(0, len(encoder_data), BATCH_SIZE):
            src = encoder_data[i:i+BATCH_SIZE].to(DEVICE)
            tgt_in = decoder_input_data[i:i+BATCH_SIZE].to(DEVICE)
            tgt_out = decoder_target_data[i:i+BATCH_SIZE].to(DEVICE)

            optimizer.zero_grad()

            output = model(src, tgt_in, pad_id)
            loss = criterion(output.view(-1, VOCAB_SIZE),
                             tgt_out.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            print("in step i ", i, total_loss)

        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

# ======================
# MAIN
# ======================

def main():
    random.seed(RANDOM_SEED)

    print("Loading data from zip...")
    pairs = load_parallel_data(ZIP_PATH)

    random.shuffle(pairs)

    pairs = pairs[:20000]  # debug subset first

    train_tokenizer(pairs)

    sp = spm.SentencePieceProcessor()
    sp.load(f"{TOKENIZER_PREFIX}.model")

    encoder_data, decoder_input_data, decoder_target_data = encode_data(pairs, sp)

    model = TranslationTransformer(sp.get_piece_size())

    train_model(model, encoder_data,
                decoder_input_data,
                decoder_target_data,
                sp.pad_id())

if __name__ == "__main__":
    main()