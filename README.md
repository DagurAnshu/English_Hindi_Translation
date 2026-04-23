# English_Hindi_Translation
Try to implement transformers learning LSTM on real world project Hindi_English Translation

A deep learning based Hindi--> English translation system using a Transformer model build from scratch

THis Project implements full pipeline implementation
  1. Data loading from parallel corpus
  2. Subword tokenization using SentencePiece
  3. Transformer encoder-decoder architecture
  4. Training loop with masking and padding

---
# Features:
  1. Hindi --> English Translation
  2. Transformer Architecture and implemetation
  3. SubWord Tokenization using SentencePiece
  4. Data Preprocessing pipeline , try to cover each aspect
---

# Project WorkFLow

1. Data Loading:
   1. read data directly from .zip file
   2. clean and filter the data and use sentencepiece
2. Tokenization:
   1. Train a BPE tokenizer on the data
   1. Restrict vocublary size to 32000, (as per my machine constraint)
3.Encoding:
  1. <PAD>
  2. <BOS>
  3.<EOS>

  4. Fixed length of embedding vector to 128

4. Transformer Model Architecture:
Implement like we implement a normal transformer
  1. Embedding Layer
  2. Positional Encoding
  3. Transformer
    i. Encoded Layers: 4
    ii. Decoded Layers: 4
    iii. Heads: 4
  4. Fully connected Layer

5. Training:
   i.Optimizer AdamW
   ii. Loss : CrossEntropyLoss
   ii. Batch size: 64

6. Other Details:
   i. MAX_LEN: 128
   ii. VOCAB_SIZE: 32000
   iii. EPOCHS : 3

---
# STEPS TO RUN

1.Load Dataset from .zip properly
2. Train Tokenizer
3. Encode Dataset
4. Train Transformer Model

----
# DATASET:
  Get this data from IITB Hindi Parallel Corpus
  Take only 20,000 sentence pair (for fast running)

---
# KEY LEARNINGS:
  i. Implement transformer from scratch
  ii. Implement masking
  iii. Handle Padding efficiently in 
  iv. Understand NLP Pipeline effectively and completely

---

# Limitations:
  i. No inference implemented yet
  ii. No Beam Search 
  iii. Not implement proper UI for it
  
  
