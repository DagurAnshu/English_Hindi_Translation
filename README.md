# English_Hindi_Translation
Try to implement transformers learning LSTM on real world project Hindi_English Translation

A deep learning based Hindi--> English translation system using a Transformer model build from scratch

THis Project implements full pipeline implementation
  . Data loading from parallel corpus
  . Subword tokenization using SentencePiece
  . Transformer encoder-decoder architecture
  . Training loop with masking and padding

---
# Features:
  1. Hindi --> English Translation
  2. Transformer Architecture and implemetation
  3. SubWord Tokenization using SentencePiece
  4. Data Preprocessing pipeline , try to cover each aspect
---

# Project WorkFLow

1. Data Loading:
   . read data directly from .zip file
   . clean and filter the data and use sentencepiece
2. Tokenization:
   . Train a BPE tokenizer on the data
   . Restrict vocublary size to 32000, (as per my machine constraint)
3.Encoding:
  . <PAD>
  . <BOS>
  .<EOS>

  . Fixed length of embedding vector to 128

4. Transformer Model Architecture:
Implement like we implement a normal transformer
  . Embedding Layer
  . Positional Encoding
  . Transformer
    . Encoded Layers: 4
    . Decoded Layers: 4
    . Heads: 4
  . Fully connected Layer

5. Training:
   .Optimizer AdamW
   . Loss : CrossEntropyLoss
   . Batch size: 64

6. Other Details:
   . MAX_LEN: 128
   . VOCAB_SIZE: 32000
   . EPOCHS : 3

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
  . Implement transformer from scratch
  . Implement masking
  . Handle Padding efficiently in 
  . Understand NLP Pipeline effectively and completely

---

# Limitations:
  . No inference implemented yet
  . No Beam Search 
  . Not implement proper UI for it
  
  
