# TOKENIZATION

## Abstract

This repository presents a practical study of the preprocessing stages used in Natural Language Processing (NLP) and language-model pipelines. The work progresses from rule-based tokenization to subword tokenization with GPT-2 Byte-Pair Encoding (BPE), followed by input-target sequence construction, pretrained word embeddings, trainable token embeddings, and positional embeddings.

The repository is organized into three related modules: `TOKENIZER`, `TOKENIZER_BPE`, and `Token_embeding`. Each module is implemented primarily through Jupyter notebooks and uses text data and standard NLP/deep-learning libraries to demonstrate the underlying concepts.

**Keywords—** Natural Language Processing, Tokenization, Byte-Pair Encoding, BPE, GPT-2, Word2Vec, Token Embedding, Positional Embedding, PyTorch, Language Models.

## I. Introduction

A language model does not process raw text directly. Text must first be converted into discrete token IDs and then into numerical representations that can be processed by a neural network.

This repository studies that progression in a practical manner:

```text
Raw Text
   │
   ▼
Rule-Based Tokenization
   │
   ▼
Vocabulary and Token IDs
   │
   ▼
Subword Tokenization (GPT-2 BPE)
   │
   ▼
Input-Target Sequence Construction
   │
   ▼
Token Embeddings
   │
   ▼
Positional Embeddings
   │
   ▼
Model-Ready Representations
```

The modules are related but independently executable. Together, they form a learning-oriented implementation of the preprocessing concepts required before constructing a transformer-based language model.

## II. Repository Structure

```text
TOKENIZATION/
├── README.md
│
├── TOKENIZER/
│   ├── README.md
│   ├── TOKENIZER.ipynb
│   ├── TOKENIZER_V2.ipynb
│   ├── verdict.txt
│   ├── word1.txt
│   └── words.txt
│
├── TOKENIZER_BPE/
│   ├── README.md
│   ├── Byte_pair_encoding.ipynb
│   ├── verdict.txt
│   └── word1.txt
│
└── Token_embeding/
    ├── README.md
    ├── token_embeding.ipynb
    ├── positional_embeding.ipynb
    └── verdict.txt
```

The repository currently stores the modules as normal directories rather than Git submodules.

## III. TOKENIZER Module

The `TOKENIZER` module introduces basic text tokenization and then extends it into a vocabulary-based encoder-decoder.

### A. Basic Tokenization

`TOKENIZER.ipynb` reads the text corpus from `verdict.txt` and uses Python regular expressions to split text around whitespace and selected punctuation.

The notebook progressively develops the regular expression used for tokenization and removes empty elements after splitting.

The implemented pattern handles characters including:

```text
, . : ; ? _ ! " ( ) ' -- whitespace
```

### B. Vocabulary Construction

`TOKENIZER_V2.ipynb` extends the basic tokenizer into a vocabulary-based tokenizer.

The implementation:

1. Loads text from `word1.txt`.
2. Tokenizes the corpus using regular-expression rules.
3. Removes empty tokens.
4. Builds a unique, sorted token list.
5. Adds `|<endofline>|` and `|<unk>|` special tokens.
6. Creates a token-to-integer mapping.

The mappings are maintained as:

```python
str_to_int
int_to_str
```

### C. Encoding and Decoding

The tokenizer class provides:

```python
encoder(text)
decoder(ids)
```

The encoder converts text into token IDs. Tokens not present in the vocabulary are replaced with `|<unk>|`.

The decoder maps IDs back to tokens and performs basic whitespace cleanup around punctuation.

## IV. TOKENIZER_BPE Module

The `TOKENIZER_BPE` module studies subword tokenization and preparation of language-model training data.

The implementation is contained primarily in `Byte_pair_encoding.ipynb`.

### A. GPT-2 BPE Tokenization

The notebook uses the `tiktoken` package with the GPT-2 encoding:

```python
tokenizer = tiktoken.get_encoding("gpt2")
```

Text is converted into integer token IDs and then decoded back into text to verify the representation. The notebook also explicitly allows the GPT-2 end-of-text token `<|endoftext|>` during encoding.

An example containing an uncommon word is used to illustrate the motivation for subword tokenization.

### B. Input-Target Pair Construction

After tokenization, the notebook creates training examples using a sliding-window method.

For context length 4:

```text
Input  : [t1, t2, t3, t4]
Target : [t2, t3, t4, t5]
```

The target sequence is shifted by one token and represents the next-token prediction objective used in autoregressive language-model training.

The notebook also demonstrates overlapping windows by varying the stride.

### C. PyTorch Dataset

A custom dataset named `tokenizerv1` is implemented using PyTorch's `Dataset` abstraction.

Its main operations are:

- Encode the corpus with the GPT-2 tokenizer.
- Extract fixed-length token chunks.
- Create one-position-shifted target chunks.
- Convert chunks into PyTorch tensors.
- Provide `__len__` and `__getitem__`.

The dataset is parameterized by `max_length` and `stride`.

### D. PyTorch DataLoader

The helper function `crearte_dataloader_v1` wraps the dataset with PyTorch's `DataLoader`.

The demonstrated parameters include:

```text
batch_size
max_length
stride
shuffle
drop_last
num_workers
```

The notebook verifies the resulting input and target batches using tensor examples.

## V. Token Embedding Module

The `Token_embeding` module studies pretrained word vectors, trainable token embeddings, and positional embeddings.

### A. Pretrained Word2Vec

`token_embeding.ipynb` uses Gensim to load the pretrained Google News Word2Vec model:

```python
model = api.load("word2vec-google-news-300")
```

The notebook works with 300-dimensional word vectors and examines the learned vector space.

The experiments include retrieving individual word vectors, finding similar words, measuring word similarity, performing vector arithmetic, and measuring vector-difference magnitudes.

Examples include relationships involving `person`, `men`, `women`, `boy`, `car`, `engine`, `king`, and `queen`.

### B. Trainable Token Embeddings

The notebook introduces PyTorch's embedding layer:

```python
torch.nn.Embedding(vocab_size, output_dim)
```

A small example uses:

```text
vocab_size = 6
output_dim = 3
input_ids = [2, 3, 5, 1]
```

Each token ID indexes the embedding matrix and produces a dense vector. The notebook verifies both individual and multiple-token lookups.

### C. Positional Embeddings

`positional_embeding.ipynb` extends token representations by introducing positional information.

The notebook references GPT-2's vocabulary size of 50,257 and uses `tiktoken` for GPT-2 tokenization. It defines a PyTorch `Dataset` and `DataLoader` to create fixed-length input-target sequences.

For the demonstrated configuration:

```text
context = 4
output_dim = 256
batch_size = 8
```

the token embedding layer produces:

```text
(8, 4, 256)
```

A separate positional embedding layer is then created using:

```python
pos_embedding_layer = torch.nn.Embedding(context, output_dim)
```

The positional vectors are intended to be combined with token embeddings so that the resulting representation contains both token identity and position information.

## VI. Software and Libraries

| Component | Purpose |
|---|---|
| Python | Core implementation language |
| Jupyter Notebook | Interactive development and experimentation |
| `re` | Rule-based tokenization |
| PyTorch | Dataset, DataLoader, and trainable embeddings |
| `tiktoken` | GPT-2 BPE tokenization |
| Gensim | Pretrained Word2Vec loading and analysis |
| NumPy | Vector operations |

## VII. Execution

### A. Basic Tokenizer

Open:

```text
TOKENIZER/TOKENIZER.ipynb
TOKENIZER/TOKENIZER_V2.ipynb
```

The notebooks require the accompanying text resources such as `verdict.txt` and `word1.txt`.

### B. BPE and Data Loading

Install the required packages:

```bash
pip install torch tiktoken
```

Open:

```text
TOKENIZER_BPE/Byte_pair_encoding.ipynb
```

Run the notebook cells sequentially.

### C. Token and Positional Embeddings

Install the required libraries:

```bash
pip install torch gensim numpy tiktoken
```

Open:

```text
Token_embeding/token_embeding.ipynb
Token_embeding/positional_embeding.ipynb
```

The Word2Vec notebook downloads the pretrained Google News model through Gensim. The model is large and requires substantial storage for the initial download.

## VIII. Complete Learning Pipeline

```text
Stage 1
Rule-Based Tokenization
        │
        ▼
Stage 2
Vocabulary + Token IDs
        │
        ▼
Stage 3
GPT-2 BPE Tokenization
        │
        ▼
Stage 4
Sliding-Window Input/Target Pairs
        │
        ▼
Stage 5
PyTorch Token Embeddings
        │
        ▼
Stage 6
Positional Embeddings
        │
        ▼
Transformer-Ready Input
```

## IX. Technical Observations

The repository demonstrates the distinction between the major representation levels used in an NLP pipeline.

### A. Word-Level Tokenization

The basic tokenizer separates text according to whitespace and manually selected punctuation rules and relies on a fixed vocabulary.

### B. Subword Tokenization

The GPT-2 tokenizer represents text with subword units, reducing the dependence on complete-word vocabulary entries.

### C. Token IDs

Token IDs are discrete integer indices and are not themselves continuous semantic representations.

### D. Token Embeddings

PyTorch's `Embedding` layer maps discrete token IDs to dense vectors.

### E. Positional Embeddings

Positional embeddings provide information about token order independently of token identity.

## X. Limitations

The repository is primarily educational and experimental.

The current work is notebook-based rather than packaged as a reusable Python library. The basic tokenizer uses manually defined regular-expression rules and a fixed vocabulary source. The BPE module uses GPT-2 BPE through `tiktoken` rather than implementing BPE training from scratch. The embedding notebooks demonstrate representation concepts but do not implement or train a complete language model.

The notebooks also contain exploratory cells and recorded outputs, so they should be treated as learning material rather than production-ready components.

## XI. Conclusion

TOKENIZATION provides a practical progression through the major preprocessing stages required for language-model development. It begins with rule-based tokenization and vocabulary mapping, moves to GPT-2 BPE subword tokenization, constructs next-token prediction samples using sliding windows, and then studies token, word, and positional embeddings.

The project therefore establishes the preprocessing and representation layer that precedes the construction of transformer architectures.

## References

[1] T. Sennrich, B. Haddow, and A. Birch, “Neural Machine Translation of Rare Words with Subword Units,” *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics*, 2016.

[2] OpenAI, “tiktoken,” GitHub repository. Available: https://github.com/openai/tiktoken

[3] Gensim, “gensim,” documentation and library resources. Available: https://radimrehurek.com/gensim/

[4] PyTorch, “Embedding,” PyTorch documentation. Available: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html

## Author

**Harshit Kumar**

GitHub: https://github.com/harshit-033