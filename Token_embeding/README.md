# Token Embedding

## Abstract

This repository studies word and token representations used in natural language processing and language models. The notebook first examines pretrained Word2Vec embeddings using the Google News model and explores vector similarity and vector arithmetic. It then introduces trainable token embeddings using PyTorch's `Embedding` layer and demonstrates how token IDs are mapped to dense vector representations.

## I. Introduction

Machine learning models cannot directly operate on textual words or token IDs. An embedding layer provides a numerical representation in which each discrete token is associated with a continuous vector.

This repository approaches the topic in two stages:

1. Analysis of pretrained word embeddings using Word2Vec.
2. Construction and inspection of trainable token embeddings using PyTorch.

The work is implemented in a single Jupyter Notebook, `token_embeding.ipynb`.

## II. Methodology

### A. Pretrained Word2Vec Embeddings

The notebook uses Gensim to load the pretrained Google News Word2Vec model:

```python
model = api.load("word2vec-google-news-300")
```

The loaded model provides 300-dimensional vectors for a vocabulary of approximately 3 million entries.

The notebook retrieves the vector representation of individual words and examines the semantic relationships encoded in the pretrained space.

### B. Word Similarity

Cosine-based similarity is examined using examples such as:

```text
men <-> women
men <-> car
men <-> boy
car <-> engine
```

This demonstrates that embeddings can represent different degrees of semantic association between words.

### C. Vector Arithmetic

The notebook also evaluates relationships through vector arithmetic. An analogy-style query is constructed using positive and negative word sets, for example:

```text
king + women - man
```

The pretrained model is queried for nearby vectors to observe the resulting semantic relationship.

The notebook additionally measures Euclidean magnitudes between selected word-vector differences, including:

- man and woman
- semiconductor and earthworm
- nephew and niece

These experiments provide a basic view of how semantic relationships can be expressed geometrically in an embedding space.

### D. Trainable Token Embeddings

The second part of the notebook introduces PyTorch's embedding layer:

```python
torch.nn.Embedding(vocab_size, output_dim)
```

A small vocabulary and embedding dimension are used to make the mapping explicit:

```python
vocab_size = 6
output_dim = 3
```

An input sequence of token IDs is represented as:

```text
[2, 3, 5, 1]
```

Each integer is used as an index into the embedding matrix, producing a corresponding dense vector.

The notebook also inspects the embedding matrix itself and demonstrates both single-token and multi-token lookups.

## III. Implementation

The repository contains the following file:

```text
Token_embeding/
└── token_embeding.ipynb
```

### Notebook Components

| Component | Description |
|---|---|
| Gensim Word2Vec | Loads and queries pretrained word embeddings. |
| Similarity Analysis | Computes semantic similarity between word vectors. |
| Vector Arithmetic | Explores relationships through positive and negative vector combinations. |
| PyTorch Embedding | Demonstrates a trainable mapping from token IDs to dense vectors. |

## IV. Software and Libraries

- Python
- Gensim
- NumPy
- PyTorch
- Jupyter Notebook

## V. Execution

Install the required libraries:

```bash
pip install gensim numpy torch jupyter
```

Open the notebook:

```text
token_embeding.ipynb
```

Run the cells sequentially to reproduce the pretrained embedding experiments and PyTorch token-embedding examples.

The pretrained Google News Word2Vec model is large and must be downloaded when first loaded through Gensim.

## VI. Conclusion

This repository provides a practical introduction to embedding representations by connecting pretrained word vectors with the mechanics of trainable token embeddings. The Word2Vec experiments demonstrate semantic relationships in a pretrained vector space, while the PyTorch implementation shows how discrete token IDs are converted into dense vectors that can be learned as part of a neural language model.

## Author

Harshit Kumar

GitHub: https://github.com/harshit-033
