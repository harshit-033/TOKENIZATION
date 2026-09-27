# BPE and Data Loader

## Abstract

This repository implements the preprocessing pipeline used to prepare text for language model training. The work examines byte pair encoding (BPE) tokenization with the GPT-2 tokenizer and develops input-target sequences using a sliding-window approach. A PyTorch `Dataset` and `DataLoader` are then implemented to organize the tokenized corpus into training batches.

## I. Introduction

Tokenization is a fundamental stage in language model training because raw text must be converted into numerical representations before it can be processed by a neural network. This repository studies subword tokenization through GPT-2 BPE and demonstrates how tokenized text can be transformed into sequences suitable for next-token prediction.

The implementation is developed in a Jupyter Notebook and uses a text corpus stored in `verdict.txt`.

## II. Methodology

### A. BPE Tokenization

The repository uses the `tiktoken` library and the GPT-2 encoding:

```python
tokenizer = tiktoken.get_encoding("gpt2")
```

Text is converted into integer token IDs and decoded back into text to verify the tokenization process. A special end-of-text token is explicitly allowed during encoding.

The notebook also demonstrates the motivation for subword tokenization using an example containing an uncommon word. The example illustrates how subword-level encoding can represent text that may not be present as a complete word in a fixed vocabulary while retaining more linguistic structure than character-level tokenization.

### B. Input-Target Sequence Construction

After tokenization, the corpus is converted into training examples using a sliding-window strategy.

For a context length of (4), an input sequence is paired with the sequence shifted by one position:

```text
Input  : [t1, t2, t3, t4]
Target : [t2, t3, t4, t5]
```

The notebook also shows the corresponding progressive next-token examples, demonstrating how increasingly larger contexts can be used to predict the following token.

### C. PyTorch Dataset

A custom PyTorch `Dataset` named `tokenizerv1` is implemented. The dataset:

1. Encodes the complete text corpus using GPT-2 BPE.
2. Divides the token sequence into overlapping chunks.
3. Creates an input chunk and a one-position-shifted target chunk for each sample.
4. Stores the resulting sequences as PyTorch tensors.
5. Provides dataset length through `__len__`.
6. Provides individual input-target pairs through `__getitem__`.

The chunk length is controlled by `max_length`, while the overlap between consecutive samples is controlled by `stride`.

### D. PyTorch Data Loader

A helper function, `crearte_dataloader_v1`, constructs the complete data-loading pipeline. It initializes the GPT-2 tokenizer, creates the custom dataset, and wraps it with PyTorch's `DataLoader`.

The implementation exposes the following parameters:

- `batch_size`
- `max_length`
- `stride`
- `shuffle`
- `drop_last`
- `num_workers`

The notebook verifies the loader using small examples and prints batched input and target tensors.

## III. Implementation

The primary implementation is contained in:

```text
BPE_and_data_loader/
├── Byte_pair_encoding.ipynb
├── verdict.txt
└── word1.txt
```

### Files

| File | Description |
|---|---|
| `Byte_pair_encoding.ipynb` | Main notebook containing BPE experiments, input-target construction, dataset implementation, and data-loader verification. |
| `verdict.txt` | Text corpus used for tokenization and sequence generation. |
| `word1.txt` | Additional text resource included in the repository. |

## IV. Software and Libraries

- Python
- PyTorch
- tiktoken
- Jupyter Notebook

## V. Execution

Install the required libraries:

```bash
pip install torch tiktoken
```

Open the notebook:

```text
Byte_pair_encoding.ipynb
```

Run the cells sequentially to reproduce the tokenization, sequence construction, dataset creation, and data-loader examples.

## VI. Conclusion

This repository demonstrates the preprocessing stages required before training a language model: subword tokenization, numerical encoding, shifted input-target sequence generation, and batched data loading. The implementation provides a practical foundation for connecting raw text data to subsequent model-training components.

## Author

Harshit Kumar

GitHub: https://github.com/harshit-033
