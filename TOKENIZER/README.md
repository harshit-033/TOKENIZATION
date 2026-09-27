# Basic Tokenizer

A simple Python tokenizer built to understand the fundamentals of text tokenization and vocabulary based text encoding.

## Overview

This repository develops a basic tokenizer in two stages.

The first version focuses on splitting raw text into tokens using regular expressions.

The second version extends the tokenizer with vocabulary construction, token to ID conversion, unknown token handling, and decoding.

The project is intended for learning how tokenization works before moving to more advanced methods used in language models.

## Repository Structure

| File | Description |
|---|---|
| TOKENIZER.ipynb | First tokenizer implementation using regular expressions. |
| TOKENIZER_V2.ipynb | Extended tokenizer with vocabulary, encoding, decoding, and unknown token handling. |
| verdict.txt | Text corpus used for the initial tokenizer. |
| words.txt | Large word list used to build the tokenizer vocabulary. |

## Version 1

TOKENIZER.ipynb demonstrates the basic tokenization process.

The implementation:

1. Loads text from verdict.txt.
2. Uses Python regular expressions to split text.
3. Separates selected punctuation marks from words.
4. Removes empty and whitespace-only elements.
5. Produces a sequence of tokens.

The main tokenization pattern handles whitespace and punctuation such as commas, periods, colons, semicolons, question marks, exclamation marks, quotation marks, parentheses, apostrophes, and double hyphens.

Example:

```text
I had, always thought Jack Gisburn.
```

becomes:

```text
I
had
,
always
thought
Jack
Gisburn
.
```

## Version 2

TOKENIZER_V2.ipynb builds on the first implementation and introduces a vocabulary based tokenizer.

The workflow is:

```text
Text Corpus
    |
Tokenization
    |
Unique Tokens
    |
Sorted Vocabulary
    |
Token to ID Mapping
    |
Encoder and Decoder
```

The vocabulary is created from the unique tokens in words.txt. Two special tokens are then added:

```text
|<endofline>|
|<unk>|
```

The tokenizer maintains two mappings:

```text
str_to_int
int_to_str
```

This allows text to be converted into integer IDs and integer IDs to be converted back into text.

### Encoding

The encoder tokenizes input text and converts every token into its corresponding vocabulary ID.

Tokens that are not present in the vocabulary are replaced with:

```text
|<unk>|
```

This provides basic unknown token handling.

### Decoding

The decoder converts token IDs back into their corresponding strings and performs basic punctuation spacing cleanup.

## Example

For an input such as:

```text
hello how are you?my name is harshit
```

the encoder produces a sequence of integer IDs based on the vocabulary.

If a word or token is not present in the vocabulary, the corresponding ID is the ID assigned to `|<unk>|`.

The decoder can then convert the IDs back into text.

## Requirements

Python 3.x

Jupyter Notebook or JupyterLab

The implementation uses the Python standard library, primarily the `re` module.

No external package is required for the tokenizer itself.

## Usage

Clone the repository:

```bash
git clone https://github.com/harshit-033/basic_tokenizer.git
cd basic_tokenizer
```

Open the notebooks:

```bash
jupyter notebook TOKENIZER.ipynb
```

or

```bash
jupyter notebook TOKENIZER_V2.ipynb
```

Run the cells in order to reproduce the tokenization, vocabulary construction, encoding, and decoding steps.

## Current Scope

This project currently provides:

- Rule based tokenization
- Regular expression based token splitting
- Vocabulary construction
- Token to integer mapping
- Integer to token mapping
- Unknown token handling
- Basic encoding
- Basic decoding
- Punctuation spacing cleanup during decoding

## Limitations

This is an educational tokenizer and is not intended to replace production tokenization libraries.

It does not currently implement:

- Byte Pair Encoding
- WordPiece
- Unigram tokenization
- Subword vocabulary training
- Unicode normalization
- Efficient vocabulary serialization
- A standalone reusable Python package

The current implementation also uses a large word list as the vocabulary source rather than learning a vocabulary from a training corpus.

## Learning Progression

The repository is structured as a progression:

```text
Raw Text
    |
Basic Tokenization
    |
Vocabulary
    |
Token IDs
    |
Encoding
    |
Decoding
    |
Unknown Token Handling
```

This provides a foundation for understanding the preprocessing pipeline used by language models.

## Author

Harshit Kumar

Repository: https://github.com/harshit-033/basic_tokenizer
