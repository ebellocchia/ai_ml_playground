# Transformer

Decoder-only transformer implemented from scratch following the "Attention Is All You Need" paper.
The only difference from the paper is that in `Add&Norm` the `LayerNorm` is applied before the layer, which is proven to improve training performance.\
The transformer can be trained to:
- tell fairy tales, using around 60 fairy tales from the `tales.csv` file
- generate a shakespeare-like writing, using the `shakespeare.txt` file

CUDA is used to speed up the training. Being a small model, I trained it on my laptop (NVIDIA RTX 4090 Mobile).

`nn_transformer_charbychar.py`: it works character-by-character (i.e. the model learns to predict the next character depending on the previous ones). The generated output sounds like a fairy tale or a shakespeare writing, but the sentences don't really make sense (which is normal with a character-by-character training).
`nn_transformer_tokenizer.py`: it uses a word-level tokenizer (i.e. the model learns to predict the next word depending on the previous ones). Since it learns word-by-word, the generated output is much better than the previous implementation. A word-level tokenizer is used because the vocabulary size is not very big, almost the same size of using a word-piece tokenizer. The raw HuggingFace tokenizer is used for learning purposes.
