# Transformer

Decoder-only transformer implemented from scratch following the "Attention Is All You Need" paper.
The only difference from the paper is that in `Add&Norm` the `LayerNorm` is applied before the layer, which is proven to improve training performance.\
The transformer can be trained to:
- tell fairy tales, using around 60 fairy tales from the `tales.csv` file
- generate a shakespeare-like writing, using the `shakespeare.txt` file

CUDA is used to speed up the training. Being a small model, I trained it on my laptop (NVIDIA RTX 4090 Mobile).

The current implementation works character-by-character and the context length is 128 characters, so the generated output sounds like a fairy tale or a shakespeare writing, but the plot is not consistent and sentences don't really make sense (which is normal with a character-by-character training).\
There is also not much variety because the dataset is small.\

Next step is using a tokenizer instead of working character by character.
