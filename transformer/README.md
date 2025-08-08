# Transformer

Decoder-only transformer implemented from scratch following the "Attention Is All You Need" paper.
The only difference from the paper is that in Add&Norm the LayerNorm is applied before the layer, which is proven to improve training performance.\
The transformer is trained to tell fairy tales, using around 60 fairy tales from the `tales.csv` file.\
CUDA is used to speed up the training.

The current implementation works character by character and the context length is 128 characters, so the generated output sounds like a fairy tale but the plot is not consistent and some sentences don't really make sense (which is normal with this kind of implementation).\
There is also not much variety in the tales because the dataset is small.\
Next step is using a tokenizer instead of working character by character.
