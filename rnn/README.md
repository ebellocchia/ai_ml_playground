# Recurrent Neural Network

Recurrent neural network that generates fantasy-like names. RNN, GRU or LSTM can be used.\
The network is trained using around 3000 fantasy names from the `names.txt` file.

`nn_words.py`: it trains the network name-by-name using the CPU.\
`nn_words_dataset.py`: it trains the network using a random batch of names and uses CUDA to speed up the training.
