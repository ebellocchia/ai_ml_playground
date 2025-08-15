# Copyright (c) 2025 Emanuele Bellocchia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


#
# Models
#

class RNN(nn.Module):
    def __init__(self, vocab_size, rnn_class, emb_size=16, hidden_size=64, num_layers=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_size)
        self.hid = rnn_class(emb_size, hidden_size, num_layers=num_layers)
        self.ln = nn.LayerNorm(hidden_size)
        self.lin = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h=None):
        emb = self.emb(x)               # C, emb_size
        hid, h = self.hid(emb, h)       # C, hidden_size - num_layers, hidden_size
        norm = self.ln(hid)             # C, hidden_size
        lin = self.lin(norm)            # C, vocab_size
        return lin, h


class MLP(nn.Module):
    def __init__(self, vocab_size, emb_size=16, hidden_size=64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_size)
        self.hid = nn.Sequential(
            nn.Linear(emb_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.bn = nn.BatchNorm1d(hidden_size)
        self.lin = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h=None):
        emb = self.emb(x)
        hid = self.hid(emb)
        bn = self.bn(hid)
        return self.lin(bn), None


#
# Dataset
#

class WordConverter:
    def __init__(self, vocab):
        self.vocab_size = len(vocab)
        self.c2i = {ch:i for i, ch in enumerate(vocab)}
        self.i2c = {i:ch for ch, i in self.c2i.items()}

    def encode(self, s):
        return [self.c2i[ch] for ch in s]

    def encode_t(self, s):
        return torch.tensor(self.encode(s)).long()

    def decode(self, li):
        return "".join([self.i2c[i] for i in li])


class WordLoader:
    WORD_BEGIN_CHAR = "<"
    WORD_END_CHAR = ">"

    def __init__(self, file_name):
        self.file_name = file_name
        self.words = []

    def load(self):
        self.words = list(
            map(
                lambda w: self.WORD_BEGIN_CHAR + w.strip() + self.WORD_END_CHAR,
                open(self.file_name, "r").read().lower().split()
            )
        )

    def get_words(self, max_num=-1):
        if max_num > 0 and max_num < len(self.words):
            indices = torch.randperm(len(self.words))[:max_num]
            words = [self.words[i] for i in indices]
        else:
            words = self.words
        vocab = sorted(list(set("".join(words))))
        return words, vocab


class WordDataset(Dataset):
    def __init__(self, words, word_conv):
        self.words = words
        self.word_conv = word_conv

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        word = self.words[idx]
        x = self.word_conv.encode_t(word[:-1])
        y = self.word_conv.encode_t(word[1:])
        return x, y

    def collate_batch(self, batch):
        pad_value = self.word_conv.encode(WordLoader.WORD_END_CHAR)[0]

        xs, ys = zip(*batch)
        xs_pad = pad_sequence(xs, batch_first=False, padding_value=pad_value)
        ys_pad = pad_sequence(ys, batch_first=False, padding_value=pad_value)
        return xs_pad, ys_pad


#
# Model training
#

class Device:
    @staticmethod
    def get():
        return "cuda" if torch.cuda.is_available() else "cpu"


class ModelTrainer:
    EPOCH_PRINT = 10

    def __init__(self, model, words, vocab, lr):
        self.model = model
        self.lr_init = lr
        self.words = words
        self.word_conv = WordConverter(vocab)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=0.9,
            patience=self.EPOCH_PRINT / 2,
            min_lr=lr / 100
        )

    def train(self, batch_size, loss_target):
        print(f"""Training started:
 - Words: {len(self.words)}
 - Batch size: {batch_size}
 - Target loss: {loss_target}
 - Learning rate: {self.__get_lr()}
 - Model: {self.model.hid.__class__.__name__}
 - Criterion: {self.criterion.__class__.__name__}
 - Optimizer: {self.optimizer.__class__.__name__}
 - Device: {Device.get()}
""")
        self.model.train()

        dataset = WordDataset(self.words, self.word_conv)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            collate_fn=lambda batch: dataset.collate_batch(batch)
        )

        epoch_num = 1
        last_time = time.time()
        loss_mean_last = 0.0

        while True:
            loss_sum = 0.0
            num_batches = 0
            for x, y in loader:
                # x, y -> seq_len * batch_size
                x, y = x.to(Device.get()), y.to(Device.get())

                # out -> seq_len, batch_size, vocab_size
                out, _ = self.model(x)

                # out.view(...) -> seq_len * batch_size, vocab_size
                # y.view(...)   -> seq_len * batch_size
                loss = self.criterion(out.view(-1, self.model.lin.out_features), y.view(-1))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_sum += loss.item()
                num_batches += 1

            loss_mean = loss_sum / num_batches
            if loss_mean < loss_target:
                print(f"{epoch_num}. Target loss reached, stopped. Loss: {loss_mean:.6f}")
                break

            self.scheduler.step(loss_mean)

            if epoch_num % self.EPOCH_PRINT == 0:
                loss_mean_diff = loss_mean_last - loss_mean
                loss_mean_last = loss_mean

                curr_time = time.time()
                elapsed_time = curr_time - last_time
                last_time = curr_time

                print(f"{epoch_num}. Loss: {loss_mean:.6f}, diff: {loss_mean_diff:.6f}, lr: {self.__get_lr():.5f}, time: {elapsed_time:.3f}s")

            epoch_num += 1

    def __get_lr(self):
        return self.optimizer.param_groups[0]["lr"]


class WordGenerator:
    def __init__(self, model, vocab):
        self.model = model
        self.word_conv = WordConverter(vocab)

    @torch.no_grad()
    def generate_word(self, start=WordLoader.WORD_BEGIN_CHAR, temp=1.0, max_len=30):
        self.model.eval()

        h = None
        word = start
        for _ in range(max_len):
            curr_char = word[-1]
            curr_char_enc = self.word_conv.encode_t(curr_char).to(Device.get())

            out, h = self.model(curr_char_enc, h)
            out_probs = F.softmax(out.squeeze(0) / temp, dim=-1)

            next_char_enc = torch.multinomial(out_probs, num_samples=1).item()
            next_char = self.word_conv.decode([next_char_enc])

            if next_char == WordLoader.WORD_END_CHAR:
                break
            word += next_char
        return word



#
# Main
#

def main():
    BATCH_SIZE = 128
    LEARN_RATE_INIT = 1e-2
    LOSS_TARGET = 0.7
    GEN_NAMES_NUM = 30
    WORDS_NUM = -1

    torch.manual_seed(0)

    word_loader = WordLoader("names.txt")
    word_loader.load()
    words, vocab = word_loader.get_words(WORDS_NUM)

    model = RNN(len(vocab), nn.LSTM).to(Device.get())

    model_trainer = ModelTrainer(model, words, vocab, LEARN_RATE_INIT)
    model_trainer.train(BATCH_SIZE, LOSS_TARGET)

    word_gen = WordGenerator(model, vocab)

    print("Generated names:")
    for i in range(GEN_NAMES_NUM):
        gen_name = word_gen.generate_word()[1:]
        print(f" {i + 1}. {gen_name}")


if __name__ == "__main__":
    main()
