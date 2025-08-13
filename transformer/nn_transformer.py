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

import math
import time
from abc import ABC, abstractmethod
from enum import auto, Enum

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


#
# Transformer
#

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, v, k, q, mask=None):
        d_k = q.shape[-1]

        # q -> batch_size, n_head, seq_len, d_k
        # k -> batch_size, n_head, seq_len, d_k -> batch_size, n_head, d_k, seq_len
        # att_scores -> batch_size, n_head, seq_len, seq_len
        att_scores = (q @ k.transpose(-1, -2)) / math.sqrt(d_k)

        if mask is not None:
            att_scores = att_scores.masked_fill(mask == 0, float("-inf"))
        att_scores = F.softmax(att_scores, dim=-1)
        att_scores = self.dropout(att_scores)

        # att_scores -> batch_size, n_head, seq_len, seq_len
        # v -> batch_size, n_head, seq_len, d_k
        # result -> batch_size, n_head, seq_len, d_k
        return att_scores @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head

        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.attention = ScaledDotProductAttention(dropout)

    def forward(self, v, k, q, mask=None):
        # v, k, q -> batch_size, seq_len, d_model
        v = self.w_v(v)
        k = self.w_k(k)
        q = self.w_q(q)

        # batch_size, n_head, seq_len, d_k
        att = self.attention(
            self.__split_head(v),
            self.__split_head(k),
            self.__split_head(q),
            mask
        )

        return self.w_o(self.__concat(att))

    def __split_head(self, t):
        batch_size, seq_len, d_model = t.shape

        # Do not split seq_len!
        # 1. batch_size, seq_len, d_model -> batch_size, seq_len, n_head, d_k
        # 2. batch_size, seq_len, n_head, d_k -> batch_size, n_head, seq_len, d_k
        return t.view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)

    def __concat(self, t):
        batch_size, n_head, seq_len, d_k = t.shape

        # 1. batch_size, n_head, seq_len, d_k -> batch_size, seq_len, n_head, d_k
        # 2. batch_size, seq_len, n_head, d_k -> batch_size, seq_len, d_model
        return t.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)


class AddAndNorm(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x, layer):
        return x + self.dropout(layer(self.ln(x)))


class FeedForward(nn.Sequential):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )


class InputEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # batch_size, seq_len, d_model
        return self.embedding(x) * math.sqrt(self.d_model)


class Projection(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.lin = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # batch_size, seq_len, vocab_size
        return self.lin(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_max_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # seq_len, d_model
        pos_enc = torch.zeros(seq_max_len, d_model)
        # seq_len
        pos = torch.arange(0, seq_max_len).float().unsqueeze(1)
        # d_model / 2
        div_term  = 10000 ** (torch.arange(0, d_model, 2).float() / d_model)
        # Alternative: div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        # seq_len, d_model
        pos_enc[:, 0::2] = torch.sin(pos / div_term)
        pos_enc[:, 1::2] = torch.cos(pos / div_term)
        # For batch_size in first position
        pos_enc = pos_enc.unsqueeze(0)

        self.register_buffer("pos_enc", pos_enc)

    def forward(self, x):
        # x -> batch_size, seq_len, d_model
        batch_size, seq_len, d_model = x.shape

        x = x + self.pos_enc[:, :seq_len, :]
        return self.dropout(x)


class DecoderOnlyLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_head, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.add_norm = nn.ModuleList([
            AddAndNorm(d_model, dropout),
            AddAndNorm(d_model, dropout),
        ])

    def forward(self, x, mask):
        out = self.add_norm[0](x, lambda x: self.self_attention(x, x, x, mask))
        return self.add_norm[1](out, self.feed_forward)


class DecoderOnly(nn.Module):
    def __init__(self, seq_max_len, n_layers, d_model, n_head, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderOnlyLayer(d_model, n_head, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.register_buffer("casual_mask", torch.tril(torch.ones(seq_max_len, seq_max_len)))

    def forward(self, x):
        casual_mask = self.__casual_mask(x)
        for layer in self.layers:
          x = layer(x, casual_mask)
        return self.norm(x)

    def __casual_mask(self, x):
        batch_size, seq_len, d_model = x.shape
        return self.casual_mask[:seq_len, :seq_len]


class TransformerDecoderOnly(nn.Module):
    def __init__(self, vocab_size, seq_max_len, n_layers=6, d_model=512, n_head=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.input_emb = InputEmbedding(d_model, vocab_size)
        self.pos_enc = PositionalEncoding(d_model, seq_max_len, dropout)
        self.decoder = DecoderOnly(seq_max_len, n_layers, d_model, n_head, d_ff, dropout)
        self.proj = Projection(d_model, vocab_size)

    def forward(self, x):
        emb = self.input_emb(x)
        pe = self.pos_enc(emb)
        dec = self.decoder(pe)
        return self.proj(dec)


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


class TrainingDataLoader(ABC):
    BEGIN_TAG = "[START]"
    END_TAG = "[END]"

    def __init__(self, file_name):
        self.file_name = file_name

    def _print_info(self, text, vocab):
        print(f"""Training data successfully loaded from file: {self.file_name}
- Total length: {len(text)}
- Vocabulary length: {len(vocab)}
""")

    @abstractmethod
    def load(self):
        pass


class ShakespeareLoader(TrainingDataLoader):
    def load(self):
        with open(self.file_name, "r") as f:
            text = self.BEGIN_TAG + f.read() + self.END_TAG
        vocab = sorted(list(set(text)))

        self._print_info(text, vocab)

        return text, vocab


class FairyTalesLoader(TrainingDataLoader):
    STORY_TAG = "[STORY]"

    def load(self):
        tales = []

        df = pd.read_csv(self.file_name)
        for i, row in df.iterrows():
            title = row['Title']
            story = row['Text'].strip()
            tales.append(
                f"{self.BEGIN_TAG} {title} {self.STORY_TAG} {story} {self.END_TAG}"
            )

        tales = "\n".join(tales)
        vocab = sorted(list(set(tales)))

        self._print_info(tales, vocab)

        return tales, vocab


class TextDataset(Dataset):
    def __init__(self, text, seq_len, wconv):
        self.sequences = self.__get_sequences(text, seq_len)
        self.wconv = wconv

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        x = self.wconv.encode_t(sequence[:-1])
        y = self.wconv.encode_t(sequence[1:])
        return x, y

    @staticmethod
    def __get_sequences(text, seq_len):
        sequences = []
        for i in range(0, len(text) - seq_len, seq_len // 2):
            # 50% overlap between sequences
            sequences.append(text[i:i+seq_len])
        return sequences


#
# Model training
#

class Device:
    @staticmethod
    def get():
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def is_cuda():
        return Device.get() == "cuda"

    @staticmethod
    def init():
        if Device.is_cuda():
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()

    @staticmethod
    def print_info():
        if Device.is_cuda():
            print(f" GPU memory (allocated): {torch.cuda.memory_allocated(0) / 1e9:.2f}GB")
            print(f" GPU memory (cached): {torch.cuda.memory_reserved(0) / 1e9:.2f}GB")


class ModelTrainer:
    ITER_PRINT = 1

    def __init__(self, model, text, vocab, lr, params_file_name):
        self.model = model
        self.text = text
        self.params_file_name = params_file_name
        self.vocab_size = len(vocab)
        self.word_conv = WordConverter(vocab)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=lr)
        self.scheduler = self.__get_scheduler(self.optimizer, 20)
        self.scaler = torch.amp.GradScaler(Device.get())

    def load_params(self):
        try:
            self.model.load_state_dict(torch.load(self.params_file_name))
            return True
        except:
            return False

    def save_params(self):
        try:
            torch.save(self.model.state_dict(), self.params_file_name)
            return True
        except:
            return False

    def train(self, batch_size, seq_len, loss_target):
        try:
            self.__train(batch_size, seq_len, loss_target)
        except KeyboardInterrupt:
            print("Training stopped")

    def __train(self, batch_size, seq_len, loss_target):
        self.model.train()

        dataset = TextDataset(self.text, seq_len, self.word_conv)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )

        print(f"""Training started:
 - Total sequences: {len(dataset)}
 - Batch size: {batch_size}
 - Sequence length: {seq_len}
 - Target loss: {loss_target}
 - Model: {self.model.__class__.__name__}
 - Criterion: {self.criterion.__class__.__name__}
 - Optimizer: {self.optimizer.__class__.__name__}
 - Device: {Device.get()}
""")

        epoch_num = 1
        iter_num = 1
        loss_mean = 0.0
        loss_mean_iter = 0.0
        loss_sum = 0.0
        last_time = time.time()
        stop = False
        while not stop:
            for x, y in loader:
                # x, y -> batch_size, seq_len
                x, y = x.to(Device.get()), y.to(Device.get())

                with torch.amp.autocast(Device.get()):
                    # out -> batch_size, seq_len, vocab_size
                    out = self.model(x)
                    # out.view(...) -> batch_size * seq_len, vocab_size
                    # y.view(...)   -> batch_size * seq_len
                    loss = self.criterion(out.view(-1, self.vocab_size), y.view(-1))

                loss_sum += loss
                loss_mean = loss_sum / iter_num
                iter_num += 1

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            self.scheduler.step()

            if loss_mean < loss_target:
                print(f"{epoch_num}. Target loss reached, stopped. Loss: {loss_mean:.6f}, lr: {self.__get_lr():.4f}")
                stop = True
                break

            if epoch_num % self.ITER_PRINT == 0:
                loss_mean_iter_diff = (loss_mean_iter - loss_mean).abs()
                loss_mean_iter = loss_mean

                curr_time = time.time()
                elapsed_time = curr_time - last_time
                last_time = curr_time

                print(f"{epoch_num}. Loss: {loss_mean:.6f}, diff: {loss_mean_iter_diff:.6f}, lr: {self.__get_lr():.4f}, time: {elapsed_time:.3f}s")

            epoch_num += 1

    def __get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    @staticmethod
    def __get_scheduler(optimizer, warmup_steps):
        def lr_fct(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                return 0.95 ** ((step - warmup_steps) // 100)

        return lr_scheduler.LambdaLR(optimizer, lr_fct)


class TextGenerator:
    def __init__(self, model, vocab, seq_len):
        self.model = model
        self.seq_len = seq_len
        self.word_conv = WordConverter(vocab)

    @torch.no_grad()
    def generate_text(self, start_tag, end_tag, temp=1.0, max_len=50000, top_k=40):
        self.model.eval()

        text = start_tag
        while True:
            seq = text[-self.seq_len:]
            seq_enc = self.word_conv.encode_t(seq).to(Device.get())

            # batch_size, seq_len, vocab_size
            out = self.model(seq_enc.unsqueeze(0))
            logits = out[0, -1, :]

            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                probs = F.softmax(top_k_logits / temp, dim=-1)
                sampled_idx = torch.multinomial(probs, num_samples=1)
                next_char_enc = top_k_indices[sampled_idx].item()
            else:
                probs = F.softmax(logits / temp, dim=-1)
                next_char_enc = torch.multinomial(probs, num_samples=1).item()

            next_char = self.word_conv.decode([next_char_enc])

            text += next_char
            print(next_char, end="", flush=True)

            if len(text) > max_len or end_tag in text:
                break

        print("")


def compute_model_parameters_num(n_layers, d_model, d_ff):
    mha = (d_model * d_model * 3) + (d_model * d_model)
    ff = (d_model * d_ff) * 2
    ln = (d_model * 2) * 2

    return (mha + ff + ln) * n_layers


#
# Main
#

def main():
    class TrainingDataTypes(Enum):
        FAIRY_TALES = auto()
        SHAKESPEARE = auto()

    TRAINING_DATA_LOADERS = {
        TrainingDataTypes.FAIRY_TALES: { "loader": FairyTalesLoader, "file": "tales.csv" },
        TrainingDataTypes.SHAKESPEARE: { "loader": ShakespeareLoader, "file": "shakespeare.txt"},
    }
    BATCH_SIZE = 256
    SEQ_LEN = 128
    N_LAYERS = 8
    D_MODEL = 512
    N_HEAD = 8
    D_FF = 2048
    LEARN_RATE = 1e-3
    LOSS_TARGET = 0.5
    TRAINING_DATA_TYPE = TrainingDataTypes.FAIRY_TALES
    PARAMS_FILE_NAME = f"params_cbc_{TRAINING_DATA_TYPE.name.lower()}_b{BATCH_SIZE}_s{SEQ_LEN}__l{N_LAYERS}_m{D_MODEL}_h{N_HEAD}_f{D_FF}.pth"

    print(f"""Model:
 - n_layer: {N_LAYERS}
 - d_model: {D_MODEL}
 - n_head: {N_HEAD}
 - d_ff: {D_FF}
 - Total parameters: {compute_model_parameters_num(N_LAYERS, D_MODEL, D_FF)}
""")

    torch.manual_seed(0)

    Device.init()

    training_data_loader = TRAINING_DATA_LOADERS[TRAINING_DATA_TYPE]["loader"](TRAINING_DATA_LOADERS[TRAINING_DATA_TYPE]["file"])
    text, vocab = training_data_loader.load()
    model = TransformerDecoderOnly(
        len(vocab),
        SEQ_LEN,
        n_layers=N_LAYERS,
        d_model=D_MODEL,
        n_head=N_HEAD,
        d_ff=D_FF
    ).to(Device.get())

    model_trainer = ModelTrainer(model, text, vocab, LEARN_RATE, PARAMS_FILE_NAME)
    if model_trainer.load_params():
        answer = input(f"Model parameters successfully loaded from file {PARAMS_FILE_NAME}.\nTrain the model anyway? (y/n) ")
        train_model = answer.lower() in ["y", "yes"]
    else:
        print(f"Unable to load model parameters from file {PARAMS_FILE_NAME}")
        train_model = True

    if train_model:
        model_trainer.train(BATCH_SIZE, SEQ_LEN, LOSS_TARGET)
        if model_trainer.save_params():
            print(f"Model parameters successfully saved to: {PARAMS_FILE_NAME}")
        else:
            print("Unable to save model parameters")

    print("Generated text:")
    text_gen = TextGenerator(model, vocab, SEQ_LEN)
    text_gen.generate_text(TrainingDataLoader.BEGIN_TAG, TrainingDataLoader.END_TAG)


if __name__ == "__main__":
    main()
