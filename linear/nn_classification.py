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

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.datasets import make_moons


class MLP(nn.Sequential):
    def __init__(self, n_in, n_out):
        super().__init__(
            nn.Linear(n_in, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, n_out),
        )


x_list, y_list = make_moons(n_samples=200, noise=0.05)

x = torch.tensor(x_list).float()
y = torch.tensor(y_list).float().unsqueeze(1)

mlp = MLP(2, 1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(mlp.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
scheduler = lr_scheduler.LinearLR(
    optimizer,
    start_factor=1.0,
    end_factor=0.1,
    total_iters=10000
)

#
# Training
#
mlp.train()

batch_size = 8
epoch_print = 5
epoch_num = 100
for epoch in range(epoch_num + 1):

    ir = torch.randperm(x.shape[0])
    loss_sum = 0
    num_batches = 0
    for i in range(0, x.shape[0], batch_size):
        idx = ir[i:i+batch_size]
        xb, yb = x[idx], y[idx]
        outputs = mlp(xb)
        loss = criterion(outputs, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        num_batches += 1

    scheduler.step()

    if epoch % epoch_print == 0:
        print(f"{epoch}. Loss: {loss_sum / num_batches}, lr: {scheduler.get_last_lr()[0]}")


#
# Check results
#
@torch.no_grad()
def compute_outputs(m, x):
    m.eval()
    yp = m(x)
    yps = torch.sigmoid(yp)
    return (yps > 0.5).float()

ypn = compute_outputs(mlp, x)
print(all([(pred == asked).item() for pred, asked in zip(y, ypn)]))


h = 0.01
coord_ext = 1
x_min, x_max = x[:, 0].min() - coord_ext, x[:, 0].max() + coord_ext
y_min, y_max = x[:, 1].min() - coord_ext, x[:, 1].max() + coord_ext
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h)
)

x_grid = np.c_[xx.ravel(), yy.ravel()]
# x_grid = np.array(list(zip(xx.ravel(), yy.ravel())))
#plt.scatter(grid[:,0], grid[:,1])

y_grid = compute_outputs(mlp, torch.tensor(x_grid).float())
Z = np.array(y_grid)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(x[:, 0], x[:, 1], c=y, cmap="winter")
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()
