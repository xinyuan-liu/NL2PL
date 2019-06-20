import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import dataset
from transformer import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
hs=dataset.hearthstone()
X_train,Y_train=hs.dataset('train')
trainloader = DataLoader(TensorDataset(torch.from_numpy(X_train),torch.from_numpy(Y_train)), batch_size=64,shuffle=True, num_workers=12)

# encoder
input_dim = len(hs.NL_voc)
hid_dim = 512
n_layers = 6
n_heads = 8
pf_dim = 2048
dropout = 0.1
enc = Encoder(input_dim, hid_dim, n_layers, n_heads, pf_dim, EncoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)

# decoder
output_dim = len(hs.PL_voc)
hid_dim = 512
n_layers = 6
n_heads = 8
pf_dim = 2048
dropout = 0.1
dec = Decoder(output_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)

pad_idx = hs.NL_voc[dataset.PAD]
model = Seq2Seq(enc, dec, pad_idx, device).to(device)

print('The model has %d trainable parameters'%sum(p.numel() for p in model.parameters() if p.requires_grad))

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

optimizer = NoamOpt(hid_dim, 1, 2000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

num_epochs=50
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for i, sample_batched in enumerate(trainloader):
        model.train()
        src,trg=sample_batched
        optimizer.optimizer.zero_grad()
        print(src.shape)
        print(trg.shape)
        input()

