import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import dataset
from transformer import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
hs=dataset.hearthstone()
X_train,Y_train=hs.dataset('train')
trainloader = DataLoader(TensorDataset(torch.from_numpy(X_train),torch.from_numpy(Y_train)), batch_size=32,shuffle=True, num_workers=12)
X_dev,Y_dev=hs.dataset('dev')
devloader = DataLoader(TensorDataset(torch.from_numpy(X_dev),torch.from_numpy(Y_dev)), batch_size=32,shuffle=True, num_workers=12)

# encoder
input_dim = len(hs.NL_voc)
hid_dim = 128 * 3
n_layers = 6
n_heads = 8
pf_dim = 2048
dropout = 0.1
enc = Encoder(input_dim, hid_dim, n_layers, n_heads, pf_dim, EncoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)

# decoder
output_dim = len(hs.PL_voc)
hid_dim = 128 * 3
n_layers = 6
n_heads = 8
pf_dim = 2048
dropout = 0.1
dec = Decoder(output_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)

pad_idx = hs.NL_voc[dataset.PAD]
model = Seq2Seq(enc, dec, pad_idx, device)
#model = torch.nn.DataParallel(model)
model.to(device)
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

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    steps=len(iterator)
    for i, sample_batched in enumerate(iterator):
        src,trg=sample_batched
        src,trg=src.to(device),trg.to(device)
        optimizer.optimizer.zero_grad()
        parent,name,trg = trg.split(1, 1)
        parent.squeeze_()
        name.squeeze_()
        trg.squeeze_()
        output = model(src, parent[:,:-1], name[:,:-1], trg[:,:-1])
        output = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:,1:].contiguous().view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return (epoch_loss / steps)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src,trg=batch
            src,trg=src.to(device),trg.to(device)
            parent,name,trg = trg.split(1, 1)
            parent.squeeze_()
            name.squeeze_()
            trg.squeeze_()
            output = model(src, parent[:,:-1], name[:,:-1], trg[:,:-1])
            output = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:,1:].contiguous().view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def decode(model, src):
    model.eval()
    with torch.no_grad():
        src=torch.from_numpy(src).to(device)
        parent0=[hs.PL_voc['root']]
        name0=[hs.PL_voc['root']]
        trg0=[hs.PL_voc[dataset.SOS]]

        output = model(src, )

clip=1
num_epochs=100

best=1000
for epoch in range(num_epochs):
    train_loss = train(model, trainloader, optimizer, criterion, clip)
    valid_loss = evaluate(model, devloader, criterion)
    print("epoch:%s train_loss:%s valid_loss:%s"%(epoch,train_loss,valid_loss))
    if valid_loss<best:
        best=valid_loss
        torch.save(model,'model.weights')
