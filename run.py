

import torch
from torch import nn
import torch.nn.functional as F

import random, math


def mask_(matrices, maskval=0.0, mask_diagonal=True):
    """
    Masks out all values in the given batch of matrices where i <= j holds,
    i < j if mask_diagonal is false

    In place operation

    :param tns:
    :return:
    """

    b, h, w = matrices.size()

    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[:, indices[0], indices[1]] = maskval



def contains_nan(tensor):
    return bool((tensor != tensor).sum() > 0)



# Self-Attention
class SelfAttention(nn.Module):
    def __init__(self, emb, heads=8, mask=False):
        """
        :param emb:
        :param heads:
        :param mask:
        """

        super().__init__()

        self.emb = emb
        self.heads = heads
        self.mask = mask

        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x):

        b, t, e = x.size()
        h = self.heads
        assert e == self.emb

        keys    = self.tokeys(x)   .view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values  = self.tovalues(x) .view(b, t, h, e)

        # compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1/4))
        keys    = keys / (e ** (1/4))
        # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        #   This should be more memory efficient

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b*h, t, t)

        if self.mask: # mask out the lower half of the dot matrix,including the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        dot = F.softmax(dot, dim=2) # dot now has row-wise self-attention probabilities

        assert not contains_nan(dot[:, 1:, :]) # only the forst row may contain nan

        if self.mask == 'first':
            dot = dot.clone()
            dot[:, :1, :] = 0.0
            # - The first row of the first attention matrix is entirely masked out, so the softmax operation results
            #   in a division by zero. We set this row to zero by hand to get rid of the NaNs

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, e)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)

        return self.unifyheads(out)



# Transformer Block

class TransformerBlock(nn.Module):
    def __init__(self, emb, heads, mask, seq_length, ff_hidden_mult=4, dropout=0.5):
        super().__init__()

        self.attention = SelfAttention(emb, heads=heads, mask=mask)
        self.mask = mask

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.do = nn.Dropout(dropout)

    def forward(self, x):

        attended = self.attention(x)

        x = self.norm1(attended + x)

        x = self.do(x)

        fedforward = self.ff(x)

        x = self.norm2(fedforward + x)

        x = self.do(x)

        return x


##RTransformer

class RTransformer(nn.Module):
    """
    Transformer for sequences Regression

    """

    def __init__(self, emb, heads, depth, seq_length, num_tokens, num_classes, max_pool=True, dropout=0.0):
        """
        :param emb: Embedding dimension
        :param heads: nr. of attention heads
        :param depth: Number of transformer blocks
        :param seq_length: Expected maximum sequence length
        :param num_tokens: Number of tokens (usually words) in the vocabulary
        :param num_classes: Number of classes.
        :param max_pool: If true, use global max pooling in the last layer. If false, use global
                         average pooling.
        """
        super().__init__()

        self.num_tokens, self.max_pool = num_tokens, max_pool

        # self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
        self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, seq_length=seq_length, mask=False, dropout=dropout))

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(emb, num_classes)

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: A batch by sequence length integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        sentences_emb = x
        b, t, e = x.size()

        positions = self.pos_embedding(torch.arange(t))[None, :, :].expand(b, t, e)
        # positions = torch.tensor(positions, dtype=torch.float32)
        x = sentences_emb + positions
        x = self.do(x)

        x = self.tblocks(x)

        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1) # pool over the time dimension
        x = self.toprobs(x)
        x = torch.squeeze(x)

        return x

# Format Dataset

import torch
from torch.utils import data

class Dataset(data.Dataset):
    def __init__(self, texts, labels):
        'Initialization'
        self.labels = labels
        self.text = texts

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        if torch.is_tensor(index):
            index = index.tolist()

        # Load data and get label
        X = self.text[index ,: ,:]
        y = self.labels[index]

        return X, y



import numpy as np
# Load your own the whole dataset
TEXT_emb = np.load("sorted_embed_3days.npy")
LABEL_emb = np.load("sorted_label_3days.npy")




# Main function

from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import random, sys, math, gzip

from tqdm import tqdm

NUM_CLS = 1

def go(arg):
    """
    Creates and trains a basic transformer for any regression task.
    """

    if arg.final:

        train, val = train_test_split(TEXT_emb, test_size=0.2)
        train_label, val_label = train_test_split(LABEL_emb, test_size=0.2)
        training_set = Dataset(train, train_label)
        val_set = Dataset(val, val_label)

    else:
        train, val = train_test_split(TEXT_emb, test_size=0.2)
        train_label, val_label = train_test_split(LABEL_emb, test_size=0.2)
        train, val = train_test_split(train, test_size=0.2)
        train_label, val_label = train_test_split(train_label, test_size=0.2)

        training_set = Dataset(train, train_label)
        val_set = Dataset(val, val_label)

    trainloader =torch.utils.data.DataLoader(training_set, batch_size=arg.batch_size, shuffle=False, num_workers=2)
    testloader =torch.utils.data.DataLoader(val_set, batch_size=len(val_set), shuffle=False, num_workers=2)
    print('training examples', len(training_set))
    # print(f'- nr. of {"test" if arg.final else "validation"} examples {len(test_iter)}')

    if arg.final:
        print('test examples', len(val_set))
    else:
        print('validation examples', len(val_set))


    # create the model
    model = RTransformer(emb=arg.embedding_size, heads=arg.num_heads, depth=arg.depth, \
                         seq_length=arg.max_length, num_tokens=arg.vocab_size, num_classes=NUM_CLS, max_pool=arg.max_pool)
    #     if torch.cuda.is_available():
    #         model.cuda()

    opt = torch.optim.Adam(lr=arg.lr, params=model.parameters())

    # training loop
    seen = 0
    evaluation= {'epoch': [] ,'Train Accuracy': [], 'Test Accuracy' :[]}
    for e in tqdm(range(arg.num_epochs)):
        train_loss_tol = 0.0
        print('\n epoch ' ,e)
        model.train(True)

        for i, data in enumerate(tqdm(trainloader)):
            if i > 2:
                break
            # learning rate warmup
            # - we linearly increase the learning rate from 10e-10 to arg.lr over the first
            #   few thousand batches
            if arg.lr_warmup > 0 and seen < arg.lr_warmup:
                lr = max((arg.lr / arg.lr_warmup) * seen, 1e-10)
                opt.lr = lr

            opt.zero_grad()

            inputs, labels = data
            inputs = Variable(inputs.type(torch.FloatTensor))
            # labels = torch.tensor(labels, dtype=torch.float32)
            labels = labels.clone().detach()
            if inputs.size(1) > arg.max_length:
                inputs = inputs[:, :arg.max_length, :]
            out = model(inputs)
            out = torch.unsqueeze(out, 0)
            # print(out)
            out = out.float()
            labels = labels.float()

            # print(out.shape,labels.shape)

            loss = F.mse_loss(out, labels)
            train_loss_tol += loss

            loss.backward()

            # clip gradients
            # - If the total gradient vector has a length > 1, we clip it back down to 1.
            if arg.gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), arg.gradient_clipping)

            opt.step()

            seen += inputs.size(0)
            # tbw.add_scalar('classification/train-loss', float(loss.item()), seen)
        # print('train_loss: ',train_loss_tol)
        train_loss_tol = train_loss_tol /( i +1)
        with torch.no_grad():

            model.train(False)
            tot, cor= 0.0, 0.0

            loss_test = 0.0
            for i, data in enumerate(tqdm(testloader)):
                if i > 2:
                    break
                inputs, labels = data
                inputs, labels = torch.tensor(inputs, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)
                if inputs.size(1) > arg.max_length:
                    inputs = inputs[:, :arg.max_length, :]
                out = model(inputs)

                loss_test += F.mse_loss(out, labels)
                # tot = float(inputs.size(0))
                # cor += float(labels.sum().item())

            acc = loss_test.numpy()
            if arg.final:
                print('test accuracy', acc)
            else:
                print('validation accuracy', acc)

        torch.save(model, './checkpoint/epoch' +str(e) +'.pth')
        # print(train_loss_tol)
        # print(acc)
        train_loss_tol = train_loss_tol.detach().numpy()
        evaluation['epoch'].append(e)
        evaluation['Train Accuracy'].append(train_loss_tol)
        evaluation['Test Accuracy'].append(acc)


    evaluation = pd.DataFrame(evaluation)
    evaluation.sort_values(["Test Accuracy"] ,ascending=True ,inplace=True)

    return evaluation
    # tbw.add_scalar('classification/test-loss', float(loss.item()), e)


# Run the main function
if __name__ == "__main__":

    #print('OPTIONS ', options)
    # Tuning Parameters:
    import easydict
    from argparse import ArgumentParser
    parser = ArgumentParser()
    args = parser.parse_known_args()[0]
    args = easydict.EasyDict({
            "num_epochs": 2,
            "batch_size": 1,
            "lr": 0.0005,
            "tb_dir": "./runs",
            "final": False,
            "max_pool": False,
            "embedding_size" : 1024,
            "vocab_size" : 50000,
            "max_length" : 520,
            "num_heads" : 1,
            "depth" : 1,
            "seed" : 1,
            "lr_warmup" : 500,
            "gradient_clipping" : 1.0
    })

    evaluation = go(args)