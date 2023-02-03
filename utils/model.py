import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_model(vocab, tgt, dist_bar, N=6, embed_dim=512, ffn_dim=1024, head=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiScaleMultiHeadedAttention(head, embed_dim, dist_bar)
    ff = FeedForward(embed_dim, ffn_dim, dropout)

    model = MultiScaleTransformer3D(Encoder(EncoderLayer(embed_dim, c(attn), c(ff), dropout), N),
                                    Embeddings(embed_dim, vocab), Generator3D(embed_dim, tgt, dropout))
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class MultiScaleTransformer3D(nn.Module):
    def __init__(self, encoder, src_embed, generator):
        super(MultiScaleTransformer3D, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.generator = generator

    def forward(self, src, pos, mask):
        x = self.src_embed(src)
        dist = torch.cdist(pos, pos).float()
        feat = self.encoder(x, dist, mask.unsqueeze(1))[:, 0]
        return feat, self.generator(feat)


class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, dist, mask):
        for layer in self.layers:
            x = layer(x, dist, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """MultiRelationEncoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, dist, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, dist, mask))
        return self.sublayer[1](x, self.feed_forward)


class LayerNorm(nn.Module):
    """ layernorm layer"""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True) + self.eps
        return self.a_2 * (x - mean) / std + self.b_2


class SublayerConnection(nn.Module):
    """A residual connection followed by a layer norm. For code simplicity the norm is first as opposed to last."""

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


#######################################
## attention part
#######################################

def attention(query, key, value, dist_conv, dist, dist_bar, mask=None, dropout=None):
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
    scores *= dist_conv

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e10)

    # the mask of multi-scale self-attention
    out = []
    for i in dist_bar:

        # all interactions with the center node are allowed
        dist_mask = dist < i
        dist_mask[:, :, 0, :] = 1
        dist_mask[:, :, :, 0] = 1

        # mask the attention scores according to the distance
        scores_dist = scores.masked_fill(dist_mask == 0, -1e10)

        p_attn = F.softmax(scores_dist, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        out.append(torch.matmul(p_attn, value))
    return out, p_attn


class MultiScaleMultiHeadedAttention(nn.Module):
    def __init__(self, h, embed_dim, dist_bar, dropout=0.1):
        super(MultiScaleMultiHeadedAttention, self).__init__()
        assert embed_dim % h == 0
        # 4 linear layers
        self.linears = clones(nn.Linear(embed_dim, embed_dim), 4)
        self.cnn = nn.Sequential(nn.Conv2d(1, h, kernel_size=1), nn.ReLU(), nn.Conv2d(h, h, kernel_size=1))
        self.scale_linear = nn.Sequential(nn.Linear((len(dist_bar) + 1) * embed_dim, embed_dim), nn.ReLU(),
                                          nn.Dropout(p=dropout), nn.Linear(embed_dim, embed_dim))
        self.dropout = nn.Dropout(p=dropout)
        self.dist_bar = dist_bar + [1e10]
        self.d_k = embed_dim // h
        self.h = h
        self.attn = None

    def forward(self, query, key, value, dist, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from (B, embed_dim) => (B, head, N, d_k)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                             zip(self.linears, (query, key, value))]
        dist = dist.unsqueeze(1)
        dist_conv = self.cnn(dist)

        # 2) Apply attention on all the projected vectors in batch.
        x_list, self.attn = attention(query, key, value, dist_conv, dist,
                                      self.dist_bar, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x_list = [x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k) for x in x_list]
        x = torch.cat([self.linears[-1](x) for x in x_list], dim=-1)
        return self.scale_linear(x)


#######################################
## Position-wise feedforward neural networks
#######################################

class FeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, embed_dim, ffn_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(embed_dim, ffn_dim)
        self.w_2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, embed_dim, vocab):
        super(Embeddings, self).__init__()
        self.embed = nn.Embedding(vocab, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.embed_dim)


#######################################
## predictor
#######################################

class Generator3D(nn.Module):
    def __init__(self, embed_dim, tgt, dropout):
        super(Generator3D, self).__init__()
        self.proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Dropout(p=dropout),
                                  nn.Linear(embed_dim, tgt))

    def forward(self, x):
        return self.proj(x)

