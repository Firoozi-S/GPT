import torch
import torch.nn as nn
from torch.nn import functional as F

import globals

@torch.no_grad()
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(in_features = globals.NUM_EMBEDDINGS, out_features = globals.HEAD_SIZE, bias = False)
        self.key = nn.Linear(in_features = globals.NUM_EMBEDDINGS, out_features = globals.HEAD_SIZE, bias = False)
        self.value = nn.Linear(in_features = globals.NUM_EMBEDDINGS, out_features = globals.HEAD_SIZE, bias = False)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)

        matmul = query @ key.transpose(-2, -1)
        scale = matmul * key.shape[-1]**-0.5
        lower_triangular_mask = torch.tril(input = torch.ones_like(input = scale), diagonal = -1)
        scale[lower_triangular_mask.bool()] = -float('inf')
        softmax = F.softmax(input = scale, dim = -1)
        value = self.value(x)

        return softmax @ value

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.fn_1 = nn.Linear(in_features = globals.NUM_EMBEDDINGS, out_features = 2 * globals.NUM_EMBEDDINGS, bias = False)
        self.fn_2 = nn.Linear(in_features = 2 * globals.NUM_EMBEDDINGS, out_features = globals.NUM_EMBEDDINGS, bias = False)

    def forward(self, x):
        x = self.fn_1(x)
        x = nn.ReLU()(x)
        return self.fn_2(x)

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList(modules = [ScaledDotProductAttention() for _ in range(globals.NUM_HEADS)])
        self.proj = nn.Linear(in_features = globals.NUM_EMBEDDINGS, out_features = globals.NUM_EMBEDDINGS)
        self.dropout = nn.Dropout(globals.DROPOUT_RATE)

    def forward(self, x):
        x = torch.cat(tensors = [h(x) for h in self.heads], dim = -1)
        output = self.dropout(self.proj(x))
        return output

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention = MultiHeadAttention()
        self.feed_forward = FeedForward()
        self.layer_norm_1 = nn.LayerNorm(normalized_shape = globals.NUM_EMBEDDINGS)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape = globals.NUM_EMBEDDINGS)

    def forward(self, x):
        x = x + self.self_attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x
        
class GPT(nn.Module):
    def __init__(self, vocab_size):
        super(GPT, self).__init__()
        self.token_embedding_table = nn.Embedding(num_embeddings = vocab_size, embedding_dim = globals.NUM_EMBEDDINGS)
        self.position_embedding_table = nn.Embedding(num_embeddings = globals.BLOCK_SIZE, embedding_dim = globals.NUM_EMBEDDINGS)
        self.blocks = nn.Sequential(*[Block() for _ in range(globals.NUM_LAYERS)])
        self.layer_norm = nn.LayerNorm(normalized_shape = globals.NUM_EMBEDDINGS)
        self.linear_head = nn.Linear(in_features = globals.NUM_EMBEDDINGS, out_features = vocab_size)

        self.apply(self.__init__weights)

    def __init__weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)

    def forward(self, data, targets = None):
        B, T = data.shape
        token_embedding = self.token_embedding_table(data)
        position_embedding = self.position_embedding_table(torch.arange(T))
        x = token_embedding + position_embedding
        x = self.blocks(x)
        x = self.layer_norm(x)

        logits = self.linear_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, index, num_tokens):

        for _ in range(num_tokens):
            index_cond = index[ : , -globals.BATCH_SIZE : ]

            logits, _ = self(index_cond)

            logits = logits[:, -1, :]

            probabilities = F.softmax(input = logits, dim = -1)
            
            index_next = torch.multinomial(input = probabilities, num_samples = 1)

            index = torch.cat(tensors = (index, index_next), dim = 1)

        return index

