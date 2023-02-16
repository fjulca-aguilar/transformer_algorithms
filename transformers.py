import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import random

_eps = 1**-5
dtype = torch.float
print(f'dtype= {dtype}')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_len = 20

################################################
# Algorithm 4
################################################
class Attention(nn.Module):
    def __init__(self, dx, dz, dattn, dout):
        super().__init__()
        self.Wq = nn.Parameter(torch.randn((dattn, dx), dtype=dtype) / math.sqrt(dattn))
        self.bq = nn.Parameter(torch.zeros((dattn, 1), dtype=dtype))
        self.Wk = nn.Parameter(torch.randn((dattn, dz), dtype=dtype) / math.sqrt(dattn))
        self.bk = nn.Parameter(torch.zeros((dattn, 1), dtype=dtype))
        self.Wv = nn.Parameter(torch.randn((dout, dz), dtype=dtype) / math.sqrt(dout))
        self.bv = nn.Parameter(torch.zeros((dout, 1), dtype=dtype))
        self.dattn_sqrt = math.sqrt(dattn)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, X, Z, mask=None):
        """
        Parameters:
            X: Primary sequence (query).
            Z: Context sequence.
            X.shape = (dx, lx)
            Z.shape = (dz, lz)
            mask: boolean mask: mask[i,j] == True makes Z[i] influence 
            X[j]. 
        Output:
            tensor representation of X folding information from Z.
        """
        q = self.Wq @ X  + self.bq
        k = self.Wk @ Z + self.bk
        v = self.Wv @ Z + self.bv # dout x lz 
        
        score = k.T @ q # lz x lx        
        if mask is not None:
            assert mask.shape[0] == Z.shape[1] and mask.shape[1] == X.shape[1], \
                    f"Mask dimensions should be ({Z.shape[1]}, {X.shape[1]})"
            inf = 1000
            # score[~mask] = -inf
            score = score.masked_fill(~mask, -inf)

        return v @ self.softmax(score / self.dattn_sqrt) # dout x lx

################################################
# Algorithm 5
################################################
class MHAttention(nn.Module):
    def __init__(self, dx, dz, dattn, dout, dmid, H):
        super().__init__()
        assert dattn % H == 0, f'dattn={dattn} should be divisible by H={H}'
        self.attention = nn.ModuleList([Attention(dx, dz, dattn // H, dattn // H) for _ in range(H)])
        self.Wo = nn.Parameter(torch.randn((dout, dattn)))
        self.bo = nn.Parameter(torch.zeros((dout, 1)))

    def forward(self, X, Z, mask=None):
        '''
        Parameters:
            X: Primary (query) sequence.
            Z: Context sequence.
            X.shape = (dx, lx)
            Z.shape = (dz, lz)
            mask: boolean mask: mask[i,j] == True makes token in Z[i] influence 
            token X[j]. 
        Output:
            tensor representation of X with Z as context.
        '''
        Y = [attn(X, Z, mask) for attn in self.attention]
        Y = torch.cat(Y, dim=0)
        return self.Wo @ Y + self.bo

class Layer_norm(nn.Module):
    def __init__(self, de, beta=None):
        super().__init__()
        self.lamb = torch.nn.Parameter(torch.randn((de, 1)) / math.sqrt(de))
        self.beta = torch.nn.Parameter(torch.randn((de, 1)) / math.sqrt(de)) if beta is None else beta
    
    def forward(self, e):
        std, m = torch.std_mean(e, dim=0, keepdim=True)
        # add _eps to denominator for numerical stability
        return ((e - m) / (std + _eps)) * self.lamb + self.beta

class Unembedding(nn.Module):
    def __init__(self, W):
        super().__init__()
        self.W = W

    def forward(self, e):
        return nn.functional.softmax(self.W.dot(e))
       
################################################
# Algorithm 8: EDTransformer
################################################
class EDTransformer(nn.Module):
    def __init__(self, Lenc, Ldec, H, de, dmlp, We, Wp, Wu, dattn, dmid):
        '''
        Parameters:
            Lenc: Number of encoder (multi head) attention layers.
            Ldec: Number of decoder (multi head) attention layers.
            H: Number of heads per attention layer, assumes same number for encoder and decoder.
            de: Embedding dimension in attention.
            dmlp: Number of units per MLP layer, applied after each attention layer.
            We: Token embedding matrix.
            Wp: Positional embbeding matrix.
            Wu: Unembbeding matrix.
            dattn: number of units in attention layers.
            dmid: Number of output units per single attention layer in multi head attention layer. 
        '''
        super().__init__()

        self.We = torch.nn.Embedding(59, de)
        #We
        self.Wp = Wp
        self.Lenc = Lenc
        self.Ldec = Ldec
        self.encoder_MHeadAttentionLayers = nn.ModuleList()
        self.encoder_first_layer_norms = nn.ModuleList()
        self.encoder_second_layer_norms = nn.ModuleList()
        self.Wmlps1 = nn.ParameterList()
        self.Wmlps2 = nn.ParameterList()
        self.bmlps1 = nn.ParameterList()
        self.bmlps2 = nn.ParameterList()
        self.decoder_MHeadAttentionLayers = nn.ModuleList()
        self.decoder_first_layer_norms = nn.ModuleList()
        self.decoder_second_layer_norms = nn.ModuleList()
        self.decoder_third_layer_norms = nn.ModuleList()
        self.Wmlps3 = nn.ParameterList()
        self.Wmlps4 = nn.ParameterList()
        self.bmlps3 = nn.ParameterList()
        self.bmlps4 = nn.ParameterList()
        self.Wu = nn.Parameter(torch.randn((59, de)) / math.sqrt(59)) # Wu
        
        for _ in range(Lenc):
            self.encoder_MHeadAttentionLayers.append(MHAttention(de, de, dattn, de, dmid, H))
            self.encoder_first_layer_norms.append(Layer_norm(de))
            self.encoder_second_layer_norms.append(Layer_norm(de))
            self.Wmlps1.append(torch.nn.Parameter(torch.randn((dmlp, de))/ math.sqrt(dmlp)))
            self.Wmlps2.append(torch.nn.Parameter(torch.randn((de, dmlp)) / math.sqrt(de)))
            self.bmlps1.append(torch.nn.Parameter(torch.zeros((dmlp, 1))))
            self.bmlps2.append(torch.nn.Parameter(torch.zeros((de, 1))))

        for _ in range(Ldec):
            self.decoder_MHeadAttentionLayers.append(MHAttention(de, de, dattn, de, dmid, H))
            self.decoder_first_layer_norms.append(Layer_norm(de))
            self.decoder_second_layer_norms.append(Layer_norm(de))
            self.decoder_third_layer_norms.append(Layer_norm(de))
            self.Wmlps3.append(torch.nn.Parameter(torch.randn((dmlp, de)) / math.sqrt(dmlp)))
            self.Wmlps4.append(torch.nn.Parameter(torch.randn((de, dmlp)) / math.sqrt(de)))
            self.bmlps3.append(torch.nn.Parameter(torch.zeros((dmlp, 1))))
            self.bmlps4.append(torch.nn.Parameter(torch.zeros((de, 1))))
        self.relu=nn.ReLU()
        self.register_buffer("unidirectional_attention_mask", torch.triu(torch.ones((59, 59))) > 0)

    def forward(self, x, z):
        lz = z.shape[0]
        # e = [torch.from_numpy(self.We[:, z[t]] + self.Wp[:, t].reshape(-1, 1)) for t in range(lz)]
        # print('self.We(z).transpose(1,0) shape', self.We(z).transpose(1,0).shape)
        # print('self.Wp[:, :lz].transpose(1,0) shape', self.Wp[:, :lz].transpose(1,0).shape)
        Z = self.We(z).transpose(1,0) + self.Wp[:, :lz]
        # Z = torch.cat(e, dim=1) # Z.shape = (de, lz)
        for l in range(self.Lenc):
            Z = Z + self.encoder_MHeadAttentionLayers[l](Z, Z, mask=None)
            Z = self.encoder_first_layer_norms[l](Z)
            Z = Z + self.Wmlps2[l] @ self.relu(self.Wmlps1[l] @ Z + self.bmlps1[l]) + self.bmlps2[l]
            Z = self.encoder_second_layer_norms[l](Z)

        lx = x.shape[0]
        X = self.We(x).transpose(1,0) + self.Wp[:, :lx]
        # unidirectional_attention_mask = torch.triu(torch.ones((X.shape[1], X.shape[1]))) > 0
        for l in range(self.Ldec):
            X = X + self.decoder_MHeadAttentionLayers[l](X, X, mask=self.unidirectional_attention_mask[:lx, :lx])
            X = self.decoder_first_layer_norms[l](X)
            X = X + self.decoder_MHeadAttentionLayers[l](X, Z, mask=None)
            X = self.decoder_second_layer_norms[l](X)
            X = X + self.Wmlps4[l] @ self.relu(self.Wmlps3[l] @ X + self.bmlps3[l]) + self.bmlps4[l]
            X = self.decoder_third_layer_norms[l](X)
        return nn.functional.softmax(self.Wu @ X, dim=0)


################################################
# Algorithm 9: ETransformer
################################################
class ETransformer(nn.Module):
    def __init__(self, L, H, de, dmlp, df, We, Wp, Wu, dattn, dmid):
        '''
        Parameters:
            L: Number of (multi head) attention layers.
            H: Number of heads per attention layer.
            de: Embedding dimension in attention.
            dmlp: Number of units per MLP layer, applied after each attention layer.
            df: Number of units for final linar projection layer.
            We: Token embedding matrix.
            Wp: Positional embbeding matrix.
            Wu: Unembbeding matrix.
            dattn: number of units in attention layers.
            dmid: Number of output units per single attention layer in multi head attention layer. 
        '''
        super().__init__()
        self.We = We
        self.Wp = Wp
        self.L = L
        self.mHeadAttentionLayers = nn.ModuleList()
        self.first_layer_norms = nn.ModuleList()
        self.second_layer_norms = nn.ModuleList()
        self.Wmlps1 = nn.ModuleList()
        self.Wmlps2 = nn.ModuleList()
        self.bmlps1 = nn.ModuleList()
        self.bmlps2 = nn.ModuleList()
        self.Wf = torch.nn.Parameter(torch.randn(df, de))
        self.bf = torch.nn.Parameter(torch.randn((df, 1)))
        self.final_layer_norm = Layer_norm()
        self.Wu = Wu
        self.gelu = nn.GELU()
        
        for _ in range(L):
            self.mHeadAttentionLayers.append(MHAttention(de, de, dattn, de, dmid, H))
            self.first_layer_norms.append(Layer_norm())
            self.second_layer_norms.append(Layer_norm())
            self.Wmlps1.append(torch.nn.Parameter(torch.randn((dmlp, de))))
            self.Wmlps2.append(torch.nn.Parameter(torch.randn((de, dmlp))))
            self.bmlps1.append(torch.nn.Parameter(torch.randn((dmlp, 1))))
            self.bmlps2.append(torch.nn.Parameter(torch.randn((de, 1))))

    def forward(self, x):
        # x = tensor of shape (lx) with token IDs
        # assert len(x.shape) == 2, print(f'Wrong input dimension: {len(x.shape)}, expected 2')
        lx = len(x)
        e = [torch.from_numpy(self.We[:, x[t]] + self.Wp[:, t].reshape(-1, 1)) for t in range(lx)]
        X = torch.cat(e, dim=1) # X.shape = (dx, lx)
        #TODO: might need to make X.shape = (dx, lmax) to match fixed matrices in m. head attention layers
        # or do it before calling this function. Check that padding is not used during learning.
        mask = None # torch.tensor([True] * X.shape[0] for _ in range(X.shape[0]))
        for l in range(self.L):
            X = X + self.mHeadAttentionLayers[l](X, X, mask=mask)
            X = self.first_layer_norms[l](X)
            X = X + self.Wmlps2[l] @ self.gelu(self.Wmlps1[l] @ X + self.bmlps1[l]) + self.bmlps2[l]
            X = self.second_layer_norms[l](X)
        X = self.gelu(self.Wf @ X + self.bf)
        X = self.final_layer_norm(X)
        return nn.functional.softmax(self.Wu @ X, dim=0)
     


def positional_embedding(de, lmax):
    '''
    de: embedding dimension
    lmax: max sequence position
    '''
    print(f'Positional embedding with de={de}, lmax={lmax}')
    sin_cos_len = de // 2
    lmax_exp = 1. / (lmax ** (2. * torch.arange(sin_cos_len) / de))
    lmax_exp = lmax_exp.reshape((-1, 1))
    t = torch.arange(0, lmax, dtype=torch.float32).reshape((1, -1))
    angles = lmax_exp @ t
    Wp = torch.zeros((de, lmax))
    Wp[1::2, :] = torch.sin(angles)
    Wp[::2, :] = torch.cos(angles)
    return Wp

def visualize_pos_embedding(Wp):
    plt.pcolormesh(Wp, cmap='RdBu')
    plt.ylabel('De')
    plt.xlabel('Position')
    plt.colorbar()
    plt.show()


def char2num(c):
    return ord(c)
    # return ord(c) - ord('a')

def num2char(num):
    return chr(num + ord('a'))

def num_seq2char_seq(num_seq):
    return [num2char(num) for num in num_seq]

# mask_token = char2num('z') + 1
# start_token = char2num('z') + 2
# end_token = char2num('z') + 3

################################################
# Load dataset
################################################
def load_dataset():
    data = []
    with open('names.txt') as f:
        for line in f:
            line = line.strip().lower()
            data.append([start_token] + 
            [char2num(c) for c in line if ord('a') <= ord(c) <= ord('z')] +
            [end_token])
    print(f'Num data elements: {len(data)}')    
    print(f'First 10 elements {data[:10]}')
    return data

def char2num_seq(char_seq, vocabulary):
    num_seq = []
    for c in char_seq:
        if c not in vocabulary:
            vocabulary[c] = len(vocabulary) + 1
        num_seq.append(vocabulary[c])  
    return num_seq          


def loadSeq2SeqDataset(file_path='spa-eng/spa.txt'):
    """
    Loads dataset with language2language sentences 
    returns:
        (z, x): list of tensors with source and target tensor sentences, respectively
        each language has i
    """

    vocab_z, vocab_x = set(), set()
    z, x = [], []
    with open(file_path) as f:
        for line in tqdm(f):
            parts = line.strip().lower().split('\t')
            if len(parts[0]) > max_len or len(parts[1]) > max_len:
                continue
            z.append(parts[0])
            vocab_z.update(parts[0])
            x.append(parts[1])
            vocab_x.update(parts[1])
    char2num_z = {char:num for num, char in enumerate(sorted(vocab_z))}
    char2num_x = {char:num for num, char in enumerate(sorted(vocab_x))}
    print('Vocab z', vocab_z, 'num elem.', len(vocab_z))
    print('Vocab x', vocab_x, 'num elem.', len(vocab_x))
    # TODO: change this to max vocab value NOT MAX_LEN
    next_char_id = max(len(vocab_x), len(vocab_z))
    start_token = next_char_id
    end_token = next_char_id + 1
    for i in range(3):
        print(z[i], x[i])

    z = [torch.tensor([start_token] + [char2num_z[char] for char in line] + [end_token]) for line in z]
    x = [torch.tensor([start_token] + [char2num_x[char] for char in line] + [end_token]) for line in x]
    return z[:1], char2num_z, x[:1], char2num_x

            

    # eng2span = []
    # max_len = 0
    # max_ord = 0
    # min_ord = 1000000
    # vocabulary = {}
    # with open() as f:
    #     for line in tqdm(f):
    #         line = line.strip().lower()
    #         parts = line.split('\t')            
    #         eng2span.append((char2num_seq(parts[0], vocabulary), 
    #             char2num_seq(parts[1], vocabulary)))
    #         max_len = max(len(eng2span[-1][0]), len(eng2span[-1][1]), max_len)
    #         max_ord = max([max_ord] + eng2span[-1][0] + eng2span[-1][1])
    #         min_ord = min([min_ord] + eng2span[-1][0] + eng2span[-1][1])
    # # print(f'Num examples: len(z)={len(z)}, len(x)={len(x)}')
    # return eng2span, max_len, max_ord, min_ord, vocabulary

################################################
# Algorithm 12
################################################

def ETraining(data, nEpochs=10, lrate=1e-3, p_mask=0.5, saved_model_path='ETraining_model.pth'):
    ntokens = ord('z') - ord('a') + 4
    We = np.identity(ntokens, dtype=np.float32)
    Wu = np.linalg.inv(We)
    max_lx = 1
    for x in data:
        if len(x) > max_lx:
            max_lx = len(x)
    Wp = positional_embedding(We.shape[0], max_lx).astype(dtype=np.float32)
    # print(f'Positional embedding {We[:10]}')
    # We = torch.from_numpy(We)
    Wu = torch.from_numpy(Wu)
    # Wp = torch.from_numpy(Wp)
    # L, H, de, dmlp, df, We, Wp, Wu, dattn, dmid
    eTransformer = ETransformer(4, 2, We.shape[0], We.shape[0], 20, We, Wp, Wu, 20, 20)
    
    for epoch in range(nEpochs):
        for data_idx, x in tqdm(enumerate(data)):
            mask_indices =  np.random.binomial(1, p_mask, len(x)).astype(int)
            mask_indices = [idx for idx, mask_it in zip(range(len(x)), mask_indices) if mask_it == 1]
            # print(mask_indices)
            # print(f'mask_indices type: {mask_indices.dtype}')
            masked_x = x.copy()
            for mask_pos in mask_indices:
                masked_x[mask_pos] = mask_token
            # masked_x[mask_indices] = mask_token
            masked_x = [start_token] + masked_x + [end_token]
            # with torch.autograd.set_detect_anomaly(True):
            P = eTransformer(masked_x)
            x = torch.tensor(x)
            loss = -torch.sum(torch.log(P[x[mask_indices], mask_indices]))
            if data_idx % 100 == 0:
                print(f'Loss value at step {data_idx}: {loss.item()}')
            eTransformer.zero_grad()
            loss.backward()
            with torch.no_grad():
                for param in eTransformer.parameters():
                    param -= lrate * param.grad
        np.random.shuffle(data)
        print(f'Saving Model after epoch {epoch + 1}')
        torch.save(eTransformer.state_dict(), saved_model_path)
            
def run_ETraining():
    data = load_dataset()[:1000]
    # print(np.random.binomial(1, 0.5, 10))
    ETraining(data)



# eng2span, max_len, max_ord, min_ord, vocabulary = loadSeq2SeqDataset()
# print('#### max len', max_len)
# print('#### max ord', max_ord)
# print('#### min ord', min_ord)
# print('#### num vocabulary keys', len(vocabulary))
# print('#### vocabulary', vocabulary)

# mask_token = len(vocabulary) + 1
# start_token = len(vocabulary) + 2
# end_token = len(vocabulary) + 3

# vocabulary_num2char = {val:key for key, val in vocabulary.items()}
# vocabulary_num2char[mask_token] = 'A'
# vocabulary_num2char[start_token] = 'B'
# vocabulary_num2char[end_token] = 'C'

# print('### data', data[:2])

# source, char2num_z, target, char2num_x = loadSeq2SeqDataset()
# print('source len', len(source))


################################################
# Algorithm 11
################################################
def EDTraining(nEpochs=10, lrate=1e-4, saved_model_path='EDTraining_model.pth', batch_size=8):
    source, char2num_z, target, char2num_x = loadSeq2SeqDataset()
    print('source len', len(source))
    de = 512
    n_different_characters = 59
    We = torch.nn.Embedding(n_different_characters, de) #torch.randn((de, len(vocabulary) + 4)) 
    Wu = torch.randn((n_different_characters, de)) # torch.nn.Embedding((max_len + 2, de)) # torch.randn((de, len(vocabulary) + 4)) # np.linalg.inv(We)
    Wp = positional_embedding(de, max_len + 2)

    transformer = EDTransformer(Lenc=2, Ldec=2, H=2, de=512, dmlp=2048, We=We, Wp=Wp, Wu=Wu, dattn=128, dmid=128)
    # for param in transformer.parameters():
    #     print('*** param', type(param.data), param.size())
    print('Network: ', transformer)
    for epoch in tqdm(range(nEpochs)):
        # for data_idx, source_target in tqdm(enumerate(zip(source, target))):
        for data_idx, source_target in enumerate(zip(source, target)):
            z = source_target[0]
            x = source_target[1]     
            P = transformer(x, z)

            loss = -torch.mean(torch.log(torch.clip(P[x[1:x.shape[0]], range(x.shape[0] - 1)], 1e-9)))
            # print(f'Loss value at step {data_idx}: {loss.item()}')
            # if data_idx % 1000 == 0:
                # print('### P ', P)
                # print('### max P ', max(P[:, 2]))
                # print(f' Sequences x={x}, z ={z}')
                # print(f' Sequences length x={x.shape[0]}, z ={z.shape[0]}')
                # print(f'Loss value at step {data_idx}: {loss.item()}')
                # print(f'Saving Model after epoch {epoch + 1}')
                # print('P[x[1:x.shape[0]-1], range(x.shape[0] - 2)]', P[x[1:x.shape[0]-1], range(x.shape[0] - 2)])
                # torch.save(transformer.state_dict(), f'{saved_model_path}_epoch_{epoch}')
                # print(zx)
            
            loss.backward()
            with torch.no_grad():
                for param in transformer.parameters():
                    param -= lrate * param.grad
                transformer.zero_grad()
        # np.random.shuffle(data)
        indx = list(range(len(source)))
        random.shuffle(indx)
        source = [source[i] for i in indx]
        target = [target[i] for i in indx]

        # print(f'Saving Model after epoch {epoch + 1}')
        torch.save(transformer.state_dict(), saved_model_path)

# Wp = positional_embedding(512, 2048).astype(dtype=np.float32)
# visualize_pos_embedding(Wp)


# EDTraining(lrate=1e-2, nEpochs=10000)
#TODO: fix indexing issue and train with mini-batch>1

################################################
# Algorithm 15
################################################
def EDInference(z, transformer):
    # We = np.identity(len(vocabulary) + 4, dtype=np.float32)
    # Wu = np.linalg.inv(We)
    x = torch.tensor([57], dtype=torch.int)
    y = torch.tensor([57])
    end_token = 58
    t = 1
    while y[0] != end_token and len(x) < max_len:
        P = transformer(x, z)
        p = P[:, x.shape[0] - 1]
        # print('x.shape[0]', x.shape[0])
        # print(p)
        # print(P)
        y = torch.tensor([np.random.choice(59, p=p.detach().numpy() ** (1 / t))])
        # print('Chosen ', vocabulary_num2char[y])
        # print('***** y', y)
        x = torch.cat([x, y])
    print('x', x)
    return x

# run inference
source, char2num_z, target, char2num_x = loadSeq2SeqDataset()
z = 'go.'
z = torch.tensor([57] + \
    [char2num_z[achar] for achar in z] + \
    [57 + 1])
num2char_x = ['0'] * (len(char2num_x) + 2)
for achar, num in char2num_x.items():
    num2char_x[num] = achar
num2char_x[57] = 'S'
num2char_x[58] = 'E'
# {num:achar for achar, num in char2num_x.items()}
# print('char2num_x.items()', char2num_x.items())
de = 512
Wp = positional_embedding(de, max_len + 2)
transformer = EDTransformer(Lenc=2, Ldec=2, H=2, de=512, dmlp=2048, We=None, Wp=Wp, Wu=None, dattn=128, dmid=128)

# # ntokens = ord('z') - ord('a') + 4
# We = np.identity(len(vocabulary) + 4, dtype=np.float32)
# Wu = np.linalg.inv(We)
# Wu = torch.from_numpy(Wu)
# Wp = positional_embedding(We.shape[0], max_len).astype(dtype=np.float32)
# transformer = EDTransformer(Lenc=2, Ldec=2, H=2, de=We.shape[0], dmlp=We.shape[0], df=20, We=We, Wp=Wp, Wu=Wu, dattn=20, dmid=20)
transformer.load_state_dict(torch.load('EDTraining_model.pth'))
transformer.eval()
x = EDInference(z, transformer)
# print('num2char_x', num2char_x)
print(f'Text in Spanish: {str([num2char_x[n]  for n in x])}')




