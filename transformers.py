import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import random

_eps = 1e-5
dtype = torch.float
device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_len = 20
START_CHAR = 'S'
END_CHAR = 'E'

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
        self.attention = nn.ModuleList([Attention(dx, dz, dattn, dmid) for _ in range(H)])
        self.Wo = nn.Parameter(torch.randn((dout, dattn * H)) / math.sqrt(dout))
        self.bo = nn.Parameter(torch.zeros((dout, 1)))

    def forward(self, X, Z, mask=None):
        """
        Parameters:
            X: Primary (query) sequence.
            Z: Context sequence.
            X.shape = (dx, lx)
            Z.shape = (dz, lz)
            mask: boolean mask: mask[i,j] == True makes token in Z[i] influence 
            token X[j]. 
        Output:
            tensor representation of X with Z as context.
        """
        Y = [attn(X, Z, mask) for attn in self.attention]
        Y = torch.cat(Y, dim=0)
        return self.Wo @ Y + self.bo


################################################
# Algorithm 6: EDTransformer
################################################
class Layer_norm(nn.Module):
    def __init__(self, de, beta=None):
        super().__init__()
        self.lamb = torch.nn.Parameter(torch.ones((de, 1)))
        self.beta = torch.nn.Parameter(torch.zeros((de, 1))) if beta is None else beta
    
    def forward(self, e):
        m = e.mean(dim=0, keepdim=True)
        v = e.var(dim=0, keepdim=True)
        # add _eps to denominator for numerical stability
        return ((e - m) / torch.sqrt(v + _eps)) * self.lamb + self.beta
       
################################################
# Algorithm 8: EDTransformer
################################################
class EDTransformer(nn.Module):
    def __init__(self, Lenc, Ldec, H, de, dmlp, Nv, lmax, dattn, dmid):
        """
        Parameters:
            Lenc: Number of encoder multi head attention layers.
            Ldec: Number of decoder multi head attention layers.
            H: Number of heads per attention layer, assumes same number for encoder and decoder.
            de: Embedding dimension in attention.
            dmlp: Number of units per MLP layer, applied after each attention layer.
            Nv: Vocabulary size.
            lmax: max length of sequence.
            Wu: Unembbeding matrix.
            dattn: number of units in attention layers.
            dmid: Number of output units per single attention layer in multi head attention layer. 
        """
        super().__init__()

        self.We = torch.nn.Embedding(Nv, de)
        self.Wp = positional_embedding(de, lmax)
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
        self.Wu = nn.Parameter(torch.randn((Nv, de)) / math.sqrt(Nv)) # 
        
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
        self.register_buffer("unidirectional_attention_mask", torch.triu(torch.ones((lmax, lmax))) > 0)

    def forward(self, x, z):
        lz = z.shape[0]
        Z = self.We(z).transpose(1,0) + self.Wp[:, :lz]
        for l in range(self.Lenc):
            Z = Z + self.encoder_MHeadAttentionLayers[l](Z, Z, mask=None)
            Z = self.encoder_first_layer_norms[l](Z)
            Z = Z + self.Wmlps2[l] @ self.relu(self.Wmlps1[l] @ Z + self.bmlps1[l]) + self.bmlps2[l]
            Z = self.encoder_second_layer_norms[l](Z)

        lx = x.shape[0]
        X = self.We(x).transpose(1,0) + self.Wp[:, :lx]
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
    def __init__(self, L, H, de, dmlp, df, Nv, lmax, dattn, dmid):
        """
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
        """
        super().__init__()
        self.We = torch.nn.Embedding(Nv, de)
        self.Wp = positional_embedding(de, lmax)
        self.L = L
        self.mHeadAttentionLayers = nn.ModuleList()
        self.first_layer_norms = nn.ModuleList()
        self.second_layer_norms = nn.ModuleList()
        self.Wmlps1 = nn.ParameterList()
        self.Wmlps2 = nn.ParameterList()
        self.bmlps1 = nn.ParameterList()
        self.bmlps2 = nn.ParameterList()
        self.Wf = torch.nn.Parameter(torch.randn(df, de) / math.sqrt(df))
        self.bf = torch.nn.Parameter(torch.zeros((df, 1)) / math.sqrt(df))
        self.final_layer_norm = Layer_norm(df)
        self.Wu = nn.Parameter(torch.randn((Nv, df)) / math.sqrt(Nv))
        self.gelu = nn.GELU()
        
        for _ in range(L):
            self.mHeadAttentionLayers.append(MHAttention(de, de, dattn, de, dmid, H))
            self.first_layer_norms.append(Layer_norm(de))
            self.second_layer_norms.append(Layer_norm(de))
            self.Wmlps1.append(torch.nn.Parameter(torch.randn((dmlp, de)) / math.sqrt(dmlp)))
            self.Wmlps2.append(torch.nn.Parameter(torch.randn((de, dmlp)) / math.sqrt(de)))
            self.bmlps1.append(torch.nn.Parameter(torch.zeros((dmlp, 1))))
            self.bmlps2.append(torch.nn.Parameter(torch.zeros((de, 1))))

    def forward(self, x):
        lx = x.shape[0]
        X = self.We(x).transpose(1,0) + self.Wp[:, :lx]
        for l in range(self.L):
            X = X + self.mHeadAttentionLayers[l](X, X, mask=None)
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

################################################
# Load dataset
################################################
def loadSeq2SeqDataset(file_path='spa-eng/spa.txt'):
    """
    Loads dataset with language2language sentences 
    returns:
        (z, x): list of tensors with source and target tensor sentences, respectively

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

    all_chars = vocab_x.union(vocab_z)
    char2num = {char:num for num, char in enumerate(sorted(all_chars))}
    start_token = len(all_chars)
    end_token = len(all_chars) + 1
    char2num[START_CHAR] = start_token
    char2num[END_CHAR] = end_token

    z = [torch.tensor([start_token] + [char2num[char] for char in line] + [end_token]) for line in z]
    x = [torch.tensor([start_token] + [char2num[char] for char in line] + [end_token]) for line in x]
    return z, x, char2num

def load_decoder_dataset(file_path='crime_and_punishment.txt'):
    """
    Loads decoder only dataset.
    returns:
        (x): list of tensors with sentence examples.

    """    
    x = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read().lower()
    all_chars = set(text)
    char2num = {char:num for num, char in enumerate(sorted(all_chars))}
    dataset_size = 1000

    start_ix = torch.randint(len(text) - max_len, (dataset_size,))
    x = [text[ix:ix+max_len] for ix in start_ix]
    start_token = len(all_chars)
    end_token = len(all_chars) + 1
    char2num[START_CHAR] = start_token
    char2num[END_CHAR] = end_token

    x = [torch.tensor([start_token] + [char2num[char] for char in sample] + [end_token]) for sample in x]
    return x, char2num

################################################
# Algorithm 10
################################################
class DTransformer(nn.Module):
    def __init__(self, L, H, de, dmlp, Nv, lmax, dattn, dmid):
        """
        Parameters:
            L: Number of encoder multi head attention layers.
            H: Number of heads per attention layer, assumes same number for encoder and decoder.
            de: Embedding dimension in attention.
            dmlp: Number of units per MLP layer, applied after each attention layer.
            Nv: Vocabulary size.
            lmax: max length of sequence.
            dattn: number of units in attention layers.
            dmid: Number of output units per single attention layer in multi head attention layer. 
        """
        super().__init__()

        self.We = torch.nn.Embedding(Nv, de)
        self.Wp = positional_embedding(de, lmax)
        self.L = L
        self.Wmlps1 = nn.ParameterList()
        self.Wmlps2 = nn.ParameterList()
        self.bmlps1 = nn.ParameterList()
        self.bmlps2 = nn.ParameterList()
        self.mHeadAttentionLayers = nn.ModuleList()
        self.first_layer_norms = nn.ModuleList()
        self.second_layer_norms = nn.ModuleList()
        self.third_layer_norm = Layer_norm(de)
        self.Wu = nn.Parameter(torch.randn((Nv, de)) / math.sqrt(Nv)) # 
        
        for _ in range(L):
            self.mHeadAttentionLayers.append(MHAttention(de, de, dattn, de, dmid, H))
            self.first_layer_norms.append(Layer_norm(de))
            self.second_layer_norms.append(Layer_norm(de))
            self.Wmlps1.append(torch.nn.Parameter(torch.randn((dmlp, de)) / math.sqrt(dmlp)))
            self.Wmlps2.append(torch.nn.Parameter(torch.randn((de, dmlp)) / math.sqrt(de)))
            self.bmlps1.append(torch.nn.Parameter(torch.zeros((dmlp, 1))))
            self.bmlps2.append(torch.nn.Parameter(torch.zeros((de, 1))))
        self.gelu=nn.GELU()
        self.register_buffer("unidirectional_attention_mask", torch.triu(torch.ones((lmax, lmax))) > 0)

    def forward(self, x):
        lx = x.shape[0]
        X = self.We(x).transpose(1,0) + self.Wp[:, :lx]
        for l in range(self.L):
            X = self.first_layer_norms[l](X)
            X = X + self.mHeadAttentionLayers[l](X, X, self.unidirectional_attention_mask[:lx, :lx])
            X = self.second_layer_norms[l](X)
            X = X + self.Wmlps2[l] @ self.gelu(self.Wmlps1[l] @ X + self.bmlps1[l]) + self.bmlps2[l]
        X = self.third_layer_norm(X)
        return nn.functional.softmax(self.Wu @ X, dim=0)


################################################
# Algorithm 11
################################################
def EDTraining(nEpochs=10, lrate=1e-4, saved_model_path='EDTraining_model.pth'):
    source, target, char2num = loadSeq2SeqDataset()
    print('*** dataset size:', len(source))
    print('*** vocabulary size:', len(char2num))
    
    transformer = EDTransformer(Lenc=2, Ldec=2, H=2, de=512, dmlp=2048, Nv=len(char2num), lmax=max_len + 2, dattn=128, dmid=128)
    # print('Network: ', transformer)
    for epoch in tqdm(range(nEpochs)):
        for data_idx, source_target in tqdm(enumerate(zip(source, target))):
        # for data_idx, source_target in enumerate(zip(source, target)):
            z = source_target[0]
            x = source_target[1]     
            P = transformer(x, z)

            loss = -torch.mean(torch.log(torch.clip(P[x[1:x.shape[0]], range(x.shape[0] - 1)], 1e-9)))
            # print(f'Loss value at step {data_idx}: {loss.item()}')
            if data_idx % 500 == 0:
                print(f'Loss value at step {data_idx}: {loss.item()}')
                print('P[x[1:x.shape[0]-1], range(x.shape[0] - 2)]', P[x[1:x.shape[0]-1], range(x.shape[0] - 2)])
                print(f'Saving Model after epoch {epoch + 1}')
                torch.save(transformer.state_dict(), f'{saved_model_path}_epoch_{epoch}')
            
            loss.backward()
            with torch.no_grad():
                for param in transformer.parameters():
                    param -= lrate * param.grad
                transformer.zero_grad()
        indx = list(range(len(source)))
        random.shuffle(indx)
        source = [source[i] for i in indx]
        target = [target[i] for i in indx]

        torch.save(transformer.state_dict(), saved_model_path)

################################################
# Algorithm 12
################################################
def ETraining(nEpochs=10, lrate=1e-3, p_mask=0.5, saved_model_path='ETraining_model.pth'):
    data, char2num = load_decoder_dataset()
    print('*** dataset size:', len(data))
    print('*** vocabulary size:', len(char2num))
    eTransformer = ETransformer(L=2, H=2, de=512, dmlp=2048, df=128, Nv=len(char2num) + 1, lmax=max_len + 2, dattn=128, dmid=128)
    mask_token = len(char2num)
    for epoch in range(nEpochs):
        for data_idx, x in tqdm(enumerate(data)):
            mask_indices =  np.random.binomial(1, p_mask, x.shape[0])
            mask_indices = np.where(mask_indices)
            masked_x = x.clone().detach()
            masked_x[mask_indices] = mask_token
            P = eTransformer(masked_x)
            loss = -torch.mean(torch.log(P[x[mask_indices], mask_indices]))
            if data_idx % 500 == 0:
                print(f'Loss value at step {data_idx}: {loss.item()}')            
            loss.backward()
            with torch.no_grad():
                for param in eTransformer.parameters():
                    param -= lrate * param.grad
                eTransformer.zero_grad()
        np.random.shuffle(data)
        print(f'Saving Model after epoch {epoch + 1}')
        torch.save(eTransformer.state_dict(), saved_model_path)

ETraining(lrate=1e-3, nEpochs=2)


################################################
# Algorithm 13
################################################
def DTraining(nEpochs=10, lrate=1e-4, saved_model_path='DTraining_model.pth'):
    data, char2num = load_decoder_dataset()
    print('*** dataset size:', len(data))
    print('*** vocabulary size:', len(char2num))
    # L, H, de, dmlp, Nv, lmax, dattn, dmid):
    # TODO: check if I need start and end characters
    
    transformer = DTransformer(L=2, H=2, de=512, dmlp=2048, Nv=len(char2num), lmax=max_len + 2, dattn=128, dmid=128)

    for epoch in tqdm(range(nEpochs)):
        for data_idx, x in tqdm(enumerate(data)):     
            P = transformer(x)
            loss = -torch.mean(torch.log(torch.clip(P[x[1:x.shape[0]], range(x.shape[0] - 1)], 1e-9)))
            if data_idx % 500 == 0:
                print(f'Loss value at step {data_idx}: {loss.item()}')
                print('P[x[1:x.shape[0]-1], range(x.shape[0] - 2)]', P[x[1:x.shape[0]-1], range(x.shape[0] - 2)])
                print(f'Saving Model after epoch {epoch + 1}')
                torch.save(transformer.state_dict(), f'{saved_model_path}_epoch_{epoch}')            
            loss.backward()
            with torch.no_grad():
                for param in transformer.parameters():
                    param -= lrate * param.grad
                transformer.zero_grad()
        random.shuffle(data)      
        torch.save(transformer.state_dict(), saved_model_path)

# DTraining(lrate=1e-3, nEpochs=2)
# Wp = positional_embedding(512, 2048).astype(dtype=np.float32)
# visualize_pos_embedding(Wp)


# EDTraining(lrate=1e-3, nEpochs=3)
###############################################
# Algorithm 14
###############################################
def DInference(x, lgen, transformer, char2num):
    l = x.shape[0]
    y = torch.tensor([char2num['S']])
    t = 1
    for i in range(lgen):
        P = transformer(x)
        p = P[:, l + i - 1]
        y = torch.tensor([np.random.choice(len(char2num), p=p.detach().numpy() ** (1 / t))])
        x = torch.cat([x, y])
    return x[l:]

################################################
# Algorithm 15
################################################
def EDInference(z, transformer, char2num):
    x = torch.tensor([char2num['S']], dtype=torch.int)
    y = torch.tensor([char2num['S']])
    end_token = char2num['E']
    t = 1
    while y[0] != end_token and len(x) < max_len:
        P = transformer(x, z)
        p = P[:, x.shape[0] - 1]
        y = torch.tensor([np.random.choice(len(char2num), p=p.detach().numpy() ** (1 / t))])
        x = torch.cat([x, y])
    return x

# run EDInference
# _, _, char2num = loadSeq2SeqDataset()
# z = 'i agree.'
# z = torch.tensor([char2num['S']] + \
#     [char2num[achar] for achar in z] + \
#     [char2num['E']])
# num2char = ['0'] * (len(char2num))
# for achar, num in char2num.items():
#     num2char[num] = achar

# transformer = EDTransformer(Lenc=2, Ldec=2, H=2, de=512, dmlp=2048, Nv=len(char2num), lmax=max_len + 2, dattn=128, dmid=128)

# # # ntokens = ord('z') - ord('a') + 4
# # We = np.identity(len(vocabulary) + 4, dtype=np.float32)
# # Wu = np.linalg.inv(We)
# # Wu = torch.from_numpy(Wu)
# # Wp = positional_embedding(We.shape[0], max_len).astype(dtype=np.float32)
# # transformer = EDTransformer(Lenc=2, Ldec=2, H=2, de=We.shape[0], dmlp=We.shape[0], df=20, We=We, Wp=Wp, Wu=Wu, dattn=20, dmid=20)
# transformer.load_state_dict(torch.load('EDTraining_model.pth'))
# transformer.eval()
# x = EDInference(z, transformer, char2num)
# # print('num2char_x', num2char_x)
# print(f'Text in Spanish: {str([num2char[n]  for n in x])}')

# run DInference
# _, char2num = load_decoder_dataset()
# x = '' # 'he had'
# x = torch.tensor([char2num[START_CHAR]] + \
#     [char2num[achar] for achar in x])
# num2char = ['0'] * (len(char2num))
# for achar, num in char2num.items():
#     num2char[num] = achar

# transformer = DTransformer(L=2, H=2, de=512, dmlp=2048, Nv=len(char2num), lmax=max_len + 2, dattn=128, dmid=128)

# # # ntokens = ord('z') - ord('a') + 4
# # We = np.identity(len(vocabulary) + 4, dtype=np.float32)
# # Wu = np.linalg.inv(We)
# # Wu = torch.from_numpy(Wu)
# # Wp = positional_embedding(We.shape[0], max_len).astype(dtype=np.float32)
# # transformer = EDTransformer(Lenc=2, Ldec=2, H=2, de=We.shape[0], dmlp=We.shape[0], df=20, We=We, Wp=Wp, Wu=Wu, dattn=20, dmid=20)
# transformer.load_state_dict(torch.load('DTraining_model.pth_epoch_1'))
# transformer.eval()
# x = DInference(x, 20, transformer, char2num)
# # print('num2char_x', num2char_x)
# print(f"Book text: {''.join([num2char[n]  for n in x])}")

