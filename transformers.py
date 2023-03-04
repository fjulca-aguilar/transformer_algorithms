import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import random
from torch.utils.tensorboard import SummaryWriter

_eps = 1e-5
dtype = torch.float
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
            inf = 1000
            score = score.masked_fill(mask==0, -inf)

        return v @ self.softmax(score / self.dattn_sqrt) # dout x lx

################################################
# Algorithm 5
################################################
class MHAttention(nn.Module):
    def __init__(self, dx, dz, dattn, dout, dmid, H):
        super().__init__()
        self.attention = nn.ModuleList([Attention(dx, dz, dattn, dmid) for _ in range(H)])
        self.Wo = nn.Parameter(torch.randn((dout, dattn * H), dtype=dtype) / math.sqrt(dout))
        self.bo = nn.Parameter(torch.zeros((dout, 1), dtype=dtype))

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
        self.lamb = torch.nn.Parameter(torch.ones((de, 1), dtype=dtype))
        self.beta = torch.nn.Parameter(torch.zeros((de, 1), dtype=dtype)) if beta is None else beta
    
    def forward(self, e):
        m = e.mean(dim=0, keepdim=True)
        v = e.var(dim=0, keepdim=True)
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
        self.Wu = nn.Parameter(torch.randn((Nv, de), dtype=dtype) / math.sqrt(Nv)) # 
        
        for _ in range(Lenc):
            self.encoder_MHeadAttentionLayers.append(MHAttention(de, de, dattn, de, dmid, H))
            self.encoder_first_layer_norms.append(Layer_norm(de))
            self.encoder_second_layer_norms.append(Layer_norm(de))
            self.Wmlps1.append(torch.nn.Parameter(torch.randn((dmlp, de), dtype=dtype) / math.sqrt(dmlp)))
            self.Wmlps2.append(torch.nn.Parameter(torch.randn((de, dmlp), dtype=dtype) / math.sqrt(de)))
            self.bmlps1.append(torch.nn.Parameter(torch.zeros((dmlp, 1), dtype=dtype)))
            self.bmlps2.append(torch.nn.Parameter(torch.zeros((de, 1), dtype=dtype)))

        for _ in range(Ldec):
            self.decoder_MHeadAttentionLayers.append(MHAttention(de, de, dattn, de, dmid, H))
            self.decoder_first_layer_norms.append(Layer_norm(de))
            self.decoder_second_layer_norms.append(Layer_norm(de))
            self.decoder_third_layer_norms.append(Layer_norm(de))
            self.Wmlps3.append(torch.nn.Parameter(torch.randn((dmlp, de), dtype=dtype) / math.sqrt(dmlp)))
            self.Wmlps4.append(torch.nn.Parameter(torch.randn((de, dmlp), dtype=dtype) / math.sqrt(de)))
            self.bmlps3.append(torch.nn.Parameter(torch.zeros((dmlp, 1), dtype=dtype)))
            self.bmlps4.append(torch.nn.Parameter(torch.zeros((de, 1), dtype=dtype)))
        self.relu=nn.ReLU()
        self.register_buffer("unidirectional_attention_mask", torch.triu(torch.ones((lmax, lmax))))

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
        self.Wf = torch.nn.Parameter(torch.randn((df, de), dtype=dtype) / math.sqrt(df))
        self.bf = torch.nn.Parameter(torch.zeros((df, 1), dtype=dtype) / math.sqrt(df))
        self.final_layer_norm = Layer_norm(df)
        self.Wu = nn.Parameter(torch.randn((Nv, df), dtype=dtype) / math.sqrt(Nv))
        self.gelu = nn.GELU()
        
        for _ in range(L):
            self.mHeadAttentionLayers.append(MHAttention(de, de, dattn, de, dmid, H))
            self.first_layer_norms.append(Layer_norm(de))
            self.second_layer_norms.append(Layer_norm(de))
            self.Wmlps1.append(torch.nn.Parameter(torch.randn((dmlp, de), dtype=dtype) / math.sqrt(dmlp)))
            self.Wmlps2.append(torch.nn.Parameter(torch.randn((de, dmlp), dtype=dtype) / math.sqrt(de)))
            self.bmlps1.append(torch.nn.Parameter(torch.zeros((dmlp, 1), dtype=dtype)))
            self.bmlps2.append(torch.nn.Parameter(torch.zeros((de, 1), dtype=dtype)))

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
    t = torch.arange(0, lmax, dtype=dtype).reshape((1, -1))
    angles = lmax_exp @ t
    Wp = torch.zeros((de, lmax), dtype=dtype)
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
        self.Wu = nn.Parameter(torch.randn((Nv, de), dtype=dtype) / math.sqrt(Nv)) # 
        
        for _ in range(L):
            self.mHeadAttentionLayers.append(MHAttention(de, de, dattn, de, dmid, H))
            self.first_layer_norms.append(Layer_norm(de))
            self.second_layer_norms.append(Layer_norm(de))
            self.Wmlps1.append(torch.nn.Parameter(torch.randn((dmlp, de), dtype=dtype) / math.sqrt(dmlp)))
            self.Wmlps2.append(torch.nn.Parameter(torch.randn((de, dmlp), dtype=dtype) / math.sqrt(de)))
            self.bmlps1.append(torch.nn.Parameter(torch.zeros((dmlp, 1), dtype=dtype)))
            self.bmlps2.append(torch.nn.Parameter(torch.zeros((de, 1), dtype=dtype)))
        self.gelu=nn.GELU()
        self.register_buffer("unidirectional_attention_mask", torch.triu(torch.ones((lmax, lmax), dtype=dtype)) > 0)

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
def EDTraining(transformer,
               source,
               target,
               nEpochs=10, 
               lrate=1e-4, 
               model_path='EDTraining_model.pth'):
    writer = SummaryWriter(f'{model_path}_train')
    writer.add_graph(transformer, (source[0], target[0]))
    running_loss = 0
    for epoch in tqdm(range(nEpochs)):
        for data_idx, (z, x) in tqdm(enumerate(zip(source, target))):     
            P = transformer(x, z)
            loss = -torch.sum(torch.log(torch.clip(P[x[1:x.shape[0]], range(x.shape[0] - 1)], 1e-9)))
            running_loss += loss.item()         
            loss.backward()
            with torch.no_grad():
                for param in transformer.parameters():
                    param -= lrate * param.grad
                transformer.zero_grad()
            if (data_idx + 1) % 500 == 0:
                writer.add_scalar('Training loss', running_loss / 500., epoch * len(source) + data_idx + 1)
                running_loss = 0
                torch.save(transformer.state_dict(), model_path)
        indx = list(range(len(source)))
        random.shuffle(indx)
        source = [source[i] for i in indx]
        target = [target[i] for i in indx]
        torch.save(transformer.state_dict(), model_path)

################################################
# Algorithm 12
################################################
def ETraining(eTransformer, dataset, mask_token, nEpochs=10, lrate=1e-3, p_mask=0.5, model_path='ETraining_model.pth'):
    writer = SummaryWriter(f'{model_path}_train')
    writer.add_graph(eTransformer, dataset[0])
    running_loss = 0
    for epoch in range(nEpochs):
        for data_idx, x in tqdm(enumerate(dataset)):
            mask_indices =  np.random.binomial(1, p_mask, x.shape[0])
            mask_indices = np.where(mask_indices)
            masked_x = x.clone().detach()
            masked_x[mask_indices] = mask_token
            P = eTransformer(masked_x)
            loss = -torch.sum(torch.log(P[x[mask_indices], mask_indices]))
            running_loss += loss.item()
            loss.backward()
            with torch.no_grad():
                for param in eTransformer.parameters():
                    param -= lrate * param.grad
                eTransformer.zero_grad()
            if (data_idx + 1) % 500 == 0:
                writer.add_scalar('Training loss', running_loss / 500., epoch * len(dataset) + data_idx + 1)
                running_loss = 0
                torch.save(eTransformer.state_dict(), model_path)  
        np.random.shuffle(dataset)
        print(f'Saving Model after epoch {epoch + 1}')
        torch.save(eTransformer.state_dict(), model_path)


################################################
# Algorithm 13
################################################
def DTraining(dTransformer, dataset, nEpochs=10, lrate=1e-4, model_path='DTraining_model.pth'):
    writer = SummaryWriter(f'{model_path}_train')
    writer.add_graph(dTransformer, dataset[0])
    running_loss = 0
    for epoch in tqdm(range(nEpochs)):
        for data_idx, x in tqdm(enumerate(dataset)):     
            P = dTransformer(x)
            loss = -torch.sum(torch.log(torch.clip(P[x[1:x.shape[0]], range(x.shape[0] - 1)], 1e-9)))
            running_loss += loss.item()            
            loss.backward()
            with torch.no_grad():
                for param in dTransformer.parameters():
                    param -= lrate * param.grad
                dTransformer.zero_grad()
            if (data_idx + 1) % 500 == 0:
                writer.add_scalar('Training loss', running_loss / 500., epoch * len(dataset) + data_idx + 1)
                running_loss = 0
                torch.save(dTransformer.state_dict(), model_path)
        random.shuffle(dataset)      
        torch.save(dTransformer.state_dict(), model_path)
    writer.close()


###############################################
# Algorithm 14
###############################################
def DInference(x, lgen, transformer, t=1, max_len=20):
    assert lgen > 0, f'lgen must be > 0, but got {lgen}.'
    l = x.shape[0]
    for _ in range(lgen):
        # use x[-(max_len - 1):] to handle sequences longer than max_len
        P = transformer(x[-(max_len - 1):])
        p = P[:, -1]
        y = torch.multinomial(p ** (1 / t), num_samples=1)
        x = torch.cat([x, y])
    return x[l:]


################################################
# Algorithm 15
################################################
def EDInference(z, transformer, bos_token, eos_token, t=1, max_len=20):
    x = torch.tensor([bos_token], dtype=torch.int)
    y = torch.tensor([bos_token], dtype=torch.int)
    while y[0] != eos_token and len(x) < max_len:
        P = transformer(x, z)
        p = P[:, -1]
        y = torch.multinomial(p ** (1 / t), num_samples=1)
        x = torch.cat([x, y])
    return x
