import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import random
from torch.utils.tensorboard import SummaryWriter
import os
from torch.nn import functional as F

_eps = 1e-5
device = 'cuda' if torch.cuda.is_available() else 'cpu' # 'mps' 

torch.manual_seed(113)

def random_init(shape, scale=0.02):
    '''Returns a tensor with shape and values sampled from a normal distribution with mean 0 and standard deviation scale.'''
    return torch.randn(shape, device=device) * scale

def paramater(init_func, *args, **kwargs):
    '''Returns a torch.nn.Parameter with data initialized with init_func.'''
    return nn.Parameter(init_func(*args, **kwargs))

def paramater_list(n, init_func, *args, **kwargs):
    '''Returns a ParameterList of n torch.nn.Parameter with data initialized with init_func.'''
    return nn.ParameterList([paramater(init_func, *args, **kwargs) for _ in range(n)])

def module_list(n, module, *args, **kwargs):
    '''Returns a ModuleList of n modules.'''
    return nn.ModuleList([module(*args, **kwargs) for _ in range(n)])

def log_loss(running_loss, batch_size, epoch, step, dataset_size, writer, transformer, model_path, lrate):
    avg_loss = running_loss / (step + 1)
    global_step = epoch * (dataset_size // batch_size) + step
    print(f'Avg. training loss at global step={global_step}: {avg_loss}')
    writer.add_scalar('Avg. training loss', avg_loss, global_step)
    torch.save(transformer.state_dict(), model_path)
    with torch.no_grad():
        for name, param in transformer.named_parameters():                                   
            update_scale = torch.linalg.norm(lrate * param.grad)
            weight_scale = torch.linalg.norm(param)
            writer.add_scalar(f'Update:Weight ratio/{name}', update_scale / weight_scale, global_step)

################################################
# Algorithm 4
################################################
class Attention(nn.Module):
    def __init__(self, dx, dz, dattn, dout):
        super().__init__()
        self.Wq = paramater(random_init, (dattn, dx))
        self.bq = paramater(torch.zeros, (dattn, 1))
        self.Wk = paramater(random_init, (dattn, dz)) 
        self.bk = paramater(torch.zeros, (dattn, 1))
        self.Wv = paramater(random_init, (dout, dz))
        self.bv = paramater(torch.zeros, (dout, 1))
        self.dattn_sqrt = math.sqrt(dattn)
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, X, Z, mask=None):
        """
        Parameters:
            X: Primary (query) sequence. X.shape = (bs, dx, lx)
            Z: Context sequence. Z.shape = (bs, dz, lz)
            mask: boolean mask, mask[i,j] == True makes token in Z[i] influence
            token X[j].
        Output:
            tensor representation of X with Z as context.
        """
        q = self.Wq @ X  + self.bq # bs x dattn x lx
        k = self.Wk @ Z + self.bk # bs x dattn x lz
        v = self.Wv @ Z + self.bv  # bs x dout x lz        
        score = k.transpose(-2, -1) @ q # bs x lz x lx
        if mask is not None:
            score = score.masked_fill(mask==0, float('-inf'))

        return v @ self.softmax(score / self.dattn_sqrt) # bs x dout x lx

################################################
# Algorithm 5
################################################
class MHAttention(nn.Module):
    def __init__(self, dx, dz, dattn, dout, dmid, H):
        super().__init__()
        self.attention = module_list(H, Attention, dx, dz, dattn, dmid)
        self.Wo = paramater(random_init, (dout, dmid * H))
        self.bo = paramater(torch.zeros, (dout, 1))

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
        Y = torch.cat(Y, dim=-2) # bs x (dmid * H) x lx
        return self.Wo @ Y + self.bo # bs x dout x lx


################################################
# Algorithm 6:
################################################
class Layer_norm(nn.Module):
    def __init__(self, de, beta=None):
        super().__init__()
        self.lamb = paramater(torch.ones, (de, 1))
        self.beta = paramater(torch.zeros, (de, 1)) if beta is None else beta
    
    def forward(self, e):
        m = e.mean(dim=-2, keepdim=True)
        v = e.var(dim=-2, keepdim=True)
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
        self.Wp = torch.nn.Embedding(lmax, de)
        torch.nn.init.normal_(self.We.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.Wp.weight, mean=0.0, std=0.02)
        self.Lenc = Lenc
        self.Ldec = Ldec
        self.encoder_MHeadAttentionLayers = module_list(Lenc, MHAttention, de, de, dattn, de, dmid, H)
        self.encoder_first_layer_norms = module_list(Lenc, Layer_norm, de)
        self.encoder_second_layer_norms = module_list(Lenc, Layer_norm, de)
        self.Wmlps1 = paramater_list(Lenc, random_init, (dmlp, de))
        self.Wmlps2 = paramater_list(Lenc, random_init, (de, dmlp))
        self.bmlps1 = paramater_list(Lenc, torch.zeros, (dmlp, 1))
        self.bmlps2 = paramater_list(Lenc, torch.zeros, (de, 1))
        self.decoder_MHeadAttentionLayers = module_list(Ldec, MHAttention, de, de, dattn, de, dmid, H)
        self.decoder_first_layer_norms = module_list(Ldec, Layer_norm, de)
        self.decoder_second_layer_norms = module_list(Ldec, Layer_norm, de)
        self.decoder_third_layer_norms = module_list(Ldec, Layer_norm, de)
        self.Wmlps3 = paramater_list(Lenc, random_init, (dmlp, de))
        self.Wmlps4 = paramater_list(Lenc, random_init, (de, dmlp))
        self.bmlps3 = paramater_list(Lenc, torch.zeros, (dmlp, 1))
        self.bmlps4 = paramater_list(Lenc, torch.zeros, (de, 1))
        self.Wu = paramater(random_init, (Nv, de))
        self.relu=nn.ReLU()
        self.register_buffer("unidirectional_attention_mask", torch.triu(torch.ones((lmax, lmax))))

    def forward(self, x, z):
        lz = z.shape[-1]
        Z = self.We(z).transpose(-2,-1) + self.Wp(torch.arange(lz, device=device)).transpose(-2,-1)
        for l in range(self.Lenc):
            Z = Z + self.encoder_MHeadAttentionLayers[l](Z, Z, mask=None)
            Z = self.encoder_first_layer_norms[l](Z)
            Z = Z + self.Wmlps2[l] @ self.relu(self.Wmlps1[l] @ Z + self.bmlps1[l]) + self.bmlps2[l]
            Z = self.encoder_second_layer_norms[l](Z)

        lx = x.shape[-1]
        X = self.We(x).transpose(-2,-1) + self.Wp(torch.arange(lx, device=device)).transpose(-2,-1)
        for l in range(self.Ldec):
            X = X + self.decoder_MHeadAttentionLayers[l](X, X, mask=self.unidirectional_attention_mask[:lx, :lx])
            X = self.decoder_first_layer_norms[l](X)
            X = X + self.decoder_MHeadAttentionLayers[l](X, Z, mask=None)
            X = self.decoder_second_layer_norms[l](X)
            X = X + self.Wmlps4[l] @ self.relu(self.Wmlps3[l] @ X + self.bmlps3[l]) + self.bmlps4[l]
            X = self.decoder_third_layer_norms[l](X)
        return nn.functional.softmax(self.Wu @ X, dim=-2)


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
        self.Wp = torch.nn.Embedding(lmax, de)
        torch.nn.init.normal_(self.We.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.Wp.weight, mean=0.0, std=0.02)
        self.L = L
        self.mHeadAttentionLayers = module_list(L, MHAttention, de, de, dattn, de, dmid, H)
        self.first_layer_norms = module_list(L, Layer_norm, de)
        self.second_layer_norms = module_list(L, Layer_norm, de)
        self.Wmlps1 = paramater_list(L, random_init, (dmlp, de))
        self.Wmlps2 = paramater_list(L, random_init, (de, dmlp))
        self.bmlps1 = paramater_list(L, torch.zeros, (dmlp, 1))
        self.bmlps2 = paramater_list(L, torch.zeros, (de, 1))
        self.Wf = paramater(random_init, (df, de))
        self.bf = paramater(torch.zeros, (df, 1))
        self.final_layer_norm = Layer_norm(df)
        self.Wu = paramater(random_init, (Nv, df))
        self.gelu = nn.GELU()
        
    def forward(self, x):
        lx = x.shape[-1]
        X = self.We(x).transpose(-2,-1) + self.Wp(torch.arange(lx, device=device)).transpose(-2,-1)
        for l in range(self.L):
            X = X + self.mHeadAttentionLayers[l](X, X, mask=None)
            X = self.first_layer_norms[l](X)
            X = X + self.Wmlps2[l] @ self.gelu(self.Wmlps1[l] @ X + self.bmlps1[l]) + self.bmlps2[l]
            X = self.second_layer_norms[l](X)
        X = self.gelu(self.Wf @ X + self.bf)
        X = self.final_layer_norm(X)
        return nn.functional.softmax(self.Wu @ X, dim=-2)

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
        self.Wp = torch.nn.Embedding(lmax, de)
        torch.nn.init.normal_(self.We.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.Wp.weight, mean=0.0, std=0.02)
        self.L = L
        self.Wmlps1 = paramater_list(L, random_init, (dmlp, de))
        self.Wmlps2 = paramater_list(L, random_init, (de, dmlp))
        self.bmlps1 = paramater_list(L, torch.zeros, (dmlp, 1))
        self.bmlps2 = paramater_list(L, torch.zeros, (de, 1))
        self.mHeadAttentionLayers = module_list(L, MHAttention, de, de, dattn, de, dmid, H)
        self.first_layer_norms = module_list(L, Layer_norm, de)
        self.second_layer_norms = module_list(L, Layer_norm, de)
        self.third_layer_norm = Layer_norm(de)
        self.Wu = paramater(random_init, (Nv, de))
        self.gelu=nn.GELU()
        self.register_buffer("unidirectional_attention_mask", torch.triu(torch.ones((lmax, lmax))))
                

    def forward(self, x):
        lx = x.shape[-1]
        X = self.We(x).transpose(-2,-1) + self.Wp(torch.arange(lx, device=device)).transpose(-2,-1)
        for l in range(self.L):
            X = self.first_layer_norms[l](X)
            X = X + self.mHeadAttentionLayers[l](X, X, self.unidirectional_attention_mask[:lx, :lx])
            X = self.second_layer_norms[l](X)
            X = X + self.Wmlps2[l] @ self.gelu(self.Wmlps1[l] @ X + self.bmlps1[l]) + self.bmlps2[l]
        X = self.third_layer_norm(X)
        return nn.functional.softmax(self.Wu @ X, dim=-2)


################################################
# Algorithm 11
################################################
def EDTraining(transformer,
               source,
               target,
               nEpochs=10, 
               lrate=1e-4, 
               model_path='EDTraining_model.pth'):
    transformer.to(device)
    writer = SummaryWriter(f'{os.path.splitext(model_path)[0]}_train')
    writer.add_graph(transformer, (source[0].to(device).view((1, -1)), target[0].to(device).view((1, -1))))
    running_loss = 0
    for epoch in tqdm(range(nEpochs)):
        for step, (z, x) in tqdm(enumerate(zip(source, target))):
            P = transformer(x.to(device).view((1, -1)), z.to(device).view(1, -1))[0]
            loss = -torch.mean(torch.log(P[x[1:x.shape[0]], range(x.shape[0] - 1)]))
            running_loss += loss.item()         
            loss.backward()
            if step % 100 == 0:
                log_loss(running_loss, 1, epoch, step, len(source), writer, transformer, model_path, lrate)
            with torch.no_grad():
                for param in transformer.parameters():
                    param -= lrate * param.grad
                transformer.zero_grad()
        indx = list(range(len(source)))
        random.shuffle(indx)
        source = [source[i] for i in indx]
        target = [target[i] for i in indx]
        torch.save(transformer.state_dict(), model_path)

################################################
# Algorithm 12
################################################
def ETraining(eTransformer, dataset, mask_token, nEpochs=10, lrate=1e-3, p_mask=0.5, model_path='ETraining_model.pth', batch_size=16):
    eTransformer.to(device)
    writer = SummaryWriter(f'{os.path.splitext(model_path)[0]}_train')
    writer.add_graph(eTransformer, dataset[0])
    running_loss = 0
    for epoch in range(nEpochs):
        for step, end_ix in tqdm(enumerate(range(batch_size, dataset.shape[0], batch_size))):
            x = dataset[end_ix - batch_size: end_ix].to(device)
            mask_indices =  torch.bernoulli(torch.ones_like(x) * p_mask)            
            masked_x = torch.where(mask_indices == 1, mask_token, x.clone().detach())
            P = eTransformer(masked_x)            
            idx = torch.nonzero(mask_indices)
            loss = -torch.mean(torch.log(P[idx[:, 0], x[idx[:, 0], idx[:, 1]], idx[:, 1]]))
            running_loss += loss.item()
            loss.backward()
            if step % 100 == 0:
                log_loss(running_loss, batch_size, epoch, step, dataset.shape[0], writer, eTransformer, model_path, lrate)
            with torch.no_grad():
                for param in eTransformer.parameters():
                    param -= lrate * param.grad
                eTransformer.zero_grad()            
        np.random.shuffle(dataset)
        print(f'Saving Model after epoch {epoch + 1}')
        torch.save(eTransformer.state_dict(), model_path)


################################################
# Algorithm 13
################################################
def DTraining(dTransformer, dataset, nEpochs=10, lrate=1e-4, model_path='DTraining_model.pth', batch_size = 16):
    dTransformer.to(device)
    writer = SummaryWriter(f'{os.path.splitext(model_path)[0]}_train')
    writer.add_graph(dTransformer, dataset[0:1].to(device))
    running_loss = 0
    lx = dataset.shape[1]
    batch_ix = torch.tensor([[i] * (lx - 1) for i in range(batch_size)], dtype=torch.long, device=device).flatten()
    pos_ix = torch.tensor(list(range(lx - 1)) * batch_size, dtype=torch.long, device=device)
    for epoch in tqdm(range(nEpochs)):
        for step, end_ix in tqdm(enumerate(range(batch_size, dataset.shape[0], batch_size))):
            x = dataset[end_ix - batch_size: end_ix].to(device)     
            P = dTransformer(x)
            loss = -torch.mean(torch.log(P[batch_ix, x[:, 1:].flatten(), pos_ix]))
            running_loss += loss.item()
            loss.backward()
            if step % 100 == 0:
                log_loss(running_loss, batch_size, epoch, step, dataset.shape[0], writer, dTransformer, model_path, lrate)
            with torch.no_grad():
                for param in dTransformer.parameters():                                   
                    param -= lrate * param.grad
                dTransformer.zero_grad()
        torch.save(dTransformer.state_dict(), model_path)
        running_loss = 0
        dataset = dataset[torch.randperm(dataset.shape[0])]
    writer.close()


###############################################
# Algorithm 14
###############################################
def DInference(x, lgen, transformer, decoder, t=1, lmax=20):
    assert lgen > 0, f'lgen must be > 0, but got {lgen}.'
    l = x.shape[0]
    x = x.to(device)
    transformer.to(device)
    print("New text: ", end="")
    for _ in range(lgen):
        # add batch axis and 
        # use x[:, -lmax:] to handle sequences longer than lmax
        inp = x[None, -lmax:].to(device)
        P = transformer(inp)[0]
        p = P[:, -1]
        y = torch.multinomial(p ** (1 / t), num_samples=1)
        x = torch.cat([x, y])
        print(decoder(y.cpu().numpy())[0], end="")
    print()
    return x.cpu().numpy()[l:]


################################################
# Algorithm 15
################################################
def EDInference(z, transformer, bos_token, eos_token, t=1, lmax=20):
    x = torch.tensor([bos_token], dtype=torch.long, device=device)
    y = torch.tensor([bos_token], dtype=torch.long, device=device)
    z = z.to(device)
    transformer.to(device)
    while y[0] != eos_token:
        P = transformer(x[None, -lmax:], z[None, -lmax:])[0]
        p = P[:, -1]
        y = torch.multinomial(p ** (1 / t), num_samples=1)
        x = torch.cat([x, y])
    return x.cpu()
