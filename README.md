# Formal algorithms for Transformers
Implementation of [Formal Algorithms for Transformers](https://arxiv.org/abs/2207.09238).
Operation and variable names follow the paper's notation.
While the paper defines the algorithms as processing 
one single input example at a time, this 
implementation adds a batch dimension to the networks in order to process 
multiple examples at a time, and accelerate training speed.

- [x] Encoder-Decoder Transformer (e.g. Attention is All You Need)
- [x] Decoder Transformer (e.g. GPT, Gopher, Gato)
- [x] Encoder Transformer (e.g. BERT)
- [x] Decoder transformer trained model checkpoint
- [ ] Encoder-Decoder transformer trained model checkpoint

  

# Running Pre-trained Models
This repository includes pre-trained Deecoder  
(`DTransformer.pth`) and Encoder-Decoder (`EDTransformer.pth`)
transformers. The first one was trained on a language translation task,
and the second one was trained on the text generation
task used by Andrej Karpathy in his [Chat GPT video lecture](https://github.com/karpathy/ng-video-lecture). 


## Decoder Transformer
The `DTransformer.pth` model was trained on a Shakespeare book 
text generation task, over 150 epochs, and maximum sequence length 256 (`--max_len 256`).
Training took about 10min on a NVIDIA A10 GPU. 

To run inference on this model, 
use `--prompt` parameter to input your prompt. 
The model generates new text in the book's style. For example, running:
```
python train_eval.py --max_len 256 --model_path DTransformer.pth --model_type DTransformer --mode eval  --prompt "." --new_text_len 500
```
generated the following output:

```
Citan:
The wellown: of rothen; but wellowenced.

LARDIUS:
A widou, with ladying:
Hogd flifer's in man won, aminshand you
Which suntrabrroud.
Then so shall'd the it thyself,
The lse, thring bliaght the thim time.

MENENETY:
Kither it no him.
TRich to my soubeld; are are ondeedw
DUn, Youllaious!

APly York:

VISHARUS:


IShounh, I Maringhingner!


TRAP:

IV:
's orbew the olise!


FRichield, of I'laste 'Dunis:


Thigh:

ISp onguits bullives,

Then oumege.



PRICH:

Why!  Plovose of 'ICllivowane
```
Note that the output might often 
change, depending on the temperature paramater `t` 
in `Algorithm 14` (default value is 1).

## Encoder-Decoder Transformer
The `EDTransformer.pth` model was trained on an English-to-Spanish translation dataset,
for 40 epochs, and with maximum sequence length 64 (`--max_len=64`). To do inference on this model,
run the `train_eval.py` script with the `--mode` parameter to indicate the inference/evaluation stage,
and `--source` to input the text to be translated. For instance, at the moment of writing this readme, running:
```
python train_eval.py --max_len 64 --model_path EDTraining_model.pth --mode eval --source 'Hello world!'
```
generated: `...`. 


# Training new models
By default, running 
```
python train_eval.py
```
trains an decoder transformer. Use the `--model_type` parameter to 
select the other models. See the `train_eval.py` script for additional 
paramaters and model architecture configuration.

Loss values, model's graph, and gradient:weight rations can be seen 
using tensorboard on a folder created by the script with the same name as the 
model's (defined by `--model_path` paramater).

To replicate the `DTransformer.pth` training, run:
```
nEpochs=150
model_type='DTransformer'
max_len=256
bs=64
lr=9e-2
model_name="DTransformer_nlayers-heads_3_lr_"$lr"_max_len_"$max_len"_bs_"$bs".pth"
python train_eval.py --lr $lr --nEpochs $nEpochs --model_type $model_type \
     --batch_size $bs --max_len $max_len --model_path $model_name
```