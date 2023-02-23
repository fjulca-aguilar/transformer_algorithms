import argparse
from transformers import *

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--mode', type=str, default='train',
                    choices=['train', 'eval'])
parser.add_argument('--model_type', type=str, default='EDTransformer',
                    choices=['EDTransformer', 'DTransformer', 'ETransformer'])
parser.add_argument('--lrate', type=float,
                    default=1e-3,
                    help='Learning rate.')
parser.add_argument('--nEpochs', type=int,
                    default=3,
                    help='Number of training epochs.')
parser.add_argument('--model_path', type=str, default='EDTraining_model.pth',
                    help='Path to trained model, to be loaded for evaluation.')
parser.add_argument('--source', type=str, default='Go.',
                    help='Source to be processed (translated) by EDTransformer.')
parser.add_argument('--prompt', type=str, default='He had',
                    help='Prompt to be processed (continued) by DTransformer.')

EDT_config ={
    'Lenc': 2, 
    'Ldec': 2, 
    'H': 2, 
    'de': 512, 
    'dmlp': 2048, 
    'dattn': 128, 
    'dmid': 128
}

DT_config ={
    'L': 2, 
    'H': 2, 
    'de': 512, 
    'dmlp': 2048, 
    'dattn': 128, 
    'dmid': 128
}

ET_config ={
    'L': 2, 
    'H': 2, 
    'de': 512, 
    'dmlp': 2048, 
    'dattn': 128, 
    'dmid': 128,
    'df': 128
}


def run_train(args):
    if args.model_type == 'EDTransformer':
        source, target, char2num = load_translation_dataset()
        transformer = EDTransformer(Nv=len(char2num), lmax=max_len, **EDT_config)
        EDTraining(transformer, source, target, lrate=1e-3, nEpochs=3)
    else:
        book, char2num, mask_token = load_book_dataset()
        if args.model_type == 'DTransformer':            
            transformer = DTransformer(Nv=len(char2num), lmax=max_len, **DT_config)
            DTraining(transformer, book, lrate=args.lrate, nEpochs=args.nEpochs)
        else:
            transformer = ETransformer(Nv=len(char2num), lmax=max_len, **ET_config)
            ETraining(transformer, book, lrate=args.lrate, mask_token=mask_token, nEpochs=args.nEpochs)


def run_eval(args):
    if args.model_type == 'EDTransformer':
        _, _, char2num = load_translation_dataset()
        transformer = EDTransformer(Nv=len(char2num), lmax=max_len, **EDT_config)
        z = torch.tensor([char2num[bos_char]] + \
            [char2num[achar] for achar in args.source.lower()] + \
            [char2num[eos_char]])
        num2char = ['0'] * (len(char2num))
        for achar, num in char2num.items():
            num2char[num] = achar

        transformer.load_state_dict(torch.load(args.model_path))
        transformer.eval()
        x = EDInference(z, transformer, char2num)
        print(f"Text in Spanish: {''.join([num2char[n]  for n in x])}")
    elif args.model_type == 'DTransformer':
        _, char2num, _ = load_book_dataset()
        transformer = DTransformer(Nv=len(char2num), lmax=max_len, **DT_config)
        x = torch.tensor([char2num[bos_char]] + \
            [char2num[achar] for achar in args.prompt.lower()])
        num2char = ['0'] * (len(char2num))
        for achar, num in char2num.items():
            num2char[num] = achar
        transformer.load_state_dict(torch.load('DTraining_model.pth'))
        transformer.eval()
        x = DInference(x, max_len - len(args.prompt.lower()), transformer, char2num)
        print(f"Book text: {''.join([num2char[n]  for n in x])}")
    else:
        print('No eval/inference algorithm for ', args.model_type)



if __name__ == '__main__':
    args = parser.parse_args()
    if args.mode == 'train':
        run_train(args)
    else:
        run_eval(args)
    