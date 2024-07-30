from model_encoder_run import RwkvEncoder
import torch
from typing import List
from transformers import AutoTokenizer
def create_empty_args():
    import argparse
    args = argparse.Namespace()
    args.my_pos_emb = 0
    args.pre_ffn = 0
    args.head_size_divisor = 8
    args.dropout = 0
    args.head_qk = 0
    args.ctx_len = 4096
    args.grad_cp = 0
    return args
def load_embedding_ckpt_and_parse_args(ckpt_file, args):
    try:
        with torch.no_grad():
            w = torch.load(ckpt_file, map_location='cpu') # load model to CPU first
            args.MODEL_NAME = ckpt_file.strip()
            #replace rwkvModel. to blanck
            for k in list(w.keys()):
                if 'rwkvModel.' in k:
                    w[k.replace('rwkvModel.', '')] = w[k]
                    del w[k]
            if not args.MODEL_NAME.endswith('.pth'):
                args.MODEL_NAME += '.pth'
            import gc
            gc.collect()
            n_embd = w['emb.weight'].shape[1]
            vocab_size = w['emb.weight'].shape[0]
            dim_att = w['blocks.0.att.key.weight'].shape[0] # note: transposed matrix
            dim_ffn = w['blocks.0.ffn.key.weight'].shape[0] # note: transposed matrix
            n_layer = 0
            keys = list(w.keys())
            version = 4
            n_head = 64
            for x in keys:
                layer_id = int(x.split('.')[1]) if ('blocks.' in x) else 0
                n_layer = max(n_layer, layer_id+1)
                if 'ln_x' in x:
                    version = max(5, version)
                if 'gate.weight' in x:
                    version = max(5.1, version)
                if int(version) == 5 and 'att.time_decay' in x:
                    n_head = w[x].shape[0]
                    if len(w[x].shape) > 1:
                        if w[x].shape[1] > 1:
                            version = max(5.2, version)
                if 'time_maa' in x:
                    version = max(6, version)
                if int(version) == 6 and 'time_faaaa' in x:
                    n_head = w[x].shape[0]

            head_size_a = dim_att // n_head
            args.n_embd = n_embd
            args.dim_att = dim_att
            args.dim_ffn = dim_ffn
            args.n_layer = n_layer
            args.version = version
            args.head_size_a = head_size_a
            args.vocab_size = vocab_size
            args.n_head = n_head
            args.n_att = dim_att
            return w
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None

def load_base_model(base_model):
    args = create_empty_args()
    w = load_embedding_ckpt_and_parse_args(base_model, args)
    print(args)
    args.emb_id = 151329
    args.pad_id = 151334
    args.mask_id = 151330
    model = RwkvEncoder(args)
    info = model.load_state_dict(w)
    print(info)
    return model


def tokenize_texts(texts :List[str],tokenizer :AutoTokenizer,emb_id = 151329, pad_id = 151334, mask_id = 151330):
    texts_idx = [tokenizer.encode(text,add_special_tokens=False) for text in texts]
    for text_idx in texts_idx:text_idx.append(emb_id)
    max_len = max([len(text_idx) for text_idx in texts_idx])
    texts_idx = [text_idx + [pad_id]*(max_len-len(text_idx)) for text_idx in texts_idx]
    
    #find the mask positions in texts_idx
    mask_positions = []
    for text_idx in texts_idx:
        mask_positions.append([i for i, x in enumerate(text_idx) if x == mask_id])
    return texts_idx,mask_positions
