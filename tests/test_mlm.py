if __name__ == '__main__':
    import sys
    import os
    os.environ["RWKV_CUDA_ON"] = '1'
    os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
    src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'src')
    sys.path.append(src_dir)
    import argparse
    from transformers import AutoTokenizer
    import torch
    parser = argparse.ArgumentParser("Test MLM model")
    parser.add_argument("--model_file",type=str,default='/media/yueyulin/data_4t/models/mlm/final/epoch_0_step_100000/RWKV-x060-MLM-ctx4096.pth.pth')
    parser.add_argument("--device",type=str,default='cpu')
    parser.add_argument("--dtype",type=str,default='float32',choices=['bfloat16','float16','float32'])
    args = parser.parse_args() 
    device = args.device
    if args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "float32":
        dtype = torch.float32
    from utilities import load_base_model,tokenize_texts
    model = load_base_model(args.model_file)
    model = model.to(device=device,dtype=dtype)
    print(model)
    model = torch.jit.script(model)
    tokenizer = AutoTokenizer.from_pretrained('THUDM/glm-4-9b-chat', trust_remote_code=True)
    print(tokenizer)
    
    ########################################################
    texts = ['巴黎是[MASK]的首都。',
             '北京是[MASK]的首都。',
             '生活的真谛是[MASK]。',
             '在二战中，阿道夫·希特勒是不可饶恕的[MASK]。',
             '1949年十月一号，发生了一件大事，那就是中华人民共和国[MASK]。',
             '雨后，我[MASK]在公园里，呼吸新鲜的空气。',
             '根据量子场论，粒子的质量来自[MASK]。',
             '雨后，彩虹出现在天边，小美陶醉地说："真[MASK]啊!"',]
    
    texts_idx,mask_positions = tokenize_texts(texts,tokenizer)
    input_ids = torch.tensor(texts_idx,dtype=torch.long,device=device)
    MAX_CUM_PROB = 0.7
    import time
    with torch.no_grad():
        with torch.autocast(device_type=device,dtype=dtype):
            print(f'start to forward[{device}]')
            start_time = time.time()
            logits,_ = model.forward(input_ids)
            end_time = time.time()
            print(f'forward time is {end_time-start_time}')
            for b in range(len(texts_idx)):
                mask_position = mask_positions[b]
                masked_prob = torch.softmax(logits[b,mask_position],dim=-1)
                print(masked_prob)
                print(masked_prob.shape)
                probs,indices = torch.topk(masked_prob,10)
                print(probs)
                print(indices)
                for position in mask_position:
                    cum_prob = 0
                    mask_idx = 0
                    for i in range(10):
                        texts_idx[b][position] = indices[mask_idx][i].item()
                        prob = probs[mask_idx][i].item()
                        cum_prob += prob
                        print(tokenizer.decode(texts_idx[b]),' prob is ',prob,' cum_prob is ',cum_prob)
                        if cum_prob > MAX_CUM_PROB:
                            break
                    mask_idx += 1
                print('----------------------------------')
    