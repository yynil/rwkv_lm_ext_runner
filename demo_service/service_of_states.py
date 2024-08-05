from fastapi import FastAPI, Request,HTTPException
from pydantic import BaseModel
from typing import List, Optional
import time
import torch
import argparse
import json
states_runner = None
states_configuration = None

def setup_env():
    import os
    parent_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'src')
    import sys
    sys.path.append(parent_path)
    print(f'add path: {parent_path} to sys.path')
    os.environ['RWKV_JIT_ON'] = '1'
    os.environ['RWKV_T_MAX'] = '4096'
    os.environ['RWKV_FLOAT_MODE'] = 'bf16'
    os.environ['RWKV_HEAD_SIZE_A'] = '64'
    os.environ['RWKV_T_MAX'] = '4096'
    os.environ["RWKV_MY_TESTING"]='x060'
    os.environ['RWKV_CTXLEN'] = '4096'
    os.environ['WKV'] = ''
    os.environ['RWKV_TRAIN_TYPE'] = ''
    os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
    # os.environ['NO_CUDA'] = '1'
    os.environ["RWKV_CUDA_ON"] = '1'
setup_env()
from states_generator import StatesGenerator
app = FastAPI()

class InputData(BaseModel):
    input_text: str
    states_name: str
    
class OutputData(BaseModel):
    output_text: str
    elapsed_time: float

@app.post("/process_text", response_model=OutputData)
def process_text(data: InputData):
    start_time = time.time()
    
    try:
        instruction = ''
        if data.states_name in states_configuration:
            instruction = states_configuration[data.states_name]['instruction']
        input_data = {'input':data.input_text}
        output_text = states_runner.generate(json.dumps(input_data,ensure_ascii=False),instruction,gen_count=512,states_name=data.states_name)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
    elapsed_time = time.time() - start_time
    return OutputData(output_text=output_text, elapsed_time=elapsed_time)

def load_states_config(states_file_config):
    with open(states_file_config,'r') as f:
        return json.load(f)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("States generator")
    parser.add_argument("--device",type=str,default='cuda:0')
    parser.add_argument('--dtype',type=str,default='bf16',choices=['fp16','fp32','bf16'])
    parser.add_argument('--states_file_config',type=str)
    parser.add_argument('--llm_model_file',type=str,default='/media/yueyulin/KINGSTON/models/rwkv6/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth')
    args = parser.parse_args() 
    print(args)
    ###Init llm
    strategy = f'{args.device} {args.dtype}'
    if args.dtype == 'fp16':
        dtype = torch.half
    elif args.dtype == 'fp32':
        dtype = torch.float32
    else:
        dtype = torch.bfloat16
    states_runner = StatesGenerator(args.llm_model_file,strategy=strategy)
    states_configuration = load_states_config(args.states_file_config)
    for states_name in states_configuration.keys():
        states_file = states_configuration[states_name]['file']
        states_runner.load_states(states_file,states_name)
    instruction ='根据input中文本内容，协助用户识别文本所属的领域。随后，找出与该领域关联最紧密的专家。接着，作为输出，列举出五至十项可在该文本中执行的具体任务。接下来，提取以下信息：领域：对于给定的示例文本，帮助用户指定一个描述性领域，概括文本的主题。请按照JSON字符串的格式回答，无法提取则不输出'
    input_text = '{\"input\":\"超长期特别国债（ultra-long special treasury bonds），一般指发行期限在10年以上的，为特定目标发行的、具有明确用途的国债。超长期特别国债专项用于国家重大战略实施和重点领域安全能力建设，2024年先发行1万亿元，期限分别为20年、30年、50年。 [1]\
    2024年5月13日，财政部网站公布2024年一般国债、超长期特别国债发行有关安排。 [6-7]2024年5月17日，30年期超长期特别国债正式首发。根据发行安排，首发的30年期超长期特别国债，为固定利率附息债，总额400亿元。 [8]6月14日,财政部发行2024年超长期特别国债（三期）（50年期），竞争性招标面值总额350亿元。 [13]7月24日，通过财政部政府债券发行系统招标发行550亿元30年期超长期特别国债，票面利率在当天通过竞争性招标确定。 [15]\"}'
    print('start to generate...')
    output = states_runner.generate(input_text,instruction,gen_count=512,states_name='domain_expert')
    print(output)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9998)