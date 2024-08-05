import os
parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
import sys
sys.path.append(parent_path)
print(f'add path: {parent_path} to sys.path')
import torch
from rwkv.model import RWKV
from rwkv.utils import PIPELINE,PIPELINE_ARGS
class StatesGenerator:
    def __init__(self,base_model,strategy="cuda fp16"):
        self.base_model = base_model
        model = RWKV(self.base_model,strategy=strategy)
        self.args = model.args
        self.pipeline = PIPELINE(model,"rwkv_vocab_v20230424")
        self.device = strategy.split(' ')[0]
        self.states = {}
        type_str = strategy.split(' ')[1]
        if type_str == 'fp16':
            self.dtype = torch.float16
        elif type_str == 'fp32':
            self.dtype = torch.float32
        else:
            self.dtype = torch.bfloat16
    def load_states(self,states_file,states_name):
        args = self.args
        states = torch.load(states_file)
        states_value = []
        for i in range(args.n_layer):
            key = f'blocks.{i}.att.time_state'
            value = states[key]
            prev_x = torch.zeros(args.n_embd,device=self.device,dtype=torch.float)
            prev_states = torch.tensor(value,device=self.device,dtype=torch.float).transpose(1,2)
            prev_ffn = torch.zeros(args.n_embd,device=self.device,dtype=torch.float)
            states_value.append(prev_x)
            states_value.append(prev_states)
            states_value.append(prev_ffn)
        self.states[states_name] = states_value

    def get_states(self,states_name):
        if states_name not in self.states:
            raise None
        else:
            states_copy = []
            for s in self.states[states_name]:
                states_copy.append(s.clone())
            return states_copy


    def generate(self,input_text,instruction,states_name,temperature = 1.0, top_p = 0.96, top_k = 20, alpha_frequency = 0.25, alpha_presence = 0.25, alpha_decay = 0.996, token_ban = [], token_stop = [0,1], chunk_len = 512,gen_count=128):
        states = self.get_states(states_name)
        gen_args = PIPELINE_ARGS(temperature = temperature, top_p = top_p, top_k = top_k, alpha_frequency = alpha_frequency, alpha_presence = alpha_presence, alpha_decay = alpha_decay, token_ban = token_ban, token_stop = token_stop, chunk_len = chunk_len)
        cat_char = '🐱'
        bot_char = '🤖'
        ctx = f'{cat_char}:{instruction}\n{input_text}\n{bot_char}:'
        with torch.no_grad():
            with torch.autocast(enabled=True,device_type='cuda' if self.device.startswith('cuda') else 'cpu',dtype=self.dtype):
                output = self.pipeline.generate(ctx,gen_count,gen_args,state=states)
        return output
    
