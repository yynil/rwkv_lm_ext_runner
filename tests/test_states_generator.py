if __name__ == '__main__':
    import sys
    import os
    os.environ["RWKV_JIT_ON"] = '1'
    src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'src')
    sys.path.append(src_dir)
    from states_generator import StatesGenerator
    import argparse
    parser = argparse.ArgumentParser("Test StatesGenerator")
    parser.add_argument("--model_file",type=str,default='/media/yueyulin/KINGSTON/models/rwkv6/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth')
    parser.add_argument("--states_file",type=str,default='/media/yueyulin/data_4t/models/states_tuning/units_extractor/trainable_model/epoch_2/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth.pth')
    args = parser.parse_args()
    state_name ='sn'
    # model_file = '/media/yueyulin/KINGSTON/models/rwkv6/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth'
    # unit_extractor_states_file = '/media/yueyulin/data_4t/models/states_tuning/units_extractor/trainable_model/epoch_2/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth.pth'
    sg = StatesGenerator(args.model_file)
    sg.load_states(args.states_file,states_name=state_name)
    instruction = input('Please input instruction')
    while(True):
        input_text = input('Please input text:')
        output_text = sg.generate(input_text,instruction,state_name,top_k=0,top_p=0,gen_count=256)
        print(f"Output is :{output_text}")
    # unit_instruction = '你是一个单位提取专家。请从input中抽取出数字和单位，请按照JSON字符串的格式回答，无法提取则不输出。'
    # input_text = '大约503万平方米'
    # unit_output = sg.generate(input_text,unit_instruction,state_name,top_k=0,top_p=0,gen_count=128)
    # print(unit_output)
    # input_text = '4845人'
    # unit_output = sg.generate(input_text,unit_instruction,state_name,top_k=0,top_p=0,gen_count=128)
    # print(unit_output)
    # input_text = '约89434户'
    # unit_output = sg.generate(input_text,unit_instruction,state_name,top_k=0,top_p=0,gen_count=128)
    # print(unit_output)
    # input_text = '可能有38.87亿平方公里'
    # unit_output = sg.generate(input_text,unit_instruction,state_name,top_k=0,top_p=0,gen_count=128)
    # print(unit_output)