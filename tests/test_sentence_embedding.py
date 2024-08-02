import sys
import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'src')
sys.path.append(src_dir)
from model_encoder_run import encode_sentence
def test_texts(args, model, device, texts, tokenizer,dtype):
    texts_idx = [tokenizer.encode(text,add_special_tokens=False) for text in texts]
    for text_idx in texts_idx:text_idx.append(args.emb_id)
    max_len = max([len(text_idx) for text_idx in texts_idx])
    texts_idx = [text_idx + [args.pad_id]*(max_len-len(text_idx)) for text_idx in texts_idx]
    
   
    input_ids = torch.tensor(texts_idx,dtype=torch.long,device=device)
    MAX_CUM_PROB = 0.7
    import time
    from sentence_transformers.util import cos_sim as sim_fn
    with torch.no_grad():
        with torch.autocast(device_type=device,dtype=dtype):
            print('start to forward[CPU]')
            start_time = time.time()
            embs = encode_sentence(model,input_ids)
            end_time = time.time()
            print(f'forward time is {end_time-start_time}')
            print(sim_fn(embs[0],embs[1:]).squeeze(0).tolist())
            
if __name__ == '__main__':
    import argparse
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser("Test MLM model")
    parser.add_argument("--model_file",type=str,default='/media/yueyulin/data_4t/models/bi_encoder/bi_encoder_all_chinese/trainable_model/epoch_9/RWKV-x060-MLM-ctx4096.pth.pth')
    parser.add_argument("--device",type=str,default='cpu')
    parser.add_argument("--dtype",type=str,default='float32',choices=['bfloat16','float16','float32'])
    args = parser.parse_args() 
    device = args.device
    import torch
    if args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "float32":
        dtype = torch.float32
    from utilities import load_base_model,tokenize_texts
    model = load_base_model(args.model_file)
    print(model)
    modle = model.to(device=device,dtype=dtype)
    model = torch.jit.script(model)
    tokenizer = AutoTokenizer.from_pretrained('THUDM/glm-4-9b-chat', trust_remote_code=True)
    print(tokenizer)
    args.emb_id = 151329
    args.pad_id = 151334
    args.mask_id = 151330
    ########################################################
    texts = ['每天吃苹果有什么好处？',
             '某些水果和蔬菜特别富含可溶性纤维。在水果中，柑橘类水果如橙子、葡萄柚和柠檬的可溶性纤维含量相当高，浆果也是如此，包括草莓、蓝莓、黑莓和覆盆子。苹果和梨也提供可溶性纤维，香蕉也是如此。',
             '苹果，落叶乔木，叶子椭圆形，花白色带有红晕。果实圆形，味甜或略酸，是常见水果，具有丰富营养成分，有食疗、辅助治疗功能。苹果原产于欧洲、中亚、西亚和土耳其一带，于十九世纪传入中国。中国是世界最大的苹果生产国，在东北、华北、华东、西北和四川、云南等地均有栽培。',
             '苹果生吃治便秘，熟吃治腹泻：苹果中含有丰富的鞣酸、果胶、膳食纤维等特殊物质，鞣酸是肠道收敛剂，它能减少肠道分泌而使大便内水分减少，从而止泻。而果胶则是个“两面派”，未经加热的生果胶有软化大便缓解便秘的作用，煮过的果胶却摇身一变，具有收敛、止泻的功效。膳食纤维又起到通便作用。']
    test_texts(args, model, device, texts, tokenizer,dtype)
    

    texts = ['每天吃苹果有什么好处？',
             '宁神安眠：苹果中含有的磷和铁等元素，易被肠壁吸收，有补脑养血、宁神安眠作用。苹果的香气是治疗抑郁和压抑感的良药。研究发现，在诸多气味中，苹果的香气对人的心理影响最大，它具有明显的消除心理压抑感的作用。',
             '美白养颜、降低胆固醇：苹果中的胶质和微量元素铬能保持血糖的稳定，还能有效地降低胆固醇。苹果中的粗纤维可促进肠胃蠕功，并富含铁、锌等微量元素，可使皮肤细润有光泽，起到美容瘦身的作用。',
             '苹果生吃治便秘，熟吃治腹泻：苹果中含有丰富的鞣酸、果胶、膳食纤维等特殊物质，鞣酸是肠道收敛剂，它能减少肠道分泌而使大便内水分减少，从而止泻。而果胶则是个“两面派”，未经加热的生果胶有软化大便缓解便秘的作用，煮过的果胶却摇身一变，具有收敛、止泻的功效。膳食纤维又起到通便作用。',
             '保护心脏：苹果的纤维、果胶、抗氧化物等能降低体内坏胆固醇并提高好胆固醇含量，所以每天吃一两个苹果不容易得心脏病。']
    test_texts(args, model, device, texts, tokenizer,dtype)
    
    texts = ['庆余年2是谁投资拍摄的？',
             '《庆余年第二季》是由孙皓执导，王倦担任编剧，张若昀、李沁领衔主演，陈道明特别主演，吴刚、田雨领衔主演，袁泉、毛晓彤特邀出演，郭麒麟特邀主演的古装传奇剧 [1] [65]。',
             '该剧于2024年5月16日在央视八套首播，腾讯视频全网独播 [60]。2024年5月16日，据“CCTV电视剧”官微，数据显示，CCTV-8黄金强档热播剧《庆余年2》当晚实时直播关注度峰值破2 [57]；5月28日，据灯塔专业版数据，《庆余年2》累计正片播放量已突破12亿 [72]。',
             '苹果生吃治便秘，熟吃治腹泻：苹果中含有丰富的鞣酸、果胶、膳食纤维等特殊物质，鞣酸是肠道收敛剂，它能减少肠道分泌而使大便内水分减少，从而止泻。而果胶则是个“两面派”，未经加热的生果胶有软化大便缓解便秘的作用，煮过的果胶却摇身一变，具有收敛、止泻的功效。膳食纤维又起到通便作用。',
             '《庆余年第二季》出品公司：中央电视台、上海腾讯企鹅影视文化传播有限公司、天津阅文影视文化传媒有限公司、新丽电视文化投资有限公司、新丽（上海）影视有限公司']
    test_texts(args, model, device, texts, tokenizer,dtype)
    
    texts = ['河北省有多少个地级市？',
             '河北省，简称“冀”，中华人民共和国省级行政区，省会石家庄，位于北纬36°05′-42°40′，东经113°27′-119°50′之间东南部、南部衔山东、河南两省，西倚太行山与山西省为邻，西北与内蒙古自治区交界，东北部与辽宁接壤。总面积18.88万平方千米。 [1-2]截至2023年末，河北省下辖11个地级市，共有49个市辖区、21个县级市、91个县、6个自治县，常住总人口为7393万人。 [3] [167]',
             '河北省地处华北，漳河以北，东临渤海、内环京津，西为太行山地，北为燕山山地，燕山以北为张北高原，其余为河北平原，有世界文化遗产3处、A级景区共513家 [176]、国家5A级旅游景区12家 [168]、国家4A级旅游景区162家 [176]、国家级历史文化名城6座 [143]。河北在战国时期大部分属于赵国和燕国，又被称为燕赵之地，地处温带大陆性季风气候。河北地处中原地区，自古有“燕赵多有慷慨悲歌之士”之称。',
             '2023年，河北省地区生产总值43944.1亿元，比上年增长5.5%。 [161]分产业看，第一产业增加值4466.2亿元，同比增长2.6%；第二产业增加值16435.3亿元，增长6.2%；第三产业增加值23042.6亿元，增长5.5%。',
             '河北省是中华民族的发祥地之一。早在五千多年前，中华民族的三大始祖黄帝、炎帝和蚩尤就在河北由征战到融合，开创了中华文明史。']
    test_texts(args, model, device, texts, tokenizer,dtype)