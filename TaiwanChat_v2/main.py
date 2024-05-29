from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
import os
import json
import regex as re

parser = argparse.ArgumentParser()
parser.add_argument('--percent', type=str, default='train[:10%]', help='part of the training data')
parser.add_argument('--export', type=str, default='TaiwanChat[:1%].json', help='export file name')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--device', type=str, default='cuda:1', help='device')
args = parser.parse_args()


def generate_output(batch):
    '''
    generate batch output from the model
    '''
    messages = []
    for content in batch:
        # print(content)
        message = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        messages.append(tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False))
    
    inputs = tokenizer.batch_encode_plus(messages, add_special_tokens=True, return_tensors="pt",padding=False, truncation=False,return_attention_mask=True).to(device)
    outputs = model.generate(inputs["input_ids"], attention_mask=inputs['attention_mask'], use_cache=True, max_new_tokens=8192, do_sample=True,temperature=0.7,max_time=60)
    outputs = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:],  skip_special_tokens=True)
    
    for output in zip(batch,outputs):
        messages = []
        messages.append({"content": output[0],"role": "user"})
        messages.append({"content": output[1],"role": "assistant"})
        yield messages
            
            
if __name__ == '__main__':
    if os.path.exists(args.export):
        os.remove(args.export)
    device = args.device if torch.cuda.is_available() else 'cpu'
    model = AutoModelForCausalLM.from_pretrained("taide/Llama3-TAIDE-LX-8B-Chat-Alpha1", attn_implementation="flash_attention_2",device_map=device,torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained("taide/Llama3-TAIDE-LX-8B-Chat-Alpha1",use_fast=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    system = '你是一個來自台灣的助理，樂於以台灣人的立場幫助使用者，會用繁體中文回答問題。'
    dataset = load_dataset('yentinglin/TaiwanChat',split=args.percent)
    
    # for item in dataset:
    #     try:
    #         print(item['messages'][0]['content'])
    #         for output in generate_output([item['messages'][0]['content']]):
    #             print(output)
    #         break
    #     except Exception as e:
    #         print(e)
    #         continue
        
    with open(os.path.join(args.export), 'a') as outfile:
        outfile.write('[\n')
        batch = []
        for index,item in enumerate(dataset):
            print("processing:", index)
            try:
                if len(item['messages'][0]['content']) < 300:
                    print("prompt:", item['messages'][0]['content'])
                    message = [
                        {"role": "system", "content": system},
                        {"role": "user", "content": item['messages'][0]['content']},
                    ]
                    messages = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
                    inputs = tokenizer.encode_plus(messages, add_special_tokens=True, return_tensors="pt",return_attention_mask=True).to(device)
                    outputs = model.generate(inputs["input_ids"],attention_mask=inputs['attention_mask'],use_cache=True,max_new_tokens=8192,do_sample=True,temperature=0.7,max_time=60)
                    output = tokenizer.decode(outputs[0],skip_special_tokens=True)
                    # clear the prompt and output
                    output = output.split('assistant\n\n')[-1]
                    print('After splitting',output)
                    conversation = []
                    conversation.append({"content": item['messages'][0]['content'],"role": "user"})
                    conversation.append({"content": output,"role": "assistant"})
                    json.dump(conversation, outfile, ensure_ascii=False,indent=4)
                    outfile.write(',\n')
                    outfile.flush()
            except Exception as e:
                print(e)
                continue