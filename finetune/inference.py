import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch, transformers, pyreft
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()


def generate_response():
    '''
    simple test for the model
    '''
    model = transformers.AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map=device)
    tokenizer = transformers.AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", use_fast=False)
    # model.resize_token_embeddings(len(tokenizer))
    streamer = transformers.TextStreamer(tokenizer,skip_prompt=True)
    reft_model = pyreft.ReftModel.load("benchang1110/Tinyllama-1.1B-Chat-REFT-v1.0", model)
    reft_model.set_device(device)
    
    while(1):
        prompt = input('USER:')
        if prompt == "exit":
            break
        print("Assistant: ")
        messages = [
            {'content': prompt, 'role': 'user'},
        ]
        prompt = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        # print(prompt)
        prompt = tokenizer(prompt, return_tensors="pt").to(device)  # move prompt to the same device as the model
        
        # have to set the following hyperparameters to make the model work (so stupid.....)
        base_unit_location = prompt["input_ids"].shape[-1] - 1  # last position    
        first_n = 8 # (number of first_n)
        last_n = 8 # (number of last_n)
        LAYER = [i for i in range(model.config.num_hidden_layers)]
        
        base_unit_locations = [[[i for i in range(first_n)] + [base_unit_location-i for i in range(last_n)]]]*len(LAYER)
        _, reft_response = reft_model.generate(
                prompt, unit_locations={"sources->base": (None, base_unit_locations)},
                intervene_on_prompt=True, max_new_tokens=256, do_sample=True, temperature=0.1,repetition_penalty=1.1,streamer=streamer
        )
    
    
        # # loreft generate
        # _, reft_response = reft_model.generate(
        #     prompt, unit_locations={"sources->base": (None, [[[base_unit_location-i for i in range(position)]]])},
        #     intervene_on_prompt=True, max_new_tokens=256, do_sample=False, 
        # )           

def generate_response_lora():
    '''
    simple test for the model
    '''
    while (1):
        model = transformers.AutoModelForCausalLM.from_pretrained('benchang1110/Tinyllama-1.1B-Chat-PEFT-v1.0', torch_dtype=torch.bfloat16, device_map=args.device,attn_implementation="flash_attention_2")
        tokenizer = transformers.AutoTokenizer.from_pretrained('benchang1110/Tinyllama-1.1B-Chat-PEFT-v1.0', use_fast=False)
        # model = transformers.AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0', torch_dtype=torch.bfloat16, device_map=args.device,attn_implementation="flash_attention_2")
        # tokenizer = transformers.AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0', use_fast=False)
        streamer = transformers.TextStreamer(tokenizer,skip_prompt=True)
        prompt = input('USER:')
        if prompt == "exit":
            break
        print("Assistant: ")
        
        system = '你是一個來自台灣的助理，樂於以台灣人的立場幫助使用者，會用繁體中文回答問題。'
        message = [
            {"content": system, "role": "system"},
            {'content': prompt, 'role': 'user'},
        ]
        untokenized_chat = tokenizer.apply_chat_template(message,tokenize=False,add_generation_prompt=False)
        inputs = tokenizer.encode_plus(untokenized_chat, add_special_tokens=True, return_tensors="pt",return_attention_mask=True).to(device)
        outputs = model.generate(inputs["input_ids"],attention_mask=inputs['attention_mask'],streamer=streamer,use_cache=True,max_new_tokens=512,do_sample=True,temperature=0.1,repetition_penalty=1.2)

def generate_response_fp():
    model = transformers.AutoModelForCausalLM.from_pretrained("./fp_export", torch_dtype=torch.bfloat16, device_map=device,attn_implementation="flash_attention_2")
    tokenizer = transformers.AutoTokenizer.from_pretrained('./fp_export', use_fast=False)
    streamer = transformers.TextStreamer(tokenizer,skip_prompt=True)
    while(1):
        prompt = input('USER:')
        if prompt == "exit":
            break
        print("Assistant: ")
        # system = '你是一個來自台灣的助理，樂於以台灣人的立場幫助使用者，會用繁體中文回答問題。'
        message = [
           {'content': prompt, 'role': 'user'},
        ]
        untokenized_chat = tokenizer.apply_chat_template(message,tokenize=False,add_generation_prompt=False)
        inputs = tokenizer.encode_plus(untokenized_chat, add_special_tokens=True, return_tensors="pt",return_attention_mask=True).to(device)
        outputs = model.generate(inputs["input_ids"],attention_mask=inputs['attention_mask'],streamer=streamer,use_cache=True,max_new_tokens=1024,do_sample=True,temperature=0.5,repetition_penalty=1.2)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # generate_response_lora()
    # generate_response()
    generate_response_fp()        
