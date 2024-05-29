from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, AutoConfig
import torch

def generate_response(input):
    '''
    simple test for the model
    '''
    # tokenzize the input
    tokenized_input = tokenizer.encode_plus(input, return_tensors='pt').to(device)
    print(tokenized_input['input_ids'])
    # generate the response
    outputs = model.generate(
        input_ids=tokenized_input['input_ids'], 
        attention_mask=tokenized_input['attention_mask'],
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        repetition_penalty=1.3,
        max_length=500
    )
    
    # decode the response
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == '__main__':
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    # config = AutoConfig.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    # model = AutoModelForCausalLM.from_pretrained("benchang1110/1.1B-Octopus-pretrained",attn_implementation="flash_attention_2",device_map=device,torch_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained("./checkpoint/checkpoint-5000",  attn_implementation="flash_attention_2",device_map=device,torch_dtype=torch.bfloat16)
    # tokenizer = AutoTokenizer.from_pretrained("benchang1110/1.1B-Octopus-pretrained",use_fast=True)
    tokenizer = AutoTokenizer.from_pretrained("taide/TAIDE-LX-7B",use_fast=True)
    # model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",attn_implementation="flash_attention_2",device_map=device,torch_dtype=torch.bfloat16)
    # tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    while(True):
        text = input("input a simple prompt:")
        print('System:', generate_response(text))
    
    
        
    