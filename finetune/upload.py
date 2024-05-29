import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse, transformers, pyreft,torch
from huggingface_hub import snapshot_download


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cpu")
args = parser.parse_args()

if __name__ == '__main__':
    # snapshot_download(repo_id="benchang1110/Tinyllama-1.1B-Chat-REFT-v1.0",local_dir='./export',revision = 'a8fc5ab00da5308d7e55bb02422080f8f2e8aea8')
    # model = transformers.AutoModelForCausalLM.from_pretrained("benchang1110/Tinyllama-1.1B-Chat-REFT-v1.0", torch_dtype=torch.bfloat16, device_map=args.device)
    # reft_model = pyreft.ReftModel.load("benchang1110/Tinyllama-1.1B-Chat-REFT-v1.0", model)
    
    # reft_model.set_device(args.device)
    # reft_model.save(
    # save_directory="./Tinyllama-1.1B-Chat-REFT-v1.0", 
    # save_to_hf_hub=True, 
    # hf_repo_name="benchang1110/Tinyllama-1.1B-Chat-REFT-v1.0",
    # )
    model = transformers.AutoModelForCausalLM.from_pretrained("./fp_export", torch_dtype=torch.bfloat16, device_map=args.device,attn_implementation="flash_attention_2")
    tokenizer = transformers.AutoTokenizer.from_pretrained("./fp_export", use_fast=True)

    model.push_to_hub(
        repo_id="benchang1110/Taiwan-tinyllama-v1.0-chat"
    )

    tokenizer.push_to_hub(
        repo_id="benchang1110/Taiwan-tinyllama-v1.0-chat"
    )