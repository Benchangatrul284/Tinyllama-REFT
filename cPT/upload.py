import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("./export", torch_dtype=torch.bfloat16, device_map='cuda:1',attn_implementation="flash_attention_2")
tokenizer = AutoTokenizer.from_pretrained('./export')

model.push_to_hub("benchang1110/1.1B-Octopus-pretrained")
tokenizer.push_to_hub("benchang1110/1.1B-Octopus-pretrained")