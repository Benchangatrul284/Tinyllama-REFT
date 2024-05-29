import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import transformers
from transformers import TrainingArguments
from datasets import load_dataset, concatenate_datasets
import torch
from peft import LoraConfig, PeftModelForCausalLM
import argparse
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

torch.multiprocessing.set_sharing_strategy('file_system')
transformers.logging.set_verbosity_error()
torch.cuda.empty_cache()
parser = argparse.ArgumentParser()

parser.add_argument("--model_name_or_path", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--max_length", type=int, default=2048)
parser.add_argument('--resume', default='', help='resume checkpoint')
parser.add_argument('--save', default='lora_checkpoint', help='path to the folder to save checkpoint')
parser.add_argument('--export', default='lora_export', help='path to the folder to upload to hub')
parser.add_argument('--epoch', default=3, help='number of epochs to train')
parser.add_argument('--batch_size', default=16, help='batch size')
parser.add_argument('--lr', default=5e-4, help='learning rate')
parser.add_argument('--test_size', default=0.01, help='test size')
parser.add_argument('--alpha', default=4, help='alpha')
parser.add_argument('--rank', default=8, help='rank')
parser.add_argument('--merge', default=True, help='merge model')
args = parser.parse_args()


def apply_chat_template(batch):
    tokenizer = transformers.AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", use_fast=False)
    return {"formatted_chat": [tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False) for chat in batch["conversation"]]}

def tokenized_dataset(dataset):
    return dataset.map(tokenize_function,batched=True,num_proc=32,batch_size=10000,remove_columns=['formatted_chat'])

def tokenize_function(dataset):
    tokenizer = transformers.AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0",use_fast=True)
    return tokenizer(dataset['formatted_chat'],padding=True,max_length=args.max_length,truncation=True)

def add_system_prompt(batch):
    return {"conversation": [ [{"role": "system", "content": "你是一個來自台灣的AI助理，樂於以台灣人的立場幫助使用者，會用繁體中文回答問題。"}] + chat for chat in batch["conversation"] ]}


if __name__ == "__main__":
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16, device_map=args.device,attn_implementation="flash_attention_2")
    tokenizer = transformers.AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0', use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id
    
    model.resize_token_embeddings(len(tokenizer))
    
    ##---------------------------------PEFT------------------------------------------------------
    config = LoraConfig(
        r=args.rank, 
        lora_alpha=args.alpha,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM" # set this for CLM or Seq2Seq
    )
    
    model = PeftModelForCausalLM(model, config)
    model.print_trainable_parameters()
    
    ##---------------------------------data processing with datasets library----------------------
    dataset1 = load_dataset('benchang1110/Chattw_v2',split='train')
    dataset2 = load_dataset('benchang1110/WildChat-Chinese',split='train')
    dataset  = concatenate_datasets([dataset1,dataset2])
    dataset = dataset.map(add_system_prompt,batched=True,batch_size=10000,num_proc=32)
    dataset = dataset.map(apply_chat_template,batched=True,remove_columns=['conversation'],batch_size=10000,num_proc=32) # formated chat
    # dataset = tokenized_dataset(dataset) (for SFTTrainer, we should not tokenize first)
    dataset = dataset.train_test_split(test_size=args.test_size)
    train_ds = dataset['train']
    test_ds = dataset['test']
    print(train_ds,test_ds)
    
    
    # #---------------------------------training---------------------------------
    training_args = TrainingArguments(
        output_dir=args.save, #The output directory
        overwrite_output_dir=True, #overwrite the content of the output directory
        num_train_epochs=args.epoch, # number of training epochs
        per_device_train_batch_size=args.batch_size, # batch size for training
        per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
        learning_rate=args.lr,
        weight_decay = 1e-4,
        warmup_ratio = 0.05,
        max_grad_norm = 1.0, #gradient clipping
        
        bf16=True,
        gradient_accumulation_steps=1,
        
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps= 10,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        save_steps = 1000,
        save_total_limit=3,
        
        eval_accumulation_steps=10,
        dataloader_num_workers=16,
        dataloader_pin_memory=False,
        remove_unused_columns=True,
        gradient_checkpointing=False,
        eval_steps=1000,
        auto_find_batch_size=True,
    )
    instruction_template = "<|system|>\n"
    response_template = "<|assistant|>\n"
    
    data_collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer, mlm=False)
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        packing=False,
        dataset_text_field="formatted_chat",
        max_seq_length=args.max_length
    )
        
    trainer.train()
    
    # saving the model
    if args.merge == True:
        merge_model = model.merge_and_unload()
        if not os.path.exists(args.export):
            os.mkdir(args.export)
        
        merge_model.save_pretrained(args.export)
        tokenizer.save_pretrained(args.export)
        print("Model saved to {} waiting to upload".format(args.export))
    
