import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
import transformers
import torch
import argparse
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
transformers.logging.set_verbosity_error()
torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', default='TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T', help='model name or path')
parser.add_argument('--dataset_name', default=None, help='dataset name')
parser.add_argument('--preprocessing_num_workers', default=48, help='number of workers for preprocessing')
parser.add_argument('--block_size', default=2048, help='block size')
parser.add_argument('--resume', default='', help='resume checkpoint')
parser.add_argument('--sample', default=None,type=int, help='sample size')
parser.add_argument('--save', default='checkpoint', help='path to the folder to save checkpoint')
parser.add_argument('--export', default='export', help='path to the folder to upload to hub')
parser.add_argument('--epoch', default=1, help='number of epochs to train')
parser.add_argument('--batch_size', default=16, help='batch size')
parser.add_argument('--lr', default=1e-4, help='learning rate')
parser.add_argument('--test_size', default=0.005, help='test size')
args = parser.parse_args()


def tokenize_function(examples):
    return tokenizer(examples[text_column_name])

def rename_column(examples):
    return {"text": [example for example in examples['article']]} 

def filter_empty_sequences(example):
    return len(example['input_ids']) > 0



def group_texts(examples):
        from itertools import chain
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        
        return result

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Downloading and loading a dataset from the hub.
    raw_datasets1 = load_dataset("benchang1110/pretrainedtw",split='train')
    raw_datasets1 = raw_datasets1.map(rename_column)
    raw_datasets2 = load_dataset("HuggingFaceTB/cosmopedia-100k", split="train")
    
    raw_datasets = concatenate_datasets([raw_datasets1,raw_datasets2])
    
    if args.sample is not None:
        raw_datasets = raw_datasets.select(range(args.sample))
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,torch_dtype=torch.float32,attn_implementation="flash_attention_2",device_map=device)
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 2048
    
    model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()
    
    ###################################################
    text_column_name = "text"

    # tokenized_datasets = raw_datasets.map(tokenize_function,batched=True,batch_size=1000,num_proc=32,remove_columns='text',load_from_cache_file = False, desc="Running tokenizer on dataset")
    tokenized_datasets = raw_datasets.map(tokenize_function,batched=True,batch_size=1000,num_proc=args.preprocessing_num_workers,remove_columns=raw_datasets.column_names,load_from_cache_file=True, desc="Running tokenizer on dataset")
    
    # tokenized_datasets = tokenized_datasets.filter(filter_empty_sequences)
    
    if args.block_size is None:
        block_size = tokenizer.model_max_length # should be 2048 here
    else:
        block_size = args.block_size
        
    lm_datasets = tokenized_datasets.map(group_texts,batched=True,num_proc=args.preprocessing_num_workers,load_from_cache_file=True,desc=f"Grouping texts in chunks of {block_size}")
    # lm_datasets = lm_datasets.filter(filter_empty_sequences)
    lm_datasets = lm_datasets.train_test_split(test_size=args.test_size)
    
    train_ds = lm_datasets['train']
    test_ds = lm_datasets['test']
    
    print(train_ds,test_ds)
    
    # # #---------------------------------training---------------------------------
    training_args = TrainingArguments(
        output_dir=args.save, #The output directory
        overwrite_output_dir=True, #overwrite the content of the output directory
        num_train_epochs=args.epoch, # number of training epochs
        per_device_train_batch_size=args.batch_size, # batch size for training
        per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
        learning_rate=args.lr,
        weight_decay=1e-4,
        lr_scheduler_type = 'cosine',
        warmup_ratio = 0.05,
        max_grad_norm=1.0, #gradient clipping
        
        bf16=True,
        gradient_accumulation_steps=5,
        
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps= 10,
        evaluation_strategy="steps",
        report_to="wandb",
        load_best_model_at_end=True,
        save_steps=5000,
        save_total_limit=3,
        
        eval_accumulation_steps=5,
        dataloader_num_workers=8,
        dataloader_pin_memory=False,
        remove_unused_columns=True,
        gradient_checkpointing=True,
        eval_steps=5000,
        auto_find_batch_size=True,
        dataloader_drop_last=False
    )
    
    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False,return_tensors='pt')
    data_collator = transformers.default_data_collator
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset = train_ds,
        eval_dataset = test_ds,
        tokenizer=tokenizer,
    )
    
    # trainer.train(resume_from_checkpoint=True)
    trainer.train()
    trainer.save_model(args.export)
    