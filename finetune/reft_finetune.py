import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import datasets
import copy
import torch
import transformers
import argparse
import pyreft
from transformers import  TrainingArguments
import transformers
import torch
import argparse
import os
import torch.multiprocessing
from pyreft import ReftDataset
torch.multiprocessing.set_sharing_strategy('file_system')
transformers.logging.set_verbosity_error()

torch.cuda.empty_cache()

IGNORE_INDEX = -100
parser = argparse.ArgumentParser()

parser.add_argument("--model_name_or_path", type=str, default="benchang1110/Taiwan-tinyllama-v1.0-chat")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--max_length", type=int, default=2048)
parser.add_argument("--layers", type=str, default="all")
parser.add_argument("--position", type=str, default="f8+l8")
parser.add_argument("--share_weights",type=bool,default=True)
parser.add_argument("--rank", type=int, default=8)
parser.add_argument("--samples", type=int, default=None)

parser.add_argument('--resume', default='', help='resume checkpoint')
parser.add_argument('--save', default='checkpoint', help='path to the folder to save checkpoint')
parser.add_argument('--export', default='export', help='path to the folder to upload to hub')

parser.add_argument('--epoch', default=5, help='number of epochs to train')
parser.add_argument('--batch_size', default=4, help='batch size')
parser.add_argument('--lr', default=1e-3, help='learning rate')

args = parser.parse_args()


def map_to_template(batch):
    return {"conversation": [[{'content':sample[0],'role':'user'},{'content':sample[1],'role':'assistant'}] for sample in zip(batch["question"],batch["answer"])]}

def apply_chat_template(batch):
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    return {"formatted_chat": [tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False) for chat in batch["conversation"]]}

def apply_chat_template_and_tokenize(batch):
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    return {"formatted_chat": [tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False) for chat in batch["conversation"]]}

def tokenized_dataset(dataset):
    return dataset.map(tokenize_function,batched=True,num_proc=32,batch_size=10000,remove_columns=['formatted_chat'])

def tokenize_function(dataset):
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    return tokenizer(dataset['formatted_chat'],padding=True,max_length=args.max_length,truncation=True)

class ReftChatDataset(ReftDataset):
        """
        Chat dataset. We intervene on a prefix + suffix
        of the input. This is suitable for supervised fine-tuning tasks.
        You should overwrite load_dataset for your specific task.
        """
        
        def load_dataset(self):
            """Load the dataset from HF or a local file. to self.task_dataset."""

            
            task_dataset = datasets.load_dataset(self.task[0], split=self.data_split)
            task_dataset = task_dataset.shuffle()
            
            # select n random examples if specificed
            if self.max_n_example is not None:
                task_dataset = task_dataset.shuffle(seed=self.seed)
                task_dataset = task_dataset.select(range(self.max_n_example))
                
            return task_dataset
    
        def postprocess(self, kwargs):
            '''
            define how to preprocess for a single data_item
            the input is a list of dictionary, the output should be two lists (write to self.input_field and self.output_field)
            ### speeding up: we may tokenize the text here, but we have to know the last position of the prefix
            '''
            # we need to apply the chat template here
            self.task_dataset = self.task_dataset.map(apply_chat_template,batched=True,remove_columns=['conversation'],batch_size=10000,num_proc=32)
            
            # we may opt to tokenize the dataset here
            # self.task_dataset = tokenized_dataset(self.task_dataset)
            # we assume that the prefix is "<|assistant|>\n" and "<|assistant|>\n" has token 32001
            # self.last_prefix_token = (self.tokenizer.encode('<|assistant|>\n',add_special_tokens=False))[0]
            # print("last_prefix_token: ",self.last_prefix_token)
            # we will iterate the dataset 
        
        
        def tokenize(self, data_item):
            '''
            the tokenize function is a function that takes a single data_item and returns a dictionary and a last_position
            return a result dictionary and last postion 
            "result" is a  dictionary with keys "input_ids" and "labels"
            "input_ids" is the full sequence
            "labels" is the sequence with the prefix replaced with IGNORE_INDEX
            
            "last_position" is the last position of the prefix
            #TO DO: the prefix may have longer length then 2048
            '''
            result = {}
            # compute the length of the user question
            text = data_item["formatted_chat"]
            # this must set according to chat template
            # tinyllama
            base_prompt = text.split("<|assistant|>\n")[0] + "<|assistant|>\n"
            # print(base_prompt)
            prompt_ids = self.tokenizer(base_prompt, max_length=self.tokenizer.model_max_length,
                truncation=True, return_tensors="pt")["input_ids"][0]
            base_prompt_length = len(prompt_ids)
            last_position = base_prompt_length - 1
            # this compute the whole input (input+output)
            base_input = text + self.tokenizer.eos_token
            input_ids = self.tokenizer(base_input, max_length=self.tokenizer.model_max_length,
                truncation=True, return_tensors="pt")["input_ids"][0]
            result["input_ids"] = input_ids

            # labels
            output_ids = copy.deepcopy(input_ids)
            output_ids[:base_prompt_length] = IGNORE_INDEX # ignore the instruction for backpropagation
            result["labels"] = output_ids
            
            return result, last_position
        
if __name__ == "__main__":
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name_or_path,torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2",device_map=args.device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    tokenizer.model_max_length = args.max_length
    # print(model.layers)
    # model.resize_token_embeddings(len(tokenizer))
    
    # get reft model
    if args.layers != "all":
        layers = [int(l) for l in args.layers.split(";")]
    else:
        temp_config = transformers.AutoConfig.from_pretrained(args.model_name_or_path)
        layers = [l for l in range(temp_config.num_hidden_layers)]
        # get all the layer output
        
        
    if "+" in args.position and not args.share_weights:
        layers += layers
    
    
    # this is a brute force way to intervene on all layers (not optimal)
    # representations = [{
    #     "layer": l, "component": "block_output","low_rank_dimension": args.rank,
    #     "intervention": pyreft.LoreftIntervention(
    #         embed_dim=model.config.hidden_size, 
    #         low_rank_dimension=args.rank,
    #     )
    # } for l in layers]
    
    representations= [{
        "component": f"model.layers[{l}].mlp.output", # string access to the model component
        "intervention": pyreft.LoreftIntervention(
        embed_dim=model.config.hidden_size, low_rank_dimension=args.rank)} for l in layers]
    
    reft_config = pyreft.ReftConfig(representations=representations)
    reft_model = pyreft.get_reft_model(model, reft_config)
    reft_model.set_device(args.device)
    reft_model.print_trainable_parameters()
    
    # ##---------------------------------data processing----------------------
    # note we don't need to apply the chat template here, as the ReftChatDataset will do it for us at preprocessing stage
    train_dataset = ReftChatDataset(task=['benchang1110/medicaltw'],data_path=None,data_split='train',max_n_example=args.samples,
                                    tokenizer=tokenizer, model=reft_model,
                                    **{"num_interventions": len(layers), "position": args.position, 
                                    "share_weights":args.share_weights})
    
    
    
    print(train_dataset)
    # # #---------------------------------training---------------------------------
    training_args = TrainingArguments(
        output_dir=args.save, #The output directory
        overwrite_output_dir=True, #overwrite the content of the output directory
        num_train_epochs=args.epoch, # number of training epochs
        per_device_train_batch_size=args.batch_size, # batch size for training
        learning_rate=args.lr,
        weight_decay = 0,
        warmup_ratio = 0.1,
        max_grad_norm=1.0, #gradient clipping
        disable_tqdm=False,
        bf16=True,
        gradient_accumulation_steps=1,
        
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps= 10,
        save_steps=1000,
        save_total_limit=1,
        
        dataloader_num_workers=args.batch_size,
        dataloader_pin_memory=False,
        remove_unused_columns=True,
        gradient_checkpointing=False,
        auto_find_batch_size=True
    )
    
    data_collator_fn = transformers.DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding="longest"
    )
    data_collator = pyreft.ReftDataCollator(data_collator=data_collator_fn)
    
    trainer = pyreft.ReftTrainerForCausalLM(model=reft_model, tokenizer=tokenizer, args=training_args,train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
    _ = trainer.train()
    # trainer.train(resume_from_checkpoint=True)
    
    reft_model.set_device("cpu") # send back to cpu before saving.
    reft_model.save(save_directory=args.export)
    
    
    
