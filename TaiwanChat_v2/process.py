from datasets import Dataset
import json
import regex
import datasets
from datasets import disable_caching
disable_caching()

def gen(filenames):
    for filename in filenames:
        with open(filename, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                for i in data:
                    yield {"conversation": i}
            except:
                print(f'Error in {filename}')
                pass


filenames = ['TaiwanChat[:10%].json','TaiwanChat[11%:20%].json','TaiwanChat[21%:30%].json','TaiwanChat[31%:40%].json']
dataset = Dataset.from_generator(lambda: gen(filenames),cache_dir="./")

# make multiple times of \n to a signle \n

def clean_newline(batch):
    # make multiple times of \n to a signle \n
    batch['conversation'] = [[{'content': regex.sub(r'\n+', '\n', message['content']), 'role': message['role']} for message in conversation] for conversation in batch['conversation']]
    return batch

def count_elements(sample,threshold=0.2):
    # print(sample['conversation'][1]['content'])
    # sample is a list of dictionary, select the 'content' key
    sample = regex.sub(r'\p{P}+', ' ', sample['conversation'][1]['content'])  # Remove all punctuation characters
    count = {}
    total_words = 0
    words = sample.split()
    for word in words:
        if any('\u4e00' <= char <= '\u9fff' for char in word):  # Check if the word contains Chinese characters
            for char in word:
                count[char] = count.get(char, 0) + 1
            total_words += len(word)  # Count each Chinese character as a word
        else:
            count[word] = count.get(word, 0) + 1
            total_words += 1  # Count each English word as a word
    if total_words == 0:
        return False     
    if len(count.keys())/ total_words < threshold :
        print(sample)
        return False
    return True

dataset = dataset.map(clean_newline, batched=True) # remove multiple newlines
original_len = len(dataset)
dataset = dataset.filter(function=count_elements) # remove the sample with duplicated text
cleaned_len = len(dataset)
print(original_len, cleaned_len)
