from datasets import Dataset
import json
import re
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
    batch['conversation'] = [[{'content': re.sub(r'\n+', '\n', message['content']), 'role': message['role']} for message in conversation] for conversation in batch['conversation']]
    return batch


dataset = dataset.map(clean_newline, batched=True)

def count_elements(s):
    s = regex.sub(r'\p{P}+', ' ', s)  # Remove all punctuation characters
    count = {}
    total_words = 0
    words = s.split()
    for word in words:
        if any('\u4e00' <= char <= '\u9fff' for char in word):  # Check if the word contains Chinese characters
            for char in word:
                count[char] = count.get(char, 0) + 1
            total_words += len(word)  # Count each Chinese character as a word
        else:
            count[word] = count.get(word, 0) + 1
            total_words += 1  # Count each English word as a word
    return count, total_words, len(count.keys()), len(count.keys())/ total_words
