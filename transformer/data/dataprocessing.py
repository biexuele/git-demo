import torch
from torch.utils.data import Dataset, random_split
import json

max_dataset_size = 220000
train_set_size = 200000
valid_set_size = 20000


class TRANS(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx >= max_dataset_size:
                    break
                sample = json.loads(line.strip())
                Data[idx] = sample
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


data = TRANS('data/translation2019zh/translation2019zh_train.json')
train_data, valid_data = random_split(data, [train_set_size, valid_set_size])
test_data = TRANS('data/translation2019zh/translation2019zh_valid.json')

print(f'train set size: {len(train_data)}')
print(f'valid set size: {len(valid_data)}')
print(f'test set size: {len(test_data)}')
print(next(iter(train_data)))

'''
通过 DataLoader 库来按 batch 加载数据，将文本转换为模型可以接受的 token IDs。
'''
from transformers import AutoTokenizer

# 汉英翻译模型 opus-mt-zh-en 对应的分词器，源语言为中文
model_checkpoint = "Helsinki-NLP/opus-mt-zh-en"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

zh_sentence = train_data[0]["chinese"]
en_sentence = train_data[0]["english"]

inputs = tokenizer(zh_sentence)
# 要编码目标语言（英语）则需要使用 text_targets 参数
# 如果忘记使用 text_targets 参数，就会使用源语言分词器对目标语言进行编码，产生糟糕的分词结果
targets = tokenizer(text_target=en_sentence)

wrong_targets = tokenizer(en_sentence)
print(tokenizer.convert_ids_to_tokens(inputs["input_ids"]))
print(tokenizer.convert_ids_to_tokens(targets["input_ids"]))
print(tokenizer.convert_ids_to_tokens(wrong_targets["input_ids"]))