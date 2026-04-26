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
from transformers import AutoTokenizer, MarianTokenizer

# 汉英翻译模型 opus-mt-zh-en 对应的分词器，源语言为中文
# model_checkpoint = "Helsinki-NLP/opus-mt-zh-en"
# 改成本地路径
model_checkpoint = r"D:/code/git-demo/transformer/models/opus-mt-en-zh"
# print(model_checkpoint)

# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer = MarianTokenizer.from_pretrained(model_checkpoint)

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

#  0423
'''
对于翻译任务，标签序列就是目标语言的 token ID 序列。与序列标注任务类似，我们会在预测
标签序列与答案标签序列之间计算损失来调整模型参数，因此我们同样需要将填充的 pad 字符设置为 -100，
以便在使用交叉熵计算序列损失时将它们忽略
'''
import torch

max_input_length = 128
max_target_length = 128

inputs = [train_data[s_idx]["chinese"] for s_idx in range(4)]
targets = [train_data[s_idx]["english"] for s_idx in range(4)]

model_inputs = tokenizer(
    inputs,
    padding=True,
    max_length=max_input_length,
    truncation=True,
    return_tensors="pt"
)
labels = tokenizer(
    text_target=targets,
    padding=True,
    max_length=max_target_length,
    truncation=True,
    return_tensors="pt"
)["input_ids"]

end_token_index = torch.where(labels == tokenizer.eos_token_id)[1]
for idx, end_idx in enumerate(end_token_index):
    labels[idx][end_idx+1:] = -100

print('batch_X shape:', {k: v.shape for k, v in model_inputs.items()})
print('batch_y shape:', labels.shape)
print(model_inputs)
print(labels)