from transformers import AutoTokenizer

model_checkpoint = "Helsinki-NLP/opus-mt-zh-en"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

zh_sentence = train_data[0]["chinese"]
en_sentence = train_data[0]["english"]

inputs = tokenizer(zh_sentence)
targets = tokenizer(text_target=en_sentence)

wrong_targets = tokenizer(en_sentence)
print(tokenizer.convert_ids_to_tokens(inputs["input_ids"]))
print(tokenizer.convert_ids_to_tokens(targets["input_ids"]))
print(tokenizer.convert_ids_to_tokens(wrong_targets["input_ids"]))