from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 指定模型名称即可对应加载
checkpoint="distilbert-base-uncased-finetuned-sst-2-english"
tokenizer=AutoTokenizer.from_pretrained(checkpoint)
model=AutoModelForSequenceClassification.from_pretrained(checkpoint)

# 测试数据
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!"
]
# 方法中，padding表示是否将输入调整为相同长度，通常是按照最长的句子作为依据做一个补齐（例如补0）
# truncation表示截断，通常用于限制句子的最大长度（通常是512）
# return_tensors表示输出pytorch格式的tensor
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)
# 可以看到反解析出来的结果多了[CLS],[SEP], 这些是模型训练的时候加入的
print(tokenizer.decode([101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172, 2607,  2026,  2878,  2166,  1012, 102]))
outputs=model(**inputs)
# 输出结果为2*2
print(outputs.logits.shape)

predications = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predications)