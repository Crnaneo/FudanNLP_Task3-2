import json
import torch
import random
from embedding import Embedding
from model import DecoderOnly  # 引入 DecoderOnly

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
# 注意：原代码中 batch_size 实际上发挥了 seq_len (序列长度) 的作用
batch_size = 1024  # 建议调整为1024，匹配 Embedding 的默认 max_len=1024，否则 position_embedding 会越界
epochs = 50
print(f"Using device: {device}")

floors = []
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for i in data:
    for j in data[i]:
        floors.append(j)

data = '<SOS>' + "<|endoftext|><SOS>".join(floors)

# 显式传入 max_len
embedding = Embedding(max_len=batch_size).to(device)
# 实例化 DecoderOnly，必须将 dim 设置为 512 以匹配 Embedding 的维度
model = DecoderOnly(embedding, dim=512).to(device)

criterion = torch.nn.CrossEntropyLoss(ignore_index=embedding.tokenizer.pad_token_id)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

tokens = embedding.tokenize(data)

# Decoder-Only 是自回归语言模型，不需要对 src 进行 [MASK] 掩码，直接通过自回归预测下一个 token

for epoch in range(epochs):
    model.train()
    # 按照 sequence length 进行切片
    for i in range(0, len(tokens) - 1, batch_size):
        # 多取 1 个 token 用于构造目标标签 (label)
        chunk = tokens[i: i + batch_size + 1]
        if len(chunk) < 2:
            continue

        chunk_data = torch.LongTensor(chunk).unsqueeze(0).to(device)

        # Decoder-Only 的输入和标签错位 1 个 token
        x = chunk_data[:, :-1]  # 输入
        y = chunk_data[:, 1:]  # 预测目标

        output = model(x)

        # 计算 Loss
        loss = criterion(output.reshape(-1, len(embedding.tokenizer)), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


def evaluate(text, max_len=512):
    model.eval()
    global device

    # 如果强制转回 cpu，模型也需要转回 cpu，否则会报错
    if device == torch.device("mps"):
        device = torch.device("cpu")
        model.to(device)

    with torch.no_grad():
        prompt_tokens = embedding.tokenize('<SOS>' + text)
        input_tensor = torch.LongTensor([prompt_tokens]).to(device)

        result_ids = []
        for i in range(max_len):
            # DecoderOnly 只需要传入当前的完整序列
            logits = model(input_tensor)

            # 取序列最后一个位置的输出作为下一个 token 的预测
            next_token_logits = logits[0, -1, :]
            next_token = torch.argmax(next_token_logits).item()

            # 遇到结束符则停止
            if next_token == embedding.tokenizer.convert_tokens_to_ids('<|endoftext|>'):
                break

            result_ids.append(next_token)

            # 将新预测的 token 拼接到输入序列中，继续下一次预测
            new_token_tensor = torch.LongTensor([[next_token]]).to(device)
            input_tensor = torch.cat([input_tensor, new_token_tensor], dim=1)

        prediction = "".join(embedding.tokenizer.decode(result_ids))
        return prediction


print(evaluate("讲一下复旦"))