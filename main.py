import json

from embedding import *
import torch;
import random;
from model import *

device=torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
batch_size = 2048;
epochs = 50;
print(device);
floors = []
with open("data.json","r",encoding="utf-8") as f:
    data=json.load(f);

for i in data:
    for j in data[i]:
        floors.append(j);

data ='<SOS>'+"<|endoftext|><SOS>".join(floors);
embedding = Embedding(batch_size).to(device);
model = Model(embedding).to(device);
criterion = nn.CrossEntropyLoss(ignore_index=embedding.tokenizer.pad_token_id)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

tokens = embedding.tokenize(data);
src = tokens.copy()
with torch.no_grad():
    for i in random.sample(range(len(src)), int(len(src)*0.15)):
        src[i] = embedding.tokenizer.convert_tokens_to_ids('[MASK]');
for epoch in range(epochs):
    model.train();
    for i in range(0,len(src),batch_size):
        src_data = torch.LongTensor(src[i:i+batch_size]).unsqueeze(0).to(device);
        tgt_data = torch.LongTensor(tokens[i:i+batch_size]).unsqueeze(0).to(device);
        src_tokens = src_data[:,:-1];
        tgt_input = tgt_data[:, :-1]
        tgt_label = tgt_data[:, 1:]
        output = model(src_tokens, tgt_input)
        loss = criterion(output.reshape(-1,len(embedding.tokenizer)),tgt_label.reshape(-1))
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

def evaluate(text,max_len=512):
    model.eval()
    global device;
    if(device == torch.device("mps")):
        device = torch.device("cpu");
    with torch.no_grad():
        src_test = torch.LongTensor([embedding.tokenize('<SOS>'+text)]).to(device)
        tgt_indices = torch.LongTensor([[embedding.tokenizer.convert_tokens_to_ids('<SOS>')]]).to(device)
        result_ids = []
        for i in range(max_len):
            logits = model(src_test, tgt_indices);
            next_token_logits = logits[0, -1, :]
            next_token = torch.argmax(next_token_logits).item()
            if next_token == embedding.tokenizer.convert_tokens_to_ids('<|endoftext|>'):
                break

            result_ids.append(next_token)
            new_token_tensor = torch.LongTensor([[next_token]]).to(device)
            tgt_indices = torch.cat([tgt_indices, new_token_tensor], dim=1)

        prediction = "".join(embedding.tokenizer.decode(result_ids));
        return prediction

print(evaluate("讲一下复旦"))
