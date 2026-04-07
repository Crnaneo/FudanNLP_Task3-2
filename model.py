from torch import nn;

class Model(nn.Module):
    def __init__(self, embedding,dim=512,n_headers=8,num_layers=3,hdim=2048, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = embedding
        self.dim = dim;
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_headers,
            dim_feedforward=hdim,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer,num_layers=num_layers);
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=n_headers,
            dim_feedforward=hdim,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer,num_layers=num_layers);
        self.fc_out = nn.Linear(dim,len(self.embedding.tokenizer))
    def forward(self,src_tokens,tgt_tokens):
        src_emb = self.embedding(src_tokens);
        tgt_emb = self.embedding(tgt_tokens);
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_tokens.size(1)).to(src_tokens.device);
        src_pad_mask = (src_tokens == self.embedding.tokenizer.pad_token_id)
        tgt_pad_mask = (tgt_tokens == self.embedding.tokenizer.pad_token_id)
        memory = self.encoder(src_emb, src_key_padding_mask=src_pad_mask);
        out = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask
        )
        return self.fc_out(out)

class DecoderOnly(nn.Module):
    def __init__(self, embedding, dim=128, n_head=8, num_layers=3,hdim=2048, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = embedding;
        self.dim = dim;
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_head,
            dim_feedforward=hdim,
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer,num_layers=num_layers);
        self.fc_out = nn.Linear(self.dim,len(self.embedding.tokenizer));

    def forward(self,x):
        x_emb = self.embedding(x);
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device);
        padding_mask = (x==self.embedding.tokenizer.pad_token_id);
        out = self.decoder(
            x_emb,mask = mask,src_key_padding_mask = padding_mask
        )
        return self.fc_out(out);

