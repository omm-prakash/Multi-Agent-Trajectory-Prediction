import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, opt, n_features, n_agent, use_agent_id, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # model architecture info
        self.d_model = opt.d_model
        self.nhead = opt.nhead
        self.n_encoder_layer = opt.n_encoder_layer
        self.n_decoder_layer = opt.n_decoder_layer
        self.dim_feedforward = opt.dim_feedforward
        self.drop = opt.drop
        self.batch_first = opt.batch_first
        self.embedding_dim = opt.embedding_dim
        self.embed_before_mlp = opt.embed_before_mlp
        self.in_mlp_layers = opt.in_mlp_layers
        self.out_mlp_layers = opt.out_mlp_layers
        self.dynamic_mask = opt.dynamic_mask
        self.use_agent_id = use_agent_id
        self.activation = nn.ReLU()

        # data specific info
        self.n_features = n_features
        self.n_agent = n_agent
        
        self.embedding = self.embedding_()
        self.in_mlp = self.in_mlp_()
        self.encoder = self.encoder_()
        self.out_mlp = self.out_mlp_()

    def init_params(self, layer):
        for name, param in layer.named_parameters():
            if 'weight' in name and param.data.dim() == 2:
                nn.init.kaiming_uniform_(param)

    def in_mlp_(self):
        assert self.in_mlp_layers[-1]==self.d_model, 'Please ensure the last layer of the MLP is same as d_model.'
        mlp = nn.Sequential()
        in_feats = self.embedding_dim+self.n_features if self.embed_before_mlp else self.n_features

        for (layer_idx, out_feats) in enumerate(self.in_mlp_layers):
            if (not self.embed_before_mlp) and (layer_idx == len(self.in_mlp_layers) - 1):
                assert self.d_model-self.embedding_dim>1, 'd_model must be greater than embedding_dim atleast by 1.'
                out_feats = out_feats - self.embedding_dim

            layer = nn.Linear(in_feats, out_feats)
            mlp.add_module(f"layer{layer_idx}", layer)
            mlp.add_module(f"activation{layer_idx}", self.activation)
            mlp.add_module(f'drop{layer_idx}', nn.Dropout(self.drop))
            in_feats = out_feats
        return mlp

    def embedding_(self):
        embedding = nn.Embedding(self.n_agent, self.embedding_dim)
        return embedding

    def pos_embedding(self, seq_len:int) -> None:
        pos_embedding = torch.empty(seq_len, self.d_model) # (seq_len, d_model)
        pos = torch.arange(0,seq_len, dtype=torch.float).unsqueeze(1) # (seq_len,1)
        denom = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float)*
                          (-torch.log(torch.tensor(10000.0))/self.d_model)) # (d_model/2)
        pos_embedding[:,0::2] = torch.sin(pos*denom) # (seq_len, d_model/2)
        pos_embedding[:,1::2] = torch.cos(pos*denom) # (seq_len, d_model/2)

        pos_embedding = pos_embedding.unsqueeze(0) # (1, seq_len, d_model)
        if not hasattr(self, 'pos_embedding'):
                self.register_buffer('pos_embedding', pos_embedding)
        return pos_embedding # (1, seq_len, d_model)

    def encoder_(self):
        encoder_layer = nn.TransformerEncoderLayer(self.d_model, 
                                                   self.nhead, 
                                                   self.dim_feedforward, 
                                                   self.drop, 
                                                   self.activation, 
                                                   batch_first=self.batch_first)
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_encoder_layer)
        return transformer_encoder
    
    def out_mlp_(self):
        mlp = nn.Sequential()
        in_feats = self.d_model
        for (layer_idx, out_feats) in enumerate(self.out_mlp_layers):
            layer = nn.Linear(in_feats, out_feats)
            layer.bias.data.zero_()
            mlp.add_module(f"layer{layer_idx}", layer)
            mlp.add_module(f"activation{layer_idx}", self.activation)
            mlp.add_module(f'drop{layer_idx}', nn.Dropout(self.drop))
            in_feats = out_feats
        layer = nn.Linear(in_feats, self.n_features-1)
        mlp.add_module(f"layer_final", layer)
        return mlp

    def forward(self, x):
        # expected x's shape: (batch, time, n_particles, n_features)
        if self.dynamic_mask:
            mask = self.dynamic_mask_(x.size(1), x.size(2)) 
        else:
            mask = self.stair_mask_(x.size(1), x.size(2))
        mask = mask.to(x.device)
        
        if self.use_agent_id:
            arr = torch.arange(0,self.n_agent,dtype=int)
            arr = arr.to(x.device)
            id_embedding = self.embedding(arr) # shape: (n_particles, embedding_dim)
            id_embedding = id_embedding.unsqueeze(0).unsqueeze(0) # shape: (1, 1, n_particles, embedding_dim)
            id_embedding = id_embedding.repeat(x.size(0), x.size(1), 1, 1) # shape: (batch, time, n_particles, embedding_dim)
            if self.embed_before_mlp:
                x = torch.concat([x, id_embedding], dim=-1) # shape: (batch, time, n_particles, n_features+embedding_dim)
                x = self.in_mlp(x) # shape: (batch, time, n_particles, d_model)
            else:
                x = self.in_mlp(x) # shape: (batch, time, n_particles, d_model)
                x = torch.concat([x, id_embedding], dim=-1) # shape: (batch, time, n_particles, n_features+embedding_dim)
                # x = x + id_embedding # shape: (batch, time, n_particles, d_model)
        else:
            # without agent ID
            self.embedding_dim = 0
            x = self.in_mlp(x) # shape: (batch, time, n_particles, d_model)

        x = x.view(x.size(0), -1, x.size(-1)) # shape: (batch, {time-1}*n_particles, d_model)
        
        # encoder
        x = self.encoder(x, mask) # shape: (batch, {time-1}*n_particles, d_model)
        
        # feed-forward neural network 
        x = x.reshape(-1, self.d_model) # shape: (batch*{time-1}*n_particles, d_model)
        x = self.out_mlp(x) # shape: (-1, n_features-1)
        return x

    def stair_mask_(self, time, agent):
        sz = time*agent
        mask = torch.zeros(sz, sz)
        for step in range(time):
            start = agent * step
            stop = start + agent
            # The players can look at the other players.
            mask[start:stop, :stop] = 1
        mask = mask==0
        mask.requires_grad_(False)
        if not hasattr(self, 'mask'):
            self.register_buffer('mask', mask)
        return mask # shape: (time*n_particles, time*n_particles)

    def dynamic_mask_(self, time, agent):
        sz = time*agent
        mask = torch.triu(torch.ones(sz,sz), diagonal=1)
        mask = mask==1
        mask.requires_grad_(False)
        if not hasattr(self, 'mask'):
            self.register_buffer('mask', mask)
        return mask # shape: (time*n_particles, time*n_particles)
