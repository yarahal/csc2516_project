import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric import nn as geom_nn

class GAT(nn.Module):
    def __init__(self,config,device):
        super().__init__()
        self.output_size = config['hidden_size']
        self.n_heads = config['n_heads']
        self.radius = config['radius']
        self.embedding_size = config['embedding_size']
        self.device = device

    def create_layers(self):
        self.Hl = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.Hl.append(geom_nn.conv.GATConv(9,self.output_size,self.n_heads))
        self.norm_layers.append(geom_nn.norm.BatchNorm(self.output_size*self.n_heads))
        for _ in range(self.radius-1):
            self.Hl.append(geom_nn.conv.GATConv(self.output_size*self.n_heads,self.output_size,self.n_heads))
            self.norm_layers.append(geom_nn.norm.BatchNorm(self.output_size*self.n_heads))

        self.mlp = nn.Sequential(
            nn.Linear(self.output_size*self.n_heads,self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size,self.embedding_size)
        )
    
    def forward(self,x,edge_index,batch):
        batch_size = torch.unique(batch).size(0)
        f = torch.zeros(batch_size,self.hidden_size*self.n_heads,device=self.device)
        for i in range(self.radius):
            x = self.Hl[i](x, edge_index)
            x = self.norm_layers[i](x)
            x = F.relu(x)
            x_sparse = F.softmax(x,dim=1) 
            f = f + geom_nn.pool.global_add_pool(x_sparse,batch=batch)
        f = self.mlp(f)   
        return f
    

class SiameseNetwork(nn.Module):
    def __init__(self,gan_model, config, distance="cos"):
        super().__init__()

        self.hidden_size = config['hidden_size']
        self.n_heads = config['n_heads']
        self.radius = config['radius']
        self.embedding_size = config['embedding_size']
        self.distance = distance
        self.gnn = gan_model(config)

        self.gnn.create_layers()
        
        if self.distance == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(2*self.embedding_size,1),
                nn.Sigmoid()
            )

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def forward(self,molecule_A,molecule_B):
        x_A, edge_index_A, batch_A = molecule_A.x.float(), molecule_A.edge_index, molecule_A.batch
        x_B, edge_index_B, batch_B = molecule_B.x.float(), molecule_B.edge_index, molecule_B.batch

        # get embedding_A and embedding_B
        embedding_A = self.gnn(x_A,edge_index_A, batch_A)
        embedding_B = self.gnn(x_B,edge_index_B, batch_B)

        if self.distance == "cos":
            # compute cosine similarity between embedding_A and embedding_B
            score = F.cosine_similarity(embedding_A, embedding_B, dim = 1)
        elif self.distance == "mlp":
            x = torch.cat((embedding_A,embedding_B),dim=1)
            score = self.mlp(x).squeeze(1)

        # return predicted score
        return score