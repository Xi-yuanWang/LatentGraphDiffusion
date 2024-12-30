# from utils.graph_utils import mask_adjs, pow_tensor
import torch.nn as nn
import torch
from typing import Callable, Final
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.nn import global_mean_pool
from pygho.honn.Conv import NGNNConv, GNNAKConv, DSSGNNConv, SSWLConv, SUNConv, PPGNConv, I2Conv, IGN2Conv
from pygho.backend.MaTensor import MaskedTensor
from lgd.model.utils import num2batch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import act_dict

def transfermlpparam(mlp: dict):
    mlp = mlp.copy()
    mlp.update({"tailact": True, "numlayer": 2, "norm": "ln"})
    return mlp

def get_first_node(batch, graph_idx: int):
    indices = torch.nonzero(batch.batch == graph_idx, as_tuple=False)[0]
    assert indices.numel() > 0, "Value not found in the tensor"
    return int(indices[0].item())

maconvdict = {
    "SSWL":
    lambda dim, mlp: SSWLConv(dim, dim, "sum", "DD", transfermlpparam(mlp)
                              ),
    "DSSGNN":
    lambda dim, mlp: DSSGNNConv(dim, dim, "sum", "sum", "mean",
                                "DD", transfermlpparam(mlp)),
    "GNNAK":
    lambda dim, mlp: GNNAKConv(dim, dim, "sum", "mean", "DD",
                               transfermlpparam(mlp), transfermlpparam(mlp)),
    "SUN":
    lambda dim, mlp: SUNConv(dim, dim, "sum", "mean", "DD",
                             transfermlpparam(mlp), transfermlpparam(mlp)),
    "NGNN":
    lambda dim, mlp: NGNNConv(dim, dim, "sum", "DD", transfermlpparam(mlp)
                              ),
    "PPGN":
    lambda dim, mlp: PPGNConv(dim, dim, "sum", "DD", transfermlpparam(mlp)),
    "2IGN":
    lambda dim, mlp: IGN2Conv(dim, dim, "sum", "D", transfermlpparam(mlp))
}

class MaModel(nn.Module):
    residual: Final[bool]
    def __init__(self,
                 convfn: Callable,
                 num_layer=6,
                 hiddim=128,
                 residual=True,
                 mlp: dict = {}):
        super().__init__()
        self.residual = residual
        self.subggnns = nn.ModuleList(
            [convfn(hiddim, mlp) for _ in range(num_layer)])

    def forward(self, A: MaskedTensor, X: MaskedTensor):
        '''
        TODO: !warning input must be coalesced
        '''
        for conv in self.subggnns:
            tX = conv.forward(A, X, {})
            if self.residual:
                X = X.add(tX, samesparse=True)
            else:
                X = tX
        return X

class DenoisingHOGNN(nn.Module):

    # def __init__(self, max_feat_num, max_node_num, nhid, num_layers, num_linears, 
    #                 c_init, c_hid, c_final, adim, num_heads=4, conv='GCN'):
    def __init__(self, dim_in=0, dim_out=0, **kwargs):

        super().__init__()
        self.in_dim = cfg.hognn.in_dim
        self.hid_dim = cfg.hognn.hid_dim
        self.out_dim = cfg.hognn.out_dim
        self.num_layers = cfg.hognn.num_layers
        self.act = act_dict[cfg.hognn.act]() if cfg.hognn.act is not None else nn.Identity()
        self.subgnn_type = cfg.hognn.type
        self.temb = nn.Sequential(nn.Linear(1, self.hid_dim), nn.SiLU(inplace=True), nn.Linear(self.hid_dim, self.hid_dim), nn.SiLU(inplace=True))

        self.gnn = MaModel(maconvdict[self.subgnn_type], self.num_layers, self.hid_dim, residual=True)
        # self.final = nn.Sequential(nn.Linear(c_hid, c_hid), nn.SiLU(inplace=True), nn.Linear(c_hid, c_hid), nn.SiLU(inplace=True), nn.Linear(c_hid, 1))
        self.mlp = nn.Sequential(nn.Linear(self.hid_dim, self.hid_dim), nn.SiLU(inplace=True), nn.Linear(self.hid_dim, self.hid_dim))
        
        self.node_in_mlp = nn.Sequential(nn.Linear(self.in_dim, 2 * self.hid_dim), self.act,
                                         nn.Linear(2 * self.hid_dim, self.hid_dim))
        self.edge_in_mlp = nn.Sequential(nn.Linear(self.in_dim, 2 * self.hid_dim), self.act,
                                         nn.Linear(2 * self.hid_dim, self.hid_dim))
        
        self.final_layer_node = nn.Sequential(nn.Linear(self.hid_dim, 2 * self.out_dim), self.act,
                                              nn.Linear(2 * self.out_dim, self.out_dim))
        self.final_layer_edge = nn.Sequential(nn.Linear(self.hid_dim, 2 * self.out_dim), self.act,
                                              nn.Linear(2 * self.out_dim, self.out_dim))
        
    def forward(self, batch, t=None, prompt=None, **kwargs):
        '''
        x: (B, N, d)
        adj: (B, N, N)
        '''
        batch_num_node = batch.num_node_per_graph
        batch_node_idx = num2batch(batch_num_node)
        assert torch.equal(batch_node_idx, batch.batch)
        batch_edge_idx = num2batch(batch_num_node ** 2)
        batch.batch_node_idx, batch.batch_edge_idx = batch_node_idx, batch_edge_idx
        
        batch.x = self.node_in_mlp(batch.x) # dim 4 -> 64
        batch.edge_attr = self.edge_in_mlp(batch.edge_attr) # dim 4 -> 64
        
        # print(f"GRAPH ATTRIBUTE AT THE BEGINNING: {batch.graph_attr}")
        x, mask = to_dense_batch(batch.x, batch.batch) # mask是一个num_graphs(B)×num_nodes_per_graph(N)的矩阵，如果x里面有些node_representation是pooling的结果，这些位置就会在mask中被标记为False，表示占位值
        adj = to_dense_adj(batch.edge_index, batch.batch, batch.edge_attr)
        
        adj = adj * self.temb(t.to(device=adj.device, dtype=adj.dtype).reshape(-1, 1, 1, 1))
        
        X = MaskedTensor(adj,
                 torch.logical_and(mask.unsqueeze(1), mask.unsqueeze(2)),
                 0.0,
                 True)
        A = X
        X: MaskedTensor = self.gnn(A, X)
        X = X.data # (B, N, N, d)

        X = X + X.transpose(1, 2)
        
        batch.edge_attr = torch.empty(batch.edge_attr.shape).to("cuda")
        for edge_idx in range(batch.edge_index.shape[1]):
            from_node = batch.edge_index[0][edge_idx]
            to_node = batch.edge_index[1][edge_idx]
            from_node_idx = from_node - get_first_node(batch, batch.batch[from_node])
            to_node_idx = to_node - get_first_node(batch, batch.batch[to_node])
            batch.edge_attr[edge_idx] = X[batch.batch[from_node]][from_node_idx][to_node_idx] # 这里index batch用的index应该用from_node和to_node都行
            
        batch.x = self.mlp(X).mean(dim=-2)[mask] # (B, N, N, d) -> (B, N, d)
        
        batch.x = self.final_layer_node(batch.x) # dim 64 -> 4
        batch.edge_attr = self.final_layer_edge(batch.edge_attr) # dim 64 -> 4
        batch.graph_attr = global_mean_pool(batch.x, batch.batch)
        # print(f"graph_attr.shape: {batch.graph_attr.shape}")
        return batch