# from utils.graph_utils import mask_adjs, pow_tensor
import torch.nn as nn
import torch
from typing import Callable, Final
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.nn import global_mean_pool
from pygho.honn.Conv import NGNNConv, GNNAKConv, DSSGNNConv, SSWLConv, SUNConv, PPGNConv, I2Conv, IGN2Conv
from pygho.backend.MaTensor import MaskedTensor
from lgd.model.utils import num2batch
from torch_geometric.graphgym.register import register_network
import torch_geometric.graphgym.register as register
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

@register_network('HOGNNEncoder')
class HOGNNEncoder(nn.Module):

    # def __init__(self, max_feat_num, max_node_num, nhid, num_layers, num_linears, 
    #                 c_init, c_hid, c_final, adim, num_heads=4, conv='GCN'):
    def __init__(self, dim_in=0, dim_out=0, cfg=None):
        super().__init__()
        
        num_layers = cfg.subgnn.get("num_layers", 6)
        self.decode_dim = cfg.get("decode_dim", 1)
        self.node_dict_dim = cfg.get('node_encoder_num_types', 1)  
        self.edge_dict_dim = cfg.get('edge_encoder_num_types', 1)
        self.in_dim = cfg.in_dim
        self.hid_dim = cfg.hid_dim
        self.out_dim = cfg.out_dim
        self.posenc_in_dim = cfg.posenc_in_dim
        self.posenc_in_dim_edge = cfg.posenc_in_dim_edge
        self.posenc_dim = cfg.posenc_dim
        self.subgnn_type = cfg.subgnn.type
        self.act = act_dict[cfg.act]() if cfg.act is not None else nn.Identity()
        
        self.temb = nn.Sequential(nn.Linear(1, self.hid_dim), nn.SiLU(inplace=True), nn.Linear(self.hid_dim, self.hid_dim), nn.SiLU(inplace=True))

        self.gnn = MaModel(maconvdict[self.subgnn_type], num_layers, self.hid_dim, residual=True)
        # self.final = nn.Sequential(nn.Linear(c_hid, c_hid), nn.SiLU(inplace=True), nn.Linear(c_hid, c_hid), nn.SiLU(inplace=True), nn.Linear(c_hid, 1))
        self.mlp = nn.Sequential(nn.Linear(self.hid_dim, self.hid_dim), nn.SiLU(inplace=True), nn.Linear(self.hid_dim, self.hid_dim))
        
        if cfg.node_encoder_name == 'Embedding':
            self.node_emb = nn.Embedding(self.node_dict_dim, self.in_dim, padding_idx=0)
        else:
            NodeEncoder = register.node_encoder_dict[cfg.node_encoder_name]
            self.node_emb = NodeEncoder(self.in_dim) # embedding dim = 64
        if cfg.edge_encoder_name == 'Embedding':
            self.edge_emb = nn.Embedding(self.edge_dict_dim, self.in_dim, padding_idx=0)
        else:
            EdgeEncoder = register.edge_encoder_dict[cfg.edge_encoder_name]
            self.edge_emb = EdgeEncoder(self.in_dim) # embedding dim = 64
        
        if self.posenc_in_dim > 0 and self.posenc_dim > 0:
            if cfg.get('pe_raw_norm', None) == 'BatchNorm':
                pe_raw_norm_node = nn.BatchNorm1d(self.posenc_in_dim, track_running_stats=not self.bn_no_runner,
                                                  eps=1e-5, momentum=self.bn_momentum)
            elif cfg.get('pe_raw_norm', None) == 'LayerNorm':
                pe_raw_norm_node = nn.LayerNorm(self.posenc_in_dim)
            else:
                pe_raw_norm_node = nn.Identity()
            self.posenc_emb = nn.Sequential(pe_raw_norm_node, nn.Linear(self.posenc_in_dim, self.posenc_dim))
            # self.posenc_emb = nn.Sequential(pe_raw_norm_node, nn.Linear(self.posenc_in_dim, 2 * self.posenc_dim), self.act,
            #                                 nn.Linear(2 * self.posenc_dim, self.posenc_dim))
        if self.posenc_in_dim_edge > 0 and self.posenc_dim > 0:
            if cfg.get('pe_raw_norm', None) == 'BatchNorm':
                pe_raw_norm_edge = nn.BatchNorm1d(self.posenc_in_dim_edge, track_running_stats=not self.bn_no_runner,
                                                  eps=1e-5, momentum=self.bn_momentum)
            elif cfg.get('pe_raw_norm', None) == 'LayerNorm':
                pe_raw_norm_edge = nn.LayerNorm(self.posenc_in_dim_edge)
            else:
                pe_raw_norm_edge = nn.Identity()
            # self.posenc_emb_edge = nn.Sequential(pe_raw_norm_edge, nn.Linear(self.posenc_in_dim_edge, 2 * self.posenc_dim), self.act,
            #                                      nn.Linear(2 * self.posenc_dim, self.posenc_dim))
            self.posenc_emb_edge = nn.Sequential(pe_raw_norm_edge, nn.Linear(self.posenc_in_dim_edge, self.posenc_dim))

        self.node_in_mlp = nn.Sequential(nn.Linear(self.hid_dim, 2 * self.hid_dim), self.act,
                                         nn.Linear(2 * self.hid_dim, self.hid_dim))
        self.edge_in_mlp = nn.Sequential(nn.Linear(self.hid_dim, 2 * self.hid_dim), self.act,
                                         nn.Linear(2 * self.hid_dim, self.hid_dim))
        
        self.final_layer_node = nn.Sequential(nn.Linear(self.hid_dim, 2 * self.out_dim), self.act,
                                              nn.Linear(2 * self.out_dim, self.out_dim))
        self.final_layer_edge = nn.Sequential(nn.Linear(self.hid_dim, 2 * self.out_dim), self.act,
                                              nn.Linear(2 * self.out_dim, self.out_dim))
        
        self.decode_node = nn.Linear(self.out_dim, self.node_dict_dim)  # virtual node
        self.decode_edge = nn.Linear(self.out_dim, self.edge_dict_dim)  # no edge, virtual edge, loop
        self.decode_graph = nn.Linear(self.out_dim, self.decode_dim)

    def forward(self, batch, t=None, prefix=None, label=None, **kwargs):
        '''
        x: (B, N, d)
        adj: (B, N, N)
        '''
        batch_num_node = batch.num_node_per_graph
        batch_node_idx = num2batch(batch_num_node)
        assert torch.equal(batch_node_idx, batch.batch)
        batch_edge_idx = num2batch(batch_num_node ** 2)
        batch.batch_node_idx, batch.batch_edge_idx = batch_node_idx, batch_edge_idx
        num_nodes, num_edges = batch.num_nodes, batch.edge_index.shape[1]
        
        h = self.node_emb(batch.x).reshape(num_nodes, -1) # ? -> 64
        e = self.edge_emb(batch.edge_attr).reshape(num_edges, -1) # ? -> 64
        
        if self.posenc_dim > 0:
            if self.posenc_in_dim > 0 and batch.get("pestat_node", None) is not None:
                batch_posenc_emb = self.posenc_emb(batch.pestat_node)
            else:
                batch_posenc_emb = torch.zeros([num_nodes, self.posenc_dim], dtype=torch.float, device=batch.x.device)
            if self.posenc_in_dim_edge > 0 and batch.get("pestat_edge", None) is not None:
                batch_posenc_emb_edge = self.posenc_emb_edge(batch.pestat_edge)
            else:
                batch_posenc_emb_edge = torch.zeros([num_edges, self.posenc_dim], dtype=torch.float, device=batch.x.device)
            # h = torch.cat([h, prefix_h, batch_posenc_emb], dim=1)
            # e = torch.cat([e, prefix_e, batch_posenc_edge], dim=1)
            h = torch.cat([h, batch_posenc_emb], dim=1)
            e = torch.cat([e, batch_posenc_emb_edge], dim=1)
            
        
        h = self.node_in_mlp(h) # 64 -> 64
        e = self.edge_in_mlp(e) # 64 -> 64
        
        # print(f"GRAPH ATTRIBUTE AT THE BEGINNING: {batch.graph_attr}")
        x, mask = to_dense_batch(h, batch.batch) # mask是一个num_graphs(B)×num_nodes_per_graph(N)的矩阵，如果x里面有些node_representation是pooling的结果，这些位置就会在mask中被标记为False，表示占位值
        adj = to_dense_adj(batch.edge_index, batch.batch, e)
        
        if t is not None:
            adj = adj * self.temb(t.to(device=adj.device, dtype=adj.dtype).reshape(-1, 1, 1, 1))
        
        X = MaskedTensor(adj,
                 torch.logical_and(mask.unsqueeze(1), mask.unsqueeze(2)),
                 0.0,
                 True)
        A = X
        X: MaskedTensor = self.gnn(A, X)
        X = X.data # (B, N, N, d)

        X = X + X.transpose(1, 2)
        
        e = torch.empty(e.shape).to("cuda")
        for edge_idx in range(batch.edge_index.shape[1]):
            from_node = batch.edge_index[0][edge_idx]
            to_node = batch.edge_index[1][edge_idx]
            from_node_idx = from_node - get_first_node(batch, batch.batch[from_node])
            to_node_idx = to_node - get_first_node(batch, batch.batch[to_node])
            e[edge_idx] = X[batch.batch[from_node]][from_node_idx][to_node_idx] # 这里index batch用的index应该用from_node和to_node都行
            
        h = self.mlp(X).mean(dim=-2)[mask] # (B, N, N, d) -> (B, N, d)
        batch.x = self.final_layer_node(h)
        batch.edge_attr = self.final_layer_node(e)
        batch.graph_attr = global_mean_pool(batch.x, batch.batch)
        # print(f"graph_attr.shape: {batch.graph_attr.shape}")
        return batch
    
    def encode(self, batch, t=None, prefix=None, label=None, **kwargs):
        return self.forward(batch, t, prefix, label, **kwargs)
    
    def decode(self, batch, **kwargs):
        return self.decode_node(batch.x), self.decode_edge(batch.edge_attr), self.decode_graph(
            batch.graph_attr).flatten()