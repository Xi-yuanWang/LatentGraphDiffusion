from typing import Callable
from torch_geometric.data import InMemoryDataset
from typing import Callable, Any
import pickle
from torch_geometric.utils import from_networkx
import torch

def join_dataset_splits(datasets):
    """Join train, val, test loader into one dataset object.

    Args:
        datasets: list of 3 PyG loader to merge

    Returns:
        joint dataset with `split_idxs` property storing the split indices
    """
    assert len(datasets) == 3, "Expecting train, val, test loader"

    n1, n2, n3 = len(datasets[0]), len(datasets[1]), len(datasets[2])
    data_list = [datasets[0].get(i) for i in range(n1)] + \
                [datasets[1].get(i) for i in range(n2)] + \
                [datasets[2].get(i) for i in range(n3)]

    datasets[0]._indices = None
    datasets[0]._data_list = data_list
    datasets[0].data, datasets[0].slices = datasets[0].collate(data_list)
    split_idxs = [list(range(n1)),
                  list(range(n1, n1 + n2)),
                  list(range(n1 + n2, n1 + n2 + n3))]
    datasets[0].split_idxs = split_idxs

    return datasets[0]


class MyDataset(InMemoryDataset):
    # root -> path to your dataset
    def __init__(self, root: str = "datasets/mydata", transform: Callable[..., Any] | None = None, pre_transform: Callable[..., Any] | None = None, pre_filter: Callable[..., Any] | None = None, log: bool = True, force_reload: bool = False) -> None:
        super().__init__(root, transform, pre_transform, pre_filter, log, force_reload)
        self.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        # will produce raw_paths
        return ['six_cycle.pkl']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # download_url(url, self.raw_dir) # no need for local data
        pass

    def process(self):
        with open(self.raw_paths[0], "rb") as f:
            graphlist = pickle.load(f)
        data_list = [from_networkx(_) for _ in graphlist]
        for data in data_list:
            data.x = torch.ones((data.num_nodes, 1), dtype=torch.long)
            data.edge_attr = torch.ones((data.edge_index.shape[1]), dtype=torch.long)
            data.y = torch.ones((1,), dtype=torch.float)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

def preformat_mydata(dataset_dir, name):
    dataset = MyDataset()
    ret = join_dataset_splits([dataset[:80], dataset[80:90], dataset[90:]])
    return ret

if __name__ == "__main__":
    dataset = MyDataset()
    print(len(dataset))
    print(dataset[0])