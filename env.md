conda create -n lgd
conda activate lgd
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install torchmetrics
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
pip install ogb
pip install matplotlib
pip install seaborn
pip install yacs
pip install tensorboardX
pip install wandb
pip install performer-pytorch
pip install pytorch_lightning
