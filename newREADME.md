环境配置
```
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
```


training encoder on my data
```
python pretrain.py --cfg cfg/my-encoder.yaml --repeat 5 wandb.use False
```

training diffusion model on my data
```
python train_diffusion.py --cfg cfg/my-diffusion_ddpm.yaml --repeat 5 wandb.use False
```

修改/新增数据集需要
1. 在datasets文件夹下创建路径mydata/raw，在mydata/raw放入数据文件
2. 在lgd loader下创建newdataset.py，在process函数中将数据文件载入并转为pyg data列表的形式
3. 修改master_loader.py, 像186-190行那样加入一个分支
4. 修改cfg/my-encoder.yaml, cfg/my-diffusion_ddpm.yaml这两个config文件。需要修改
    1. dataset.format改为Pyg-数据集名称
    2. node_encoder_num_types, edge_encoder_num_types 点，边特征的最大种类数
    3. cfg/my-diffusion_ddpm.yaml的first_stage_config改为encoder的ckpt路径

run command: python inference.py --cfg cfg/my-diffusion_ddpm_pretrained.yaml

训练encoder的code: python pretrain.py --cfg cfg/hognn-encoder.yaml --repeat 5 wandb.use False
训练diffusion的code: python train_diffusion.py --cfg cfg/my-diffusion_ddpm.yaml --repeat 5 wandb.use False
inference的code: python inference.py --cfg cfg/my-diffusion_ddpm_pretrained.yaml，跟my-diffusion_ddpm.yaml基本一样，区别就是在最后加三行pretrained，更改model.loss_fun为mse，更改train.mode为generic_generation，如my-diffusion_ddpm_pretrained.yaml所示。

在训练diffusion的cfg中，更改model.diffusion_model（“DenoisingTransformer”/“DenoisingHOGNN”）可以调整score network。如果使用hognn，需要在cfg中单独设置hognn的参数，如my-diffusion_ddpm.yaml所示。

在训练encoder的cfg中，更改model.type（“GraphTransformerEncoder”/“HOGNNEncoder”）可以调整encoder。如果使用hognn，需要在encoder.subgnn中设置type和num_layers参数（c_dim不用管），如hognn-encoder.yaml所示。

DenoisingHOGNN和HOGNNEncoder的code见lgd.model中的.py文件

mydata数据集我设置了两种，他们的format分别是PyG-mydata128和PyG-mydata1000，分别对应不同长度的dataset，更改dataset.format即可使用。