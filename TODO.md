lgd/ddpm/LGD.py里面的DiffusionWrapper可以接收参数diffusion_model，把diffusion_model改成NGNN即可，只需要让NGNN的输入能包含timestep embedding的信息，它就可以实现denoising的效果，具体输入输出格式和DenoisingTransformer（lgd/model/DenoisingTransformer.py）保持一样就行

Train encoder code: python pretrain.py --cfg cfg/hognn-encoder.yaml --repeat 5 wandb.use False
Train diffusion code: python train_diffusion.py --cfg cfg/my-diffusion_ddpm.yaml --repeat 5 wandb.use False
inference code: python inference.py --cfg cfg/my-diffusion_ddpm_pretrained.yaml

changes made: removed unsqueeze(-1)
modified c_dim to 4 to align with the dimensions of node and edge attributes (incomplete)
corrected function to compute edge_attributes -> Completed get_first_node function to achieve this transformation
added graph_attr computation (incomplete)

Current problems:
~~1. How to align c_dim and dim of node/edge attributes(4)？~~
align to 4 (information is enough), use mlp, * is better than +
如果直接把c_dim设成4，HOGNN里面temb、subgnn和mlp的隐层维度将全部为4，这样会不会影响效果？
感觉subgnn的中间维度不好提升，因为gnn的输入和输出维度都是不变的，如果想保持输入输出edge和node_attr都是4，最好所有隐层就都是4了。其他两个即便提升也只能提升中间层的维度，因为输入输出的维度必须保持为4。
~~3. What is virtual node? How to deal with the conditions in graph_attr calculation？~~
直接把所有带virtual node设成False，graph_attr就把所有的node_attr pool即可
~~4. How to complete the mapping between edge index and X? (已基本解决)~~
5. ZINC diffusion inference failed, debug inference code
~~试一试是不是virtual node的问题~~至少mydata不是virtual node的问题，注意diffusion的cfg，encoder部分要对齐
6. encoder需要改的模型就是GraphTransformerEncoder，注意forward中的prefix和label怎么处理？
直接不要
~~7. GraphTransformerEncoder的decode函数就是三个线性层，GraphTransformerDecoder有用吗？~~
√
经过check，发现LGD.py的inference中所使用的encode和decode都是first_stage_model也就是GraphTransformerEncoder的encode/forward和decode方法，并未使用其他方法，可以确定重点是GTE的forward函数改造。

Additional problems:
1. 根据观察，encoder接受的node和edge_attr输入是离散的，类型为int64，经过encoder后转化为float32，所以两种Denoising Network接受的输入都是float32，输出也是float32，后续计算都以float32类型完成。
GTE里面先给batch.x和batch.edge_attr过了两个encoder（TypeDictNode和TypeDictEdge），然后再对这些embedding做后续操作，这样可以保证embedding的类型是float。DT里面也有类似的变换，只不过不是encoder，而是两个mlp。HOGNN目前都没有使用类似的变换。（注意GTE里面也有那个node_in和edge_in_mlp）
![alt text](image.png)
这是GTE里面各个步骤中tensor的维数，如果不用posenc，那么in_dim和hid_dim就应该是相等的，否则hid_dim=in_dim+posenc_dim，最后输出前有一个mlp把hid_dim映射到out_dim
![alt text](image-1.png)
这是DT里面各个步骤中tensor的维数
![alt text](image-2.png)
HOGNNEncoder, 初始值是1？？
![alt text](image-3.png)
DenoisingHOGNN
2. 给HOGNNEncoder和DenoisingHOGNN都加了in_mlp和final_mlp，其中前者还有node和edge embedding层，posenc是没有的，后续可以考虑加上。在已经有embedding层的情况下，in_mlp不一定有用（如果没有posenc），先加上了。这样可以保证内部计算使用较大的64维向量，最后输出前映射成4。
3. 为什么train diffusion的时候保存的ckpt特别少？？？-> val和test loss都是单调递增的，为什么？？？
my encoder也有类似的问题，不过好像不是单调递增，而是从一个比较小

train好的：
my encoder 1000
my encoder 100
hognn encoder 100 (ppgn)
ngnn encoder 100
ngnn ddpm 100
ngnn ddpm
hognn ddpm 100
ngnn encoder 100 new (in tmux ngnn_encoder_100)
hognn encoder 100 new (ppgn) (in tmux hognn_encoder_100)
ngnn encoder
正在train的：(理论上hognn后面没加后缀的都是1000，可以自己check)
hognn encoder (ppgn)
hognn diffusion 100 (ppgn)
hognn diffusion 100 (ppgn) new
ngnn diffusion
ngnn diffusion 100
ngnn diffusion 100 new
ngnn encoder new (in tmux ngnn_encoder_new)

Some instructions on the new cfgs:
...complex means trained on the Complex Six Cycle dataset, otherwise means on the Six Cycle dataset
...old and ...new (correspond to HOGNN and HOGNN++ in the report): old encoders do not have dropout, batchnorm, and force undirected; old denoising networks do not have batchnorm and force undirected
...old和...new是相对的，如果有一类模型的cfg有...old，那么对应的无附注的cfg就是...new的模型结构；如果有一类模型的cfg有...new，那么对应的无附注的cfg就是...old的模型结构
...fast has the same model structure as ...old, but are trained with the improved version of the edge_attr extraction operation (faster)