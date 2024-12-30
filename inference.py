import torch
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (cfg, dump_cfg,
                                             set_cfg, load_cfg,
                                             makedirs_rm_exist)
from torch_geometric.graphgym.loader import create_loader
import lgd.config.pretrained_config
import lgd.config.defaults_config
import lgd.config.split_config
import lgd.config.data_preprocess_config
import lgd.config.pretrained_config
import lgd.config.data_preprocess_config
import lgd.config.posenc_config
import lgd.config.gt_config
from lgd.config.optimizers_config import extended_optim_cfg
from lgd.loss.subtoken_prediction_loss import subtoken_cross_entropy
from lgd.loader.master_loader import load_dataset_master
from lgd.finetuning import load_pretrained_model_cfg, \
    init_model_from_pretrained
# from lgd.asset.stats import eval_graph_list
import networkx as nx
import matplotlib.pyplot as plt
import os
import random
from lgd.ddpm.LGD import DDPM, LatentDiffusion
@torch.no_grad()
def Inference(loader, model, split='val', ensemble_mode='none', evaluate=True):
    model.eval()
    generated_graph = []
    iter = 0
    for batch in loader[1]:
        if iter == 0 and evaluate:
            visualize = True
            iter += 1
        else:
            visualize = False
        batch.split = split
        batch.to(torch.device(cfg.accelerator))
        if cfg.gnn.head == 'inductive_edge':
            pred, true, extra_stats = model(batch)
        else:
            if ensemble_mode == 'none':
                node_label, edge_label, graph_label = batch.x.clone().detach().flatten(), batch.edge_attr.clone().detach().flatten(), batch.y
                # the embed of labels and prefix are done in fine-tuning of the encoder, not pretraining
                batch.x_masked = batch.x.clone().detach()
                batch.edge_attr_masked = batch.edge_attr.clone().detach()
                ddim_steps = cfg.diffusion.get('ddim_steps', None)
                ddim_eta = cfg.diffusion.get('ddim_eta', 0.0)
                use_ddpm_steps = cfg.diffusion.get('use_ddpm_steps', False)
                _, graph_pred = model.inference(batch, ddim_steps=ddim_steps, ddim_eta=ddim_eta, use_ddpm_steps=use_ddpm_steps, visualize=visualize)
                for each in graph_pred:
                    generated_graph.append(each)
                # logging.info('graph_pred')
                # logging.info(graph_pred)
                # pred, true = model(batch)
                # pred = model(batch)
                # node_pred, edge_pred, graph_pred = model.model.decode(pred)
            else:
                raise NotImplementedError
                # batch_pred = []
                # for i in range(repeat):
                #     bc = deepcopy(batch)
                #     bc.x_masked = batch.x.clone().detach()
                #     bc.edge_attr_masked = batch.edge_attr.clone().detach()
                #     loss_generation, loss_graph, graph_pred = model.validation_step(bc)
                #     batch_pred.append(graph_pred)
                #     del bc
                # batch_pred = torch.cat(batch_pred).reshape(repeat, -1)
                # if ensemble_mode == 'mean':
                #     graph_pred = torch.mean(batch_pred, dim=0)
                # else:
                #     graph_pred = torch.median(batch_pred, dim=0)[0]
            # pred, true = model(batch)
            extra_stats = {}
        if cfg.dataset.name == 'ogbg-code2':
            loss, pred_score = subtoken_cross_entropy(pred, true)
            _true = true
            _pred = pred_score
        else:
            true = batch.y  # TODO: check this
            # loss, pred_score = compute_loss(graph_pred, true)
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = true.detach().to('cpu', non_blocking=True)
            # logging.info(_pred)
        # logger.update_stats(true=_true,
        #                     pred=_pred,
        #                     loss=_.detach().cpu().item(),
        #                     lr=0, time_used=time.time() - time_start,
        #                     params=cfg.params,
        #                     dataset_name=cfg.dataset.name,
        #                     **extra_stats)
        # time_start = time.time()
    if evaluate:
        # result_dict = eval_graph_list(test_graph_list, generated_graph, methods=methods, kernels=kernels)
        # visualize generated molecules
        visualize_samples = random.sample(generated_graph, 5)
        save_path = 'generated_graphs'
        os.makedirs(save_path, exist_ok=True)

        for i in range(len(visualize_samples)):
            G = visualize_samples[i]
            # logging.info(G)
            labels = nx.get_node_attributes(G, 'label')
            pos = nx.spring_layout(G)
            nx.draw(G, pos)
            nx.draw_networkx_labels(G, pos, labels=labels)
            nx.draw_networkx_edge_labels(G, pos)
            plt.savefig(save_path + '/sample_' + str(i) + '.png')
            plt.clf()
            
if __name__ == "__main__":
    args = parse_args()
    # Load config file
    set_cfg(cfg)
    cfg.set_new_allowed(True)
    load_cfg(cfg, args)
    # print(cfg)
    dump_cfg(cfg)
    
    model = eval(cfg.model.get('type', 'LatentDiffusion'))\
    (timesteps=cfg.diffusion.get('timesteps', 1000), conditioning_key=cfg.diffusion.conditioning_key,
        hid_dim=cfg.diffusion.hid_dim, parameterization=cfg.diffusion.get("parameterization", "x0"),
        cond_stage_key=cfg.diffusion.cond_stage_key, first_stage_config=cfg.diffusion.first_stage_config,
        cond_stage_config=cfg.diffusion.cond_stage_config, edge_factor=cfg.diffusion.get("edge_factor", 1.0),
        graph_factor=cfg.diffusion.get("graph_factor", 1.0),
        train_mode=cfg.diffusion.get("train_mode", 'sample')).to(torch.device(cfg.accelerator))
    # model.to(torch.device(cfg.accelerator))
    if cfg.pretrained.dir:
        model = init_model_from_pretrained(
            model, cfg.pretrained.dir, cfg.pretrained.freeze_main,
            cfg.pretrained.reset_prediction_head
        )
    loader = create_loader()
    Inference(loader=loader, model=model)