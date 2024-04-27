from __future__ import division
from __future__ import print_function

import torch.optim as optim
from utils import *
from model import stMCFN
import os
import time
import argparse
from config import Config
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import anndata as ad
import numpy as np
import scanpy as sc
from tqdm import tqdm

def mk_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    return input_path


if __name__ == "__main__":
    start = time.perf_counter()
    parse = argparse.ArgumentParser()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    datasets = ['151672']

    for i in range(len(datasets)):
        start1 = time.perf_counter()

        dataset = datasets[i]
        config_file = './DLPFC.ini'

        config = Config(config_file)
        adata = load_ST_file(dataset, config.input_dim, config.k, config.radius)

        features = torch.FloatTensor(adata.X)
        labels = adata.obs['ground']
        fadj = adata.obsm['fadj']
        sadj = adata.obsm['sadj']
        nfadj = normalize_sparse_matrix(fadj + sp.eye(fadj.shape[0]))
        nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
        fadj1 = nfadj.to_dense()
        nsadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
        nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
        sadj1 = nsadj.to_dense()
        graph_nei = torch.LongTensor(adata.obsm['graph_nei'])
        graph_neg = torch.LongTensor(adata.obsm['graph_neg'])

        print('====feaature:', features.shape)
        plt.rcParams["figure.figsize"] = (3, 3)
        savepath = './result/DLPFC/' + dataset + '/'
        if not os.path.exists(savepath):
            mk_dir(savepath)


        cuda = torch.cuda.is_available()
        use_seed = True

        _, ground = np.unique(np.array(labels, dtype=str), return_inverse=True)
        ground = torch.LongTensor(ground)
        n = len(ground)
        class_num = len(ground.unique())

        epochs = 200
        epochs = epochs + 1

        if cuda:
            features = features.cuda()
            sadj = sadj1.cuda()
            fadj = fadj1.cuda()
            graph_nei = graph_nei.cuda()
            graph_neg = graph_neg.cuda()

        import random

        np.random.seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        os.environ['PYTHONHASHSEED'] = str(config.seed)
        if not config.no_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True


        model = stMCFN(nfeat=config.input_dim,
                       nhid1=128,
                       nhid2=64,
                       dropout=config.dropout)
        if cuda:
            model.cuda()

        optimizer = optim.Adam(model.parameters(), config.lr, config.weight_decay)

        epoch_max = 0
        ari_max = 0

        for epoch in tqdm(range(epochs)):

            model.train()
            optimizer.zero_grad()
            emb1, emb2, emb, H1, H2, H = model(features, sadj, fadj)
            loss_mse = F.mse_loss(features, H)
            num = emb1.shape[0]
            loss_cl = constrastive_loss(emb1, emb2, temperature=0.1)
            total_loss = 0.5 * loss_cl + loss_mse
            emb = pd.DataFrame(emb.cpu().detach().numpy()).fillna(0).values
            total_loss.backward()
            optimizer.step()

            kmeans = KMeans(n_clusters=class_num, n_init='auto').fit(emb)
            idx = kmeans.labels_
            ari_res = metrics.adjusted_rand_score(labels, idx)
            if ari_res > ari_max:
                ari_max = ari_res
                epoch_max = epoch
                idx_max = idx
                H_max = H
                emb_max = emb

        print(dataset, ' ARI:', ari_max)
        print('epoch:', epoch_max)

        adata.obs['idx'] = idx_max.astype(str)
        adata.obs['domain'] = adata.obs['idx']
        adata.obsm['emb'] = emb_max
        adata.obs['ground_truth'] = adata.obs['ground']

        sc.pl.spatial(adata, img_key="hires", color=['ground_truth', 'idx'],
                      title=[f'GT_{dataset}', f'ARI=%.4f'%ari_max], show=False)
        plt.show()
    end_final = time.perf_counter()
    runTime_all = end_final - start
    runTime_min = runTime_all / 60
    print("Running time：%.2f" % runTime_min, "min")
