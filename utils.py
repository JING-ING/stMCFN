import scipy.sparse as sp
import sklearn
import torch
import networkx as nx
from sklearn.cluster import KMeans
import community as community_louvain
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch.nn.functional as F
import h5py
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import euclidean_distances

EPS = 1e-15
def load_data(dataset):
    print("load data:")
    path = "./generate_data/DLPFC/" + dataset + "/stMCFN.h5ad"
    adata = sc.read_h5ad(path)
    features = torch.FloatTensor(adata.X)
    labels = adata.obs['ground']
    fadj = adata.obsm['fadj']
    sadj = adata.obsm['sadj']
    nfadj = normalize_sparse_matrix(fadj + sp.eye(fadj.shape[0]))
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    nfadj1 = nfadj.to_dense()
    nsadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    nsadj1 =nsadj.to_dense()
    graph_nei = torch.LongTensor(adata.obsm['graph_nei'])
    graph_neg = torch.LongTensor(adata.obsm['graph_neg'])
    print("done")
    return adata, features, labels, nfadj, nfadj1, nsadj, nsadj1, graph_nei, graph_neg

def load_ST_file(dataset, highly_genes, k, radius):
    path = "/mnt/Data/DLPFC/" + dataset + "/"
    labels_path = path + "metadata.tsv"

    labels = pd.read_table(labels_path, sep='\t')
    labels = labels["layer_guess_reordered"].copy()
    NA_labels = np.where(labels.isnull())
    labels = labels.drop(labels.index[NA_labels])
    ground = labels.copy()

    ground.replace('WM', '7', inplace=True)
    ground.replace('Layer1', '1', inplace=True)
    ground.replace('Layer2', '2', inplace=True)
    ground.replace('Layer3', '3', inplace=True)
    ground.replace('Layer4', '4', inplace=True)
    ground.replace('Layer5', '5', inplace=True)
    ground.replace('Layer6', '6', inplace=True)

    adata1 = sc.read_visium(path, count_file='filtered_feature_bc_matrix.h5', load_images=True)
    adata1.var_names_make_unique()
    adata1 = adata1.copy()  ##
    obs_names = np.array(adata1.obs.index)
    positions = adata1.obsm['spatial']

    data = np.delete(adata1.X.toarray(), NA_labels, axis=0)
    obs_names = np.delete(obs_names, NA_labels, axis=0)
    positions = np.delete(positions, NA_labels, axis=0)

    adata = ad.AnnData(pd.DataFrame(data, index=obs_names, columns=np.array(adata1.var.index), dtype=np.float32))

    adata.var_names_make_unique()
    adata.obs['ground_truth'] = labels
    adata.obs['ground'] = ground
    adata.obsm['spatial'] = positions
    adata.obs['array_row'] = adata1.obs['array_row']
    adata.obs['array_col'] = adata1.obs['array_col']
    adata.uns['spatial'] = adata1.uns['spatial']
    adata.var['gene_ids'] = adata1.var['gene_ids']
    adata.var['feature_types'] = adata1.var['feature_types']
    adata.var['genome'] = adata1.var['genome']
    adata.var_names_make_unique()
    adata = normalize(adata, highly_genes=highly_genes)
    fadj = features_construct_graph(adata.X, k=k)
    sadj, graph_nei, graph_neg = spatial_construct_graph1(adata, radius=radius)

    adata.obsm["fadj"] = fadj
    adata.obsm["sadj"] = sadj
    adata.obsm["graph_nei"] = graph_nei.numpy()
    adata.obsm["graph_neg"] = graph_neg.numpy()
    adata.var_names_make_unique()
    return adata



def normalize(adata, highly_genes=3000):
    print("start select HVGs")
    sc.pp.filter_genes(adata, min_cells=100)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=highly_genes)
    adata = adata[:, adata.var['highly_variable']].copy()
    adata.X = adata.X / np.sum(adata.X, axis=1).reshape(-1, 1) * 10000
    sc.pp.scale(adata, zero_center=False, max_value=10)
    return adata

def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())

def constrastive_loss(z1, z2, temperature=0.2):
    sim_matrix = sim(z1, z2)
    f = lambda x: torch.exp(x / temperature)
    # f = lambda x: torch.exp(x / 1)
    matrix_t = f(sim_matrix)
    numerator = matrix_t.diag()
    denominator = torch.sum(matrix_t, dim=-1)
    loss = -torch.log(numerator / denominator).mean()
    return loss


def cosine_similarity(emb):
    mat = torch.matmul(emb, emb.T)
    norm = torch.norm(emb, p=2, dim=1).reshape((emb.shape[0], 1))
    mat = torch.div(mat, torch.matmul(norm, norm.T))
    if torch.any(torch.isnan(mat)):
        mat = _nan2zero(mat)
    mat = mat - torch.diag_embed(torch.diag(mat))
    return mat



def spatial_construct_graph1(adata, radius=450):

    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']
    A=np.zeros((coor.shape[0],coor.shape[0]))

    nbrs = sklearn.neighbors.NearestNeighbors(radius=radius).fit(coor)
    distances, indices = nbrs.radius_neighbors(coor, return_distance=True)

    for it in range(indices.shape[0]):
        A[[it] * indices[it].shape[0], indices[it]] = 1

    print('The graph contains %d edges, %d cells.' % (sum(sum(A)), adata.n_obs))
    print('%.4f neighbors per cell on average.' % (sum(sum(A)) / adata.n_obs))

    graph_nei = torch.from_numpy(A)
    graph_neg = torch.ones(coor.shape[0],coor.shape[0]) - graph_nei
    sadj = sp.coo_matrix(A, dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)

    return sadj, graph_nei, graph_neg

def spatial_construct_graph(adata, k=15):
    print("start spatial construct graph")

    positions = pd.DataFrame(adata.obsm['spatial'])
    A = euclidean_distances(positions)
    tmp = 0
    mink = 2
    for t in range(100, 1000, 100):
        A1 = np.where(A > t, 0, 1)
        if mink < np.min(np.sum(A1, 1)) and k < np.max(np.sum(A1, 1)):
            tmp = t
            break
    for t in range(tmp - 100, 1000, 10):
        A1 = np.where(A > t, 0, 1)
        if mink < np.min(np.sum(A1, 1)) and k < np.max(np.sum(A1, 1)):
            tmp = t
            break
    for t in range(tmp - 10, 1000, 5):
        A1 = np.where(A > t, 0, 1)
        if mink < np.min(np.sum(A1, 1)) and k < np.max(np.sum(A1, 1)):
            A = A1
            break
    row, col = np.diag_indices_from(A)
    A[row, col] = 0
    graph_nei = torch.from_numpy(A)
    graph_neg = torch.ones(positions.shape[0], positions.shape[0]) - graph_nei

    sadj = sp.coo_matrix(A, dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)

    return sadj, graph_nei, graph_neg


def features_construct_graph1(features, k=15, pca=None, mode="connectivity", metric="cosine"):
    from sklearn.metrics import pairwise_distances

    pca = PCA(n_components=50)

    data_pca = pca.fit_transform(features.toarray())
    gene_correlation = 1 - pairwise_distances(data_pca, metric="cosine")
    return gene_correlation


    print("start features construct graph")
    if pca is not None:
        features = dopca(features, dim=pca).reshape(-1, 1)

    A = kneighbors_graph(features, k + 1, mode=mode, metric=metric, include_self=True)
    A = A.toarray()
    row, col = np.diag_indices_from(A)
    A[row, col] = 0

    fadj = sp.coo_matrix(A, dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)

    return fadj



def features_construct_graph(features, k=15, pca=None, mode="connectivity", metric="cosine"):
    print("start features construct graph")
    if pca is not None:
        features = dopca(features, dim=pca).reshape(-1, 1)  # (4221,3000)

    A = kneighbors_graph(features, k + 1, mode=mode, metric=metric, include_self=True)

    A = A.toarray()

    row, col = np.diag_indices_from(A)
    A[row, col] = 0
    fadj = sp.coo_matrix(A, dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)

    return fadj

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    # -*- coding : utf-8-*-
    # coding:unicode_escape

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def get_adj(data, pca=None, k=25, mode="connectivity", metric="cosine"):
    if pca is not None:
        data = dopca(data, dim=pca)
        data = data.reshape(-1, 1)
    A = kneighbors_graph(data, k, mode=mode, metric=metric, include_self=True)
    adj = A.toarray()
    adj_n = norm_adj(adj)
    # S = cosine_similarity(data)
    return adj, adj_n  # , S


def norm_adj(A):
    normalized_D = degree_power(A, -0.5)
    output = normalized_D.dot(A).dot(normalized_D)
    return output


def dopca(data, dim=50):
    return PCA(n_components=dim).fit_transform(data)


def degree_power(A, k):
    degrees = np.power(np.array(A.sum(1)), k).flatten()
    degrees[np.isinf(degrees)] = 0.
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D


class louvain:
    def __init__(self, level):
        self.level = level
        return

    def updateLabels(self, level):
        # Louvain algorithm labels community at different level (with dendrogram).
        # Here we want the community labels at a given level.
        level = int((len(self.dendrogram) - 1) * level)
        partition = community_louvain.partition_at_level(self.dendrogram, level)
        # Convert dictionary to numpy array
        self.labels = np.array(list(partition.values()))
        return

    def update(self, inputs, adj_mat=None):
        """Return the partition of the nodes at the given level.

        A dendrogram is a tree and each level is a partition of the graph nodes.
        Level 0 is the first partition, which contains the smallest communities,
        and the best is len(dendrogram) - 1.
        Higher the level is, bigger the communities are.
        """
        self.graph = nx.from_numpy_matrix(adj_mat)
        self.dendrogram = community_louvain.generate_dendrogram(self.graph)
        self.updateLabels(self.level)
        self.centroids = computeCentroids(inputs, self.labels)
        return


def computeCentroids(data, labels):
    n_clusters = len(np.unique(labels))
    return np.array([np.mean(data[labels == i], axis=0) for i in range(n_clusters)])


def _nan2zero(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)


def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def normalize_sparse_matrix(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


class Colors():
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#279e68",
        "#d62728",
        "#633194",
        "#8c564b",
        "#F73BAD",
        "#ad494a",
        "#F6E800",
        "#01F7F7",
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
        "#ff9896",
        "#c5b0d5",
        "#c49c94",
        "#f7b6d2",
        "#dbdb8d",
        "#9edae5",
        "#8c6d31"]

def res_search_fixed_clus(cluster_type, adata, fixed_clus_count, increment=0.01):
    '''
                arg1(adata)[AnnData matrix]
                arg2(fixed_clus_count)[int]

                return:
                    resolution[int]
            '''
    if cluster_type == 'leiden':
        for res in sorted(list(np.arange(0.14, 2.5, increment))):#, reverse=True):
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique_leiden = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            # print(res,' ' , count_unique_leiden)
            if count_unique_leiden == fixed_clus_count:
                cluster_labels = np.array(adata.obs['leiden'])
                flag=0
                break
            if count_unique_leiden > fixed_clus_count:
                cluster_labels = np.array(adata.obs['leiden'])
                flag=1
                break
    elif cluster_type == 'louvain':
        for res in sorted(list(np.arange(0.14, 2.5, increment))):#, reverse=True):
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique_louvain = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
            # print(res,' ' , count_unique_louvain)
            if count_unique_louvain == fixed_clus_count:
                cluster_labels = np.array(adata.obs['louvain'])
                flag = 0
                break
            if count_unique_louvain > fixed_clus_count:
                cluster_labels = np.array(adata.obs['louvain'])
                flag = 1
                break
    return cluster_labels,flag


def PCA_process(X, nps):
    from sklearn.decomposition import PCA
    print('Shape of data to PCA:', X.shape)
    pca = PCA(n_components=nps)
    X_PC = pca.fit_transform(X)     #等价于pca.fit(X) pca.transform(X)
    print('Shape of data output by PCA:', X_PC.shape)
    print('PCA recover:', pca.explained_variance_ratio_.sum())
    return X_PC