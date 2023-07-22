import numpy as np
import pandas as pd
from sklearn import preprocessing
import torch
from sklearn.metrics import confusion_matrix
import operator
import sklearn.metrics
import json
from scDGD.classes import GaussianMixture
import seaborn as sns
import matplotlib.pyplot as plt
import umap


def order_matrix_by_max_per_class(mtrx, class_labels, comp_order=None):
    if comp_order is not None:
        temp_mtrx = np.zeros(mtrx.shape)
        for i in range(mtrx.shape[1]):
            temp_mtrx[:, i] = mtrx[:, comp_order[i]]
        mtrx = temp_mtrx
    max_id_per_class = np.argmax(mtrx, axis=1)
    max_coordinates = list(zip(np.arange(mtrx.shape[0]), max_id_per_class))
    max_coordinates.sort(key=operator.itemgetter(1))
    new_class_order = [x[0] for x in max_coordinates]
    new_mtrx = np.zeros(mtrx.shape)
    # reindexing mtrx worked on test but not in application, reverting to stupid safe for-loop
    for i in range(mtrx.shape[0]):
        new_mtrx[i, :] = mtrx[new_class_order[i], :]
    # mtrx = mtrx[new_class_order,:]
    return new_mtrx, [class_labels[i] for i in new_class_order]


def gmm_clustering(r, gmm, labels):
    # transform categorical labels into numerical
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    true_labels = le.transform(labels)
    # compute probabilities per sample and component (n_sample,n_mix_comp)
    probs_per_sample_and_component = gmm.sample_probs(torch.tensor(r))
    # get index (i.e. component id) of max prob per sample
    cluster_labels = (
        torch.max(probs_per_sample_and_component, dim=-1).indices.cpu().detach()
    )
    return cluster_labels


def compute_distances(mtrx):
    distances = sklearn.metrics.pairwise.euclidean_distances(mtrx)
    return distances


def get_connectivity_from_threshold(mtrx, threshold):
    connectivity_mtrx = np.zeros(mtrx.shape)
    idx = np.where(mtrx <= threshold)
    connectivity_mtrx[idx[0], idx[1]] = 1
    np.fill_diagonal(connectivity_mtrx, 0)
    return connectivity_mtrx


def rank_distances(mtrx):
    ranks = np.argsort(mtrx, axis=-1)
    # testing advanced indexing for ranking
    m, n = mtrx.shape
    # Initialize output array
    out = np.empty((m, n), dtype=int)
    # Use sidx as column indices, while a range array for the row indices
    # to select one element per row. Since sidx is a 2D array of indices
    # we need to use a 2D extended range array for the row indices
    out[np.arange(m)[:, None], ranks] = np.arange(n)
    return out
    # return ranks


def get_node_degrees(mtrx):
    return np.sum(mtrx, 1)


def get_secondary_degrees(mtrx):
    out = np.zeros(mtrx.shape[0])
    for i in range(mtrx.shape[0]):
        direct_neighbors = np.where(mtrx[i] == 1)[0]
        out[i] = mtrx[direct_neighbors, :].sum()
    return out


def find_start_node(d1, d2):
    minimum_first_degree = np.where(d1 == d1.min())[0]
    if len(minimum_first_degree) > 1:
        minimum_second_degree_subset = np.where(
            d2[minimum_first_degree] == d2[minimum_first_degree].min()
        )[0][0]
        return minimum_first_degree[minimum_second_degree_subset]
    else:
        return minimum_first_degree[0]


def find_next_node(c, r, i):
    connected_nodes = np.where(c[i, :] == 1)[0]
    if len(connected_nodes) > 0:
        connected_nodes_paired = list(zip(connected_nodes, r[i, connected_nodes]))
        connected_nodes_paired.sort(key=operator.itemgetter(1))
        connected_nodes = [
            connected_nodes_paired[x][0] for x in range(len(connected_nodes))
        ]
    return connected_nodes


def traverse_through_graph(connectiv_mtrx, ranks, first_degrees, second_degrees):
    # create 2 lists of node ids
    # the first one keeps track of the nodes we have used already (and stores them in the desired order)
    # the second one keeps track of the nodes we still have to sort
    node_order = []
    nodes_to_be_distributed = list(np.arange(connectiv_mtrx.shape[0]))

    start_node = find_start_node(first_degrees, second_degrees)
    node_order.append(start_node)
    nodes_to_be_distributed.remove(start_node)

    count_turns = 0
    while len(nodes_to_be_distributed) > 0:
        next_nodes = find_next_node(connectiv_mtrx, ranks, node_order[-1])
        next_nodes = list(set(next_nodes).difference(set(node_order)))
        if len(next_nodes) < 1:
            next_nodes = [
                nodes_to_be_distributed[
                    find_start_node(
                        first_degrees[nodes_to_be_distributed],
                        second_degrees[nodes_to_be_distributed],
                    )
                ]
            ]
        for n in next_nodes:
            if n not in node_order:
                node_order.append(n)
                nodes_to_be_distributed.remove(n)
                count_turns = 0
        count_turns += 1
        if count_turns >= 10:
            break

    return node_order


def order_components_as_graph_traversal(gmm):
    distance_mtrx = compute_distances(gmm.mean.detach().cpu().numpy())
    threshold = round(np.percentile(distance_mtrx.flatten(), 30), 2)

    connectivity_mtrx = get_connectivity_from_threshold(distance_mtrx, threshold)
    rank_mtrx = rank_distances(distance_mtrx)

    node_degrees = get_node_degrees(connectivity_mtrx)
    secondary_node_degrees = get_secondary_degrees(connectivity_mtrx)

    new_node_order = traverse_through_graph(
        connectivity_mtrx, rank_mtrx, node_degrees, secondary_node_degrees
    )
    return new_node_order


def clustering_matrix(gmm, rep, labels, norm=True):
    classes = list(np.unique(labels))
    true_labels = np.asarray([classes.index(i) for i in labels])
    cluster_labels = gmm_clustering(rep, gmm, labels)
    # get absolute confusion matrix
    cm1 = confusion_matrix(true_labels, cluster_labels)

    class_counts = [np.where(true_labels == i)[0].shape[0] for i in range(len(classes))]
    cm2 = cm1.astype(np.float64)
    for i in range(len(class_counts)):
        # percent_sum = 0
        for j in range(gmm.Nmix):
            if norm:
                cm2[i, j] = cm2[i, j] * 100 / class_counts[i]
            else:
                cm2[i, j] = cm2[i, j]
    cm2 = cm2.round()

    # get an order of components based on connectivity graph
    component_order = order_components_as_graph_traversal(gmm)

    # take the non-empty entries
    cm2 = cm2[: len(classes), : gmm.Nmix]

    cm3, classes_reordered = order_matrix_by_max_per_class(
        cm2, classes, component_order
    )
    out = pd.DataFrame(data=cm3, index=classes_reordered, columns=component_order)
    return out


def load_embedding(save_dir="./"):
    # get parameter dict
    with open(save_dir + "dgd_hyperparameters.json", "r") as fp:
        param_dict = json.load(fp)

    embedding = torch.load(
        save_dir + param_dict["name"] + "_representation.pt",
        map_location=torch.device("cpu"),
    )
    return embedding["z"].detach().cpu().numpy()


def load_labels(save_dir="./", split="train"):
    # load train-val-test split
    obs = pd.read_csv(save_dir + "data_obs.csv", index_col=0)
    return obs[obs["train_val_test"] == split]["cell_type"].values


def plot_cluster_heatmap(save_dir="./"):
    gmm = GaussianMixture.load(save_dir=save_dir)
    embedding = load_embedding(save_dir=save_dir)
    labels = load_labels()

    df_relative_clustering = clustering_matrix(gmm, embedding, labels)
    df_clustering = clustering_matrix(gmm, embedding, labels, norm=False)

    annotations = df_relative_clustering.to_numpy(dtype=np.float64).copy()
    annotations[annotations < 1] = None
    df_relative_clustering = df_relative_clustering.fillna(0)
    fig, ax = plt.subplots(figsize=(0.5*annotations.shape[0], 0.5*annotations.shape[1]))
    cmap = sns.color_palette("GnBu", as_cmap=True)
    sns.heatmap(
        df_relative_clustering,
        annot=annotations,
        cmap=cmap,
        annot_kws={"size": 6},
        cbar_kws={"shrink": 0.5, "location": "bottom"},
        xticklabels=True,
        yticklabels=True,
        mask=np.isnan(annotations),
        alpha=0.8,
    )
    ylabels = [
        df_clustering.index[x] + " (" + str(int(df_clustering.sum(axis=1)[x])) + ")"
        for x in range(df_clustering.shape[0])
    ]
    plt.yticks(
        ticks=np.arange(len(ylabels)) + 0.5, labels=ylabels, rotation=0, fontsize=8
    )
    plt.tick_params(axis="x", rotation=0, labelsize=8)
    plt.ylabel("Cell type")
    plt.xlabel("GMM component ID")
    plt.title("percentage of cell type in GMM cluster")
    plt.show()


def plot_latent_umap(save_dir="./", n_neighbors=15, min_dist=0.5):
    gmm = GaussianMixture.load(save_dir=save_dir)
    embedding = load_embedding(save_dir=save_dir)
    labels = load_labels()

    # make umap
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=2, min_dist=min_dist)
    projected = reducer.fit_transform(embedding)
    plot_data = pd.DataFrame(projected, columns=["UMAP1", "UMAP2"])
    plot_data["cell type"] = labels
    plot_data["cell type"] = plot_data["cell type"].astype("category")
    plot_data["cluster"] = (
        gmm.clustering(embedding).cpu().detach().numpy()
    )  # .astype(str)
    plot_data["cluster"] = plot_data["cluster"].astype("category")

    # make a plot with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    # adjust spacing between plots
    fig.subplots_adjust(wspace=1.0)
    # make text size smaller
    plt.rcParams.update({"font.size": 6})
    sns.scatterplot(data=plot_data, x="UMAP1", y="UMAP2", hue="cell type", ax=ax1, s=1)
    sns.scatterplot(data=plot_data, x="UMAP1", y="UMAP2", hue="cluster", ax=ax2, s=1)
    ax1.set_title("cell type")
    ax1.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.0, frameon=False)
    ax2.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.0, frameon=False)
    ax2.set_title("cluster")
    plt.show()
