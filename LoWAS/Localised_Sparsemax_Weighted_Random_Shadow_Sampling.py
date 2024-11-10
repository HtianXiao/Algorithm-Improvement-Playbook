from sklearn.metrics.pairwise import rbf_kernel
import pandas as pd
import numpy as np

def sparsemax(z):
    z = z - max(z)
    zsort = sorted(z, reverse=True)  # 升序
    out = len(z)
    sum = 0
    for k in range(0, len(z)):
        sum += zsort[k]
        value = (k + 1) * zsort[k]
        if (value <= sum - 1):
            out = k
            sum = sum - zsort[k] - 1
            break
    threshold = np.array((sum - 1) / out)
    return np.maximum(z - threshold, 0)

def Neb_grps(data, near_neb):
    'Function calculating nearest near_neb neighbours (among input data points), for every input data point'
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=near_neb, algorithm='ball_tree').fit(data)
    distances, indices = nbrs.kneighbors(data)
    neb_class = []
    for i in (indices):
        neb_class.append(i)
    return (np.asarray(neb_class))

def LoRAS(data, convex_neighbourhood, shadow, sigma, num_RACOS, num_convcom):
    np.random.seed(42)
    data_shadow = []

    # Generate shadow samples with Gaussian noise
    for i in range(convex_neighbourhood):
        for _ in range(shadow):
            noisy_sample = data[i] + np.random.normal(0, sigma, data[i].shape)
            data_shadow.append(noisy_sample)

    data_shadow = np.asarray(data_shadow)
    data_shadow_lc = []

    for _ in range(num_RACOS):
        idx = np.random.randint(shadow * convex_neighbourhood, size=num_convcom)
        selected_samples = data_shadow[idx, :]
        ## This is where the refinement lies
        ## Introduce this sparsemax simulated similarity weighting to eliminate lower quality shadow points
        similarity_matrix = rbf_kernel(selected_samples, gamma=1.0 / (2 * sigma ** 2))

        weights = similarity_matrix.mean(axis=0)

        weights = sparsemax(weights)
        new_sample = np.dot(weights, selected_samples)
        data_shadow_lc.append(new_sample)

    return np.asarray(data_shadow_lc)


def LoRAS_UMAP_gen(data, labels, convex_neighbourhood, umap_neighbourhood, shadow, sigma, num_RACOS, num_convcom):
    'Main LoRAS UMAP function performing UMAP, using Neb_grps function to investigate neighbours on the UMAP-'
    '-embedding and for each minority point using LoRAS function to generate synthetic samples'
    import numpy as np
    import umap.umap_ as umap

    features_1_trn = data[np.where(labels == 1)]
    features_0_trn = data[np.where(labels != 1)]
    n_feat = features_0_trn.shape[1]
    data_embedded_min = umap.UMAP(n_neighbors=umap_neighbourhood, min_dist=0.00000001, n_components=2,
                                  metric='euclidean', random_state=11).fit_transform(features_1_trn)
    nb_list = Neb_grps(data_embedded_min, convex_neighbourhood)

    RACOS_set = []
    for i in range(len(nb_list)):
        RACOS_i = LoRAS(features_1_trn[nb_list[i]], convex_neighbourhood, shadow, sigma, num_RACOS, num_convcom)
        RACOS_set.append(RACOS_i)
    LoRAS_set = np.asarray(RACOS_set)
    LoRAS_1 = np.reshape(LoRAS_set, (len(features_1_trn) * num_RACOS, n_feat))
    features_1_trn = np.concatenate((LoRAS_1, features_1_trn))
    return (np.concatenate((features_1_trn, features_0_trn)),
            np.concatenate((np.zeros(len(features_1_trn)) + 1, np.zeros(len(features_0_trn)))))
