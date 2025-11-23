import numpy as np
from collections import defaultdict
from sklearn import metrics
from sklearn.cluster import KMeans


def labels(labels_path):
    labels_list = []
    with open(labels_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 빈 줄이 아닐 경우, 정수로 변환해서 리스트에 추가
            if line:
                labels_list.append(int(line))
    return labels_list


theta_en = np.load("/users/seung-won/documents/AR_doc_topic_dist_en_20.npy")
theta_cn = np.load("/users/seung-won/documents/AR_doc_topic_dist_cn_20.npy")

# theta_en = np.load("/users/seung-won/documents/Overall_Results/NMTM/AR_doc_topic_dist_en_20.npy")
# theta_cn = np.load("/users/seung-won/documents/Overall_Results/NMTM/AR_doc_topic_dist_cn_20.npy")


labels_en_path = '/users/seung-won/documents/datasets/Amazon_Review/train_labels_en.txt'
labels_cn_path = '/users/seung-won/documents/datasets/Amazon_Review/train_labels_cn.txt'

labels_en = labels(labels_en_path)
labels_cn = labels(labels_cn_path)


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def clustering_metrics(labels, preds):
    metrics_func = [
        {
            'name': 'Purity',
            'method': purity_score
        },
        {
            'name': 'NMI',
            'method': metrics.cluster.normalized_mutual_info_score
        },
    ]

    results = dict()
    for func in metrics_func:
        results[func['name']] = func['method'](labels, preds)

    return results

def clustering_score(theta, labels, num_clusters):
    cluster_labels = np.argmax(theta, axis=1)
    # kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    # cluster_labels = kmeans.fit_predict(theta)
    return clustering_metrics(labels, cluster_labels)


theta_all = np.vstack((theta_en, theta_cn))
labels_all = labels_en + labels_cn

results_all = clustering_score(theta_all, labels_all, num_clusters=20)

print(results_all)