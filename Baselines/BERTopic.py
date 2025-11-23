import numpy as np
import umap.umap_ as umap  # Importing UMAP
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

doc_embeddings_path_en = '/workspace/doc_embeddings.npy'
doc_embeddings_en = np.load(doc_embeddings_path_en)

corpus_path_en = '/workspace/train.txt'

with open(corpus_path_en, 'r', encoding='utf-8') as file:
    corpus_en = [line.strip() for line in file]


def BERTopic(corpus, doc_embeddings, output_corpus_file, num_clusters, n_components=5):

    # Step 1: UMAP dimensionality reduction
    umap_reducer = umap.UMAP(n_components=n_components, random_state=42)
    reduced_embeddings = umap_reducer.fit_transform(doc_embeddings)

    # Step 2: KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(reduced_embeddings)

    # Step 3: Concatenate documents within the same cluster
    cluster_to_docs = {i: [] for i in range(num_clusters)}  # Dictionary to store documents by cluster
    for doc, label in zip(corpus, cluster_labels):
        cluster_to_docs[label].append(doc)

    concatenated_corpus = [" ".join(cluster_to_docs[i]) for i in range(num_clusters)]  # Concatenate documents per cluster

    # Step 4: Save the concatenated corpus to a text file
    with open(output_corpus_file, 'w', encoding='utf-8') as f:
        for concatenated_doc in concatenated_corpus:
            f.write(f"{concatenated_doc}\n")

    return cluster_labels

output_corpus_file_en = '/workspace/clusters_en_20.txt'
BERTopic(corpus_en, doc_embeddings_en, output_corpus_file_en, num_clusters=20, n_components=5)


def topic_discovery(corpus_path, output_path):
    with open(corpus_path, 'r', encoding='utf-8') as file:
        corpus = [line.strip() for line in file]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()

    top_n = 15
    top_tfidf_words = []

    for doc_idx in range(tfidf_matrix.shape[0]):
        tfidf_scores = tfidf_matrix[doc_idx].toarray().flatten()
        top_indices = np.argsort(tfidf_scores)[::-1][:top_n] 
        top_words = [feature_names[i] for i in top_indices]
        top_tfidf_words.append(top_words)

    with open(output_path, 'w', encoding='utf-8') as file:
        for doc_words in top_tfidf_words:
            file.write(" ".join(doc_words) + "\n")

    print(f"Top 15 TF-IDF words per document saved to: {output_path}")


corpus_path_en = '/workspace/clusters_en_20.txt'
output_path_en = '/workspace/topic_en.txt'

topic_discovery(corpus_path_en, output_path_en)
