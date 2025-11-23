import numpy as np
import umap.umap_ as umap  # Importing UMAP
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

doc_embeddings_path_en = '/users/seung-won/documents/DSdatasets/Bills/legal_doc_embeddings.npy'
doc_embeddings_en = np.load(doc_embeddings_path_en)

# 텍스트 파일 경로
corpus_path_en = '/users/seung-won/documents/DSdatasets/Bills/Bills_train_2000.txt'

# 코퍼스 데이터를 읽어서 corpus 변수에 저장
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


output_corpus_file_en = '/users/seung-won/documents/clusters_en_50.txt'
BERTopic(corpus_en, doc_embeddings_en, output_corpus_file_en, num_clusters=50, n_components=5)


def topic_discovery(corpus_path, output_path):
    with open(corpus_path, 'r', encoding='utf-8') as file:
        corpus = [line.strip() for line in file]

    # TF-IDF 계산
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()

    # 상위 15개 단어 추출
    top_n = 15
    top_tfidf_words = []

    for doc_idx in range(tfidf_matrix.shape[0]):
        # TF-IDF 값과 단어 인덱스 가져오기
        tfidf_scores = tfidf_matrix[doc_idx].toarray().flatten()
        top_indices = np.argsort(tfidf_scores)[::-1][:top_n]  # TF-IDF 값이 높은 순서로 정렬 후 상위 n개 선택
        top_words = [feature_names[i] for i in top_indices]
        top_tfidf_words.append(top_words)

    # 로컬 파일에 저장
    with open(output_path, 'w', encoding='utf-8') as file:
        for doc_words in top_tfidf_words:
            file.write(" ".join(doc_words) + "\n")

    print(f"Top 15 TF-IDF words per document saved to: {output_path}")


# 코퍼스 파일 경로
corpus_path_en = '/users/seung-won/documents/clusters_en_50.txt'
output_path_en = '/users/seung-won/documents/topic_en.txt'

topic_discovery(corpus_path_en, output_path_en)