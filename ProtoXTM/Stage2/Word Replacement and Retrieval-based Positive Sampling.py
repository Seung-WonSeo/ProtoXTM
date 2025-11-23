import fasttext
import fasttext.util
import numpy as np
import re
from rank_bm25 import BM25Okapi
from collections import defaultdict, Counter
from gensim.models import KeyedVectors

def read_text(path):
    texts = list()
    with open(path, 'r', encoding='utf-8', errors='ignore') as file:
        for line in file:
            texts.append(line.strip())
    return texts

def clean_topic_words(topic_en_trans):
    cleaned_topics = []
    for topic in topic_en_trans:
        cleaned_topic = re.sub(r'[^a-zA-Z\s]', '', topic)
        cleaned_topic = re.sub(r'\s+', ' ', cleaned_topic).strip()
        cleaned_topics.append(cleaned_topic)
    return cleaned_topics


def clean_chinese_topic_words(topic_cn_trans):
    cleaned_topics = []
    for topic in topic_cn_trans:
        cleaned_topic = re.sub(r'[^\u4e00-\u9fff\s]', '', topic)
        cleaned_topic = re.sub(r'\s+', ' ', cleaned_topic).strip()
        cleaned_topics.append(cleaned_topic)
    return cleaned_topics


# Reference: https://fasttext.cc/docs/en/crawl-vectors.html
ft_en = fasttext.load_model('/workspace/fasttext/cc.en.300.bin')
ft_cn = fasttext.load_model('/workspace/fasttext/cc.zh.300.bin')


# vocab
vocab_en_path = "/workspace/Amazon_Review/AR_vocab_en"
vocab_cn_path = "/workspace/Amazon_Review/AR_vocab_cn"

# raw texts
train_texts_en_path = "/workspace/Amazon_Review/train_texts_en.txt"
train_texts_cn_path = "/workspace/Amazon_Review/train_texts_cn.txt"


# translated topic text
topic_en_trans_path = "/workspace/Amazon_Review/topic_en_trans.txt"
topic_cn_trans_path = "/workspace/Amazon_Review/topic_cn_trans.txt"

vocab_en = read_text(vocab_en_path)
vocab_cn = read_text(vocab_cn_path)

topic_en_trans = read_text(topic_en_trans_path)
topic_cn_trans = read_text(topic_cn_trans_path)

train_texts_en = read_text(train_texts_en_path)
train_texts_cn = read_text(train_texts_cn_path)


topic_en_trans = clean_topic_words(topic_en_trans)
topic_cn_trans = clean_chinese_topic_words(topic_cn_trans)

# Word Replacement for English and Chinese
def paraphrasing(topic_en_trans, vocab_en, ft_en, threshold):
    processed_topics = []
    
    for topic in topic_en_trans:
        words = topic.split() 
        processed_words = []
        
        for word in words:
            if word in vocab_en:
                processed_words.append(word)
            elif word in ft_en:
                word_vector = ft_en.get_word_vector(word)
                max_similarity = 0
                best_match = None
                
                for vocab_word in vocab_en:
                    vocab_vector = ft_en.get_word_vector(vocab_word)
                    similarity = np.dot(word_vector, vocab_vector) / (np.linalg.norm(word_vector) * np.linalg.norm(vocab_vector))
                    if similarity > threshold and similarity > max_similarity:
                        max_similarity = similarity
                        best_match = vocab_word
                
                if best_match:
                    processed_words.append(best_match)
            else:
                continue
        
        processed_topics.append(" ".join(processed_words))
    return processed_topics
    

def paraphrasing_cn(topic_cn_trans, vocab_cn, ft_cn, threshold):

    processed_topics = []
    
    for topic in topic_cn_trans:
        words = topic.split() 
        processed_words = []
        
        for word in words:
            if word in vocab_cn:
                processed_words.append(word)
            elif word in ft_cn:
                word_vector = ft_cn.get_word_vector(word)
                max_similarity = 0
                best_match = None
                
                for vocab_word in vocab_cn:
                    vocab_vector = ft_cn.get_word_vector(vocab_word)
                    similarity = np.dot(word_vector, vocab_vector) / (np.linalg.norm(word_vector) * np.linalg.norm(vocab_vector))
                    if similarity > threshold and similarity > max_similarity:
                        max_similarity = similarity
                        best_match = vocab_word
                
                if best_match:
                    processed_words.append(best_match)
            else:
                continue
        processed_topics.append(" ".join(processed_words))
    return processed_topics


paraphrasing_topics_en = paraphrasing(topic_en_trans, vocab_en, ft_en, threshold=0.4)
paraphrasing_topics_cn = paraphrasing_cn(topic_cn_trans, vocab_cn, ft_cn, threshold=0.4)


def get_top_documents_for_topics(topics, train_texts_en, top_n):
    tokenized_documents = [doc.split() for doc in train_texts_en]
    bm25 = BM25Okapi(tokenized_documents)
    topic_results = {}
    for topic_idx, topic_words in enumerate(topics):
     
        doc_scores = defaultdict(float)

        for word in topic_words.split():
            
            word_scores = bm25.get_scores([word])
            for doc_idx, score in enumerate(word_scores):
               
                doc_scores[doc_idx] += score

        top_docs = sorted(
            doc_scores.items(), key=lambda x: x[1], reverse=True
        )[:top_n]

        topic_results[f"Topic_{topic_idx + 1}"] = [
            {
                "doc_index": idx,
                "bm25_total_score": total_score,
                "doc_content": train_texts_en[idx]
            }
            for idx, total_score in top_docs
        ]

    return topic_results

# n: number of top-k documents for each topic
top_documents_en_200 = get_top_documents_for_topics(paraphrasing_topics_en, train_texts_en, top_n=30)
top_documents_cn_200 = get_top_documents_for_topics(paraphrasing_topics_cn, train_texts_cn, top_n=30)


# Pseudo Labeling
from collections import defaultdict
def pseudo_label_documents(top_documents, total_docs):

    doc_label_data = defaultdict(lambda: {"topic_index": -1, "rank": float("inf"), "score": -1})
    
    for topic_idx, docs in enumerate(top_documents.values()):
        for rank, doc in enumerate(docs):
            doc_index = doc["doc_index"]
            bm25_score = doc["bm25_total_score"]

          
            if rank < doc_label_data[doc_index]["rank"]:
                doc_label_data[doc_index] = {
                    "topic_index": topic_idx,
                    "rank": rank,
                    "score": bm25_score,
                }

    labels = [-1] * total_docs
    for doc_index, data in doc_label_data.items():
        labels[doc_index] = data["topic_index"]

    return labels

total_docs_en = 20000
labels_c2e_200 = pseudo_label_documents(top_documents_en_200, total_docs_en)

total_docs_cn = 20000  
labels_e2c_200 = pseudo_label_documents(top_documents_cn_200, total_docs_cn)

def save_labels_to_npy(labels, file_path):

    labels_array = np.array(labels)  
    np.save(file_path, labels_array)
    print(f"Labels saved to {file_path}")
