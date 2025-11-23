import os
import math
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict

global_word_count = defaultdict(int)  # Global dictionary for word counts
WTOTALKEY = "!!<TOTAL_WINDOWS>!!"

def load_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        corpus = file.readlines()
    corpus = [line.strip().lower().split() for line in corpus]
    return corpus

def make_bow(corpus_path, vocab):
    with open(corpus_path, 'r', encoding='utf-8') as file:
        corpus = file.readlines()
    vectorizer = CountVectorizer(vocabulary=vocab, tokenizer=lambda x: x.split())
    bow = vectorizer.fit_transform(corpus).toarray()
    wc_dict = dict(zip(vocab, bow.sum(axis=0)))  # Sum across documents to get total word frequencies
    return bow, wc_dict

def calcwcngram(corpus_tuple, vocab1, vocab2, word_pair_list, sep_token):
    corpus1_path, corpus2_path = corpus_tuple
    all_wc_dict = defaultdict(int)
    bow1, wc_dict1 = make_bow(corpus1_path, vocab1)
    bow2, wc_dict2 = make_bow(corpus2_path, vocab2)

    all_wc_dict.update(wc_dict1)
    all_wc_dict.update(wc_dict2)

    for word_pair in word_pair_list:
        if word_pair not in all_wc_dict:
            w1, w2 = word_pair.split(sep_token)
            pair_count = ((bow1[:, vocab1.index(w1)] * bow2[:, vocab2.index(w2)]) > 0).sum()
            all_wc_dict[word_pair] = pair_count

    all_wc_dict[WTOTALKEY] = bow1.shape[0]
    return all_wc_dict

def update_global_word_count(worker_wordcount):
    for k, v in worker_wordcount.items():
        global_word_count[k] += v

def calc_assoc(word_pair, window_total, sep_token, metric='npmi'):
    word1, word2 = word_pair.split(sep_token)
    combined_count = float(global_word_count[word_pair])
    w1_count = float(global_word_count[word1])
    w2_count = float(global_word_count[word2])

    if w1_count == 0 or w2_count == 0 or combined_count == 0:
        return 0.0

    pmi = math.log((combined_count * window_total) / (w1_count * w2_count), 10)
    if metric == "npmi":
        npmi = pmi / (-1.0 * math.log(combined_count / window_total, 10))
        return npmi
    return pmi

def load_topics(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        topics = [line.strip().split() for line in file]
    return topics

def load_vocab(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        vocab = [line.strip() for line in file]
    return vocab

def process_corpus(parallel_corpus_tuples, vocab1, vocab2, word_pair_list, sep_token):
    # Sequentially process each corpus pair
    for cp in parallel_corpus_tuples:
        if not os.path.exists(cp[0]) or not os.path.exists(cp[1]):
            raise FileNotFoundError(f"One of the corpus files not found: {cp}")

        print("===> Processing corpus pair: ", cp)
        worker_wordcount = calcwcngram(cp, vocab1, vocab2, word_pair_list, sep_token)
        update_global_word_count(worker_wordcount)

def main():
    # Define paths to the corpus and vocab files
    english_corpus_files = [
        '/users/seung-won/documents/Evaluation/CNPMI/ref_corpus/en-cn/wikicomp-2014_enzh.xml_1-20000_en.txt',
        '/users/seung-won/documents/Evaluation/CNPMI/ref_corpus/en-cn/wikicomp-2014_enzh.xml_20001-40000_en.txt',
        '/users/seung-won/documents/Evaluation/CNPMI/ref_corpus/en-cn/wikicomp-2014_enzh.xml_40001-60000_en.txt',
        '/users/seung-won/documents/Evaluation/CNPMI/ref_corpus/en-cn/wikicomp-2014_enzh.xml_60001-80000_en.txt',
        '/users/seung-won/documents/Evaluation/CNPMI/ref_corpus/en-cn/wikicomp-2014_enzh.xml_80001-100000_en.txt'
        # Add more files as needed
    ]
    chinese_corpus_files = [
        '/users/seung-won/documents/Evaluation/CNPMI/ref_corpus/en-cn/wikicomp-2014_enzh.xml_1-20000_zh.txt',
        '/users/seung-won/documents/Evaluation/CNPMI/ref_corpus/en-cn/wikicomp-2014_enzh.xml_20001-40000_zh.txt',
        '/users/seung-won/documents/Evaluation/CNPMI/ref_corpus/en-cn/wikicomp-2014_enzh.xml_40001-60000_zh.txt',
        '/users/seung-won/documents/Evaluation/CNPMI/ref_corpus/en-cn/wikicomp-2014_enzh.xml_60001-80000_zh.txt',
        '/users/seung-won/documents/Evaluation/CNPMI/ref_corpus/en-cn/wikicomp-2014_enzh.xml_80001-100000_zh.txt'
        # Add more files as needed
    ]
    parallel_corpus_tuples = list(zip(english_corpus_files, chinese_corpus_files))

    # Load vocabularies
    vocab1 = load_vocab('/users/seung-won/documents/datasets/Amazon_Review/AR_vocab_en')
    vocab2 = load_vocab('/users/seung-won/documents/datasets/Amazon_Review/AR_vocab_cn')
    
    sep_token = '|'

    # Load English and Chinese topics
    # topics1 = load_topics("/users/seung-won/documents/ECN_topic_en_20.txt")
    # topics2 = load_topics("/users/seung-won/documents/ECN_topic_cn_20.txt")
    # topics1 = load_topics("/users/seung-won/documents/Overall_Results/NMTM/AR_topic_en_10.txt")
    # topics2 = load_topics("/users/seung-won/documents/Overall_Results/NMTM/AR_topic_cn_10.txt")
    topics1 = load_topics("/users/seung-won/documents/topic_en.txt")
    topics2 = load_topics("/users/seung-won/documents/topic_cn.txt")

    num_topic = len(topics1)
    num_top_word = len(topics1[0])

    print("===> Preparing word pairs.")
    word_pair_list = [f'{w1}{sep_token}{w2}' for k in range(num_topic) for w1 in topics1[k] for w2 in topics2[k]]

    # Process corpora sequentially
    process_corpus(parallel_corpus_tuples, vocab1, vocab2, word_pair_list, sep_token)

    print("===> Computing coherence metric.")
    topic_assoc = []
    window_total = float(global_word_count[WTOTALKEY])
    for word_pair in word_pair_list:
        topic_assoc.append(calc_assoc(word_pair, window_total, sep_token, metric='npmi'))

    final_score = sum(topic_assoc) / len(topic_assoc) if topic_assoc else 0
    print(f'Final CNPMI Score: {final_score:.5f}')

if __name__ == '__main__':
    main()