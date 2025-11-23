from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim.utils import simple_preprocess


# Step 1: Load corpus (reference corpus)
def read_corpus(file_list):
    """
    Reads and preprocesses text files into a list of documents.
    """
    documents = []
    for file_path in file_list:
        with open(file_path, 'r', encoding='utf-8') as f:
            documents.extend(f.readlines())
    return [simple_preprocess(doc.strip()) for doc in documents]

# Step 2: Load topics
def read_topics(topic_file):
    """
    Reads topics from topic.txt where each line contains space-separated words representing a topic.
    """
    topics = []
    with open(topic_file, 'r', encoding='utf-8') as f:
        for line in f:
            topics.append(line.strip().split())
    return topics

# For English Topics

# Main function to compute NPMI

def main():
    # Define file paths
    english_corpus_files = [
        '/users/seung-won/documents/Evaluation/CNPMI/ref_corpus/en-cn/wikicomp-2014_enzh.xml_1-20000_en.txt',
        '/users/seung-won/documents/Evaluation/CNPMI/ref_corpus/en-cn/wikicomp-2014_enzh.xml_20001-40000_en.txt',
        '/users/seung-won/documents/Evaluation/CNPMI/ref_corpus/en-cn/wikicomp-2014_enzh.xml_40001-60000_en.txt',
        '/users/seung-won/documents/Evaluation/CNPMI/ref_corpus/en-cn/wikicomp-2014_enzh.xml_60001-80000_en.txt',
        '/users/seung-won/documents/Evaluation/CNPMI/ref_corpus/en-cn/wikicomp-2014_enzh.xml_80001-100000_en.txt'
    ]
    # topic_file = "/users/seung-won/documents/RPSXTM/results/Amazon_Review/topic_en_10.txt"
    # topic_file = "/users/seung-won/documents/Overall_Results/InfoCTM/RA_topic_en_20.txt"
    topic_file = "/users/seung-won/documents/topic_en.txt"
    # topic_file = "/users/seung-won/documents/RPSXTM/results/Rakuten_Amazon/topic_en_10.txt"
    # Step 1: Load the reference corpus and preprocess
    print("Loading and preprocessing corpus...")
    documents = read_corpus(english_corpus_files)
    print(f"Loaded {len(documents)} documents.")

    # Step 2: Create a dictionary and corpus for Gensim
    dictionary = Dictionary(documents)
    corpus = [dictionary.doc2bow(doc) for doc in documents]

    # Step 3: Load the topics
    print("Loading topics...")
    topics = read_topics(topic_file)
    print(f"Loaded {len(topics)} topics.")

    # Step 4: Compute CoherenceModel using NPMI
    print("Calculating NPMI scores...")
    coherence_model_npmi = CoherenceModel(
        topics=topics, 
        texts=documents, 
        dictionary=dictionary, 
        coherence='c_npmi'  # Use NPMI coherence measure
    )
    npmi_score = coherence_model_npmi.get_coherence()
    
    print("Calculating Cv scores...")
    coherence_model_cv = CoherenceModel(
        topics=topics, 
        texts=documents, 
        dictionary=dictionary, 
        coherence='c_v'  # Use NPMI coherence measure
    )
    cv_score = coherence_model_cv.get_coherence()

    print(f"Average NPMI score: {npmi_score:.4f}")
    print(f"Average Cv score: {cv_score:.4f}")

if __name__ == "__main__":
    main()


# Main function to compute NPMI
def main():
    # Define file paths
    english_corpus_files = ['/users/seung-won/documents/DSdatasets/BioASQ/Biomedical_train.txt']
    # topic_file = "/users/seung-won/documents/RPSXTM/results/Amazon_Review/topic_en_10.txt"
    # topic_file = "/users/seung-won/documents/Overall_Results/InfoCTM/RA_topic_en_20.txt"
    topic_file = "/users/seung-won/documents/topic_en.txt"
    # topic_file = "/users/seung-won/documents/RPSXTM/results/Rakuten_Amazon/topic_en_10.txt"
    # Step 1: Load the reference corpus and preprocess
    print("Loading and preprocessing corpus...")
    documents = read_corpus(english_corpus_files)
    print(f"Loaded {len(documents)} documents.")

    # Step 2: Create a dictionary and corpus for Gensim
    dictionary = Dictionary(documents)
    corpus = [dictionary.doc2bow(doc) for doc in documents]

    # Step 3: Load the topics
    print("Loading topics...")
    topics = read_topics(topic_file)
    print(f"Loaded {len(topics)} topics.")

    # Step 4: Compute CoherenceModel using NPMI
    print("Calculating NPMI scores...")
    coherence_model_npmi = CoherenceModel(
        topics=topics, 
        texts=documents, 
        dictionary=dictionary, 
        coherence='c_npmi'  # Use NPMI coherence measure
    )
    npmi_score = coherence_model_npmi.get_coherence()
    
    print("Calculating Cv scores...")
    coherence_model_cv = CoherenceModel(
        topics=topics, 
        texts=documents, 
        dictionary=dictionary, 
        coherence='c_v'  # Use NPMI coherence measure
    )
    cv_score = coherence_model_cv.get_coherence()

    print(f"Average NPMI score: {npmi_score:.4f}")
    print(f"Average Cv score: {cv_score:.4f}")

if __name__ == "__main__":
    main()


# For Chinese Topics

# Main function to compute NPMI
def main():
    # Define file paths
    english_corpus_files = [
        '/users/seung-won/documents/Evaluation/CNPMI/ref_corpus/en-cn/wikicomp-2014_enzh.xml_1-20000_zh.txt',
        '/users/seung-won/documents/Evaluation/CNPMI/ref_corpus/en-cn/wikicomp-2014_enzh.xml_20001-40000_zh.txt',
        '/users/seung-won/documents/Evaluation/CNPMI/ref_corpus/en-cn/wikicomp-2014_enzh.xml_40001-60000_zh.txt',
        '/users/seung-won/documents/Evaluation/CNPMI/ref_corpus/en-cn/wikicomp-2014_enzh.xml_60001-80000_zh.txt',
        '/users/seung-won/documents/Evaluation/CNPMI/ref_corpus/en-cn/wikicomp-2014_enzh.xml_80001-100000_zh.txt'
    ]
    # topic_file = "/users/seung-won/documents/RPSXTM/results/Amazon_Review/topic_cn_10.txt"
    # topic_file = "/users/seung-won/documents/Overall_Results/InfoCTM/AR_topic_cn_20.txt"
    topic_file = "/users/seung-won/documents/topic_cn.txt"
    
    # Step 1: Load the reference corpus and preprocess
    print("Loading and preprocessing corpus...")
    documents = read_corpus(english_corpus_files)
    print(f"Loaded {len(documents)} documents.")

    # Step 2: Create a dictionary and corpus for Gensim
    dictionary = Dictionary(documents)
    corpus = [dictionary.doc2bow(doc) for doc in documents]

    # Step 3: Load the topics
    print("Loading topics...")
    topics = read_topics(topic_file)
    print(f"Loaded {len(topics)} topics.")

    # Step 4: Compute CoherenceModel using NPMI
    print("Calculating NPMI scores...")
    coherence_model_npmi = CoherenceModel(
        topics=topics, 
        texts=documents, 
        dictionary=dictionary, 
        coherence='c_npmi'  # Use NPMI coherence measure
    )
    npmi_score = coherence_model_npmi.get_coherence()
    
    print("Calculating Cv scores...")
    coherence_model_cv = CoherenceModel(
        topics=topics, 
        texts=documents, 
        dictionary=dictionary, 
        coherence='c_v'  # Use NPMI coherence measure
    )
    cv_score = coherence_model_cv.get_coherence()

    print(f"Average NPMI score: {npmi_score:.4f}")
    print(f"Average Cv score: {cv_score:.4f}")

if __name__ == "__main__":
    main()





    




















