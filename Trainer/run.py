if __name__ == "__main__":
    # Multi-lingual BERT embeddings
    doc_embeddings_en_path = "/workspace/Amazon_Review/AR_sbert_doc_embeddings_en.npz"
    doc_embeddings_cn_path = "/workspace/Amazon_Review/AR_sbert_doc_embeddings_cn.npz"
    
    # doc_embeddings_en_path = "/workspace/Amazon_Review/AR_sbert_doc_embeddings_en.npz"
    # doc_embeddings_cn_path = "/workspace/Amazon_Review/AR_sbert_doc_embeddings_cn.npz"
    
    # bow_embeddings_en_path = "/workspace/ECNews/ECN_bow_embeddings_en.npy"
    # bow_embeddings_cn_path = "/workspace/ECNews/ECN_bow_embeddings_cn.npy"
    
    bow_embeddings_en_path = "/workspace/Amazon_Review/AR_bow_embeddings_en.npy"
    bow_embeddings_cn_path = "/workspace/Amazon_Review/AR_bow_embeddings_cn.npy"

    labels_en_path = "/workspace/Amazon_Review/XLM_labels_en_50.npy"
    labels_cn_path = "/workspace/Amazon_Review/XLM_labels_cn_50.npy"
    
    # labels_en_path = "/workspace/Amazon_Review/k=50/labels_en.npy"
    # labels_cn_path = "/workspace/Amazon_Review/k=50/labels_cn.npy"

    
    # labels_c2e_path = "/workspace/ECNews/labels_c2e_70.npy"
    # labels_e2c_path = "/workspace/ECNews/labels_e2c_70.npy"
    
    labels_c2e_path = "/workspace/Amazon_Review/XLM_labels_c2e_30.npy"
    labels_e2c_path = "/workspace/Amazon_Review/XLM_labels_e2c_30.npy"

    sbert_doc_embeddings_en = np.load(doc_embeddings_en_path)  # English document embeddings
    sbert_doc_embeddings_cn = np.load(doc_embeddings_cn_path)  # Chinese document embeddings
    
    sbert_doc_embeddings_en = sbert_doc_embeddings_en['embeddings']
    sbert_doc_embeddings_cn = sbert_doc_embeddings_cn['embeddings']
    
    bow_en = np.load(bow_embeddings_en_path)  # English BoW embeddings
    bow_cn = np.load(bow_embeddings_cn_path)  # Chinese BoW embeddings
    
    labels_en = np.load(labels_en_path)
    labels_cn = np.load(labels_cn_path)
    
    labels_c2e = np.load(labels_c2e_path)
    labels_e2c = np.load(labels_e2c_path)
    

    input_size = sbert_doc_embeddings_en.shape[1]
    vocab_size_en = bow_en.shape[1]
    vocab_size_cn = bow_cn.shape[1]
    num_topics = 20
    DCL_weight = 1
    temperature = 0.1
    
    model = ProtoXTM(input_size=input_size, vocab_size_en=vocab_size_en, vocab_size_cn=vocab_size_cn,
                    num_topics=num_topics, DPCL_weight=DPCL_weight,
                    temperature=temperature, en_units=200, dropout=0.1)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # DataLoader
    batch_size_en = 512
    batch_size_cn = 512
    
    dataloader_en, dataloader_cn = create_dataloader_separate(sbert_doc_embeddings_en, bow_en, labels_en,
                                                              labels_c2e, sbert_doc_embeddings_cn, bow_cn,
                                                              labels_cn, labels_e2c,
                                                              batch_size_en, batch_size_cn)

    # train
    trained_model = train_ProtoXTM(model, dataloader_en, dataloader_cn, optimizer, num_epochs=500, device='cuda')
