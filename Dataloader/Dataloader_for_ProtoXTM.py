from torch.utils.data import TensorDataset, DataLoader
import torch

def create_dataloader_separate(sbert_doc_embeddings_en, bow_en, labels_en, labels_c2e,
                               sbert_doc_embeddings_cn, bow_cn, labels_cn, labels_e2c,
                               batch_size_en, batch_size_cn):
    # Convert English data to tensors
    embeddings_en = torch.tensor(sbert_doc_embeddings_en, dtype=torch.float32)
    bow_en_tensor = torch.tensor(bow_en, dtype=torch.float32)
    labels_en_tensor = torch.tensor(labels_en, dtype=torch.float32)
    labels_c2e = torch.tensor(labels_c2e, dtype=torch.float32)
    dataset_en = TensorDataset(embeddings_en, bow_en_tensor, labels_en_tensor, labels_c2e)
    dataloader_en = DataLoader(dataset_en, batch_size=batch_size_en, shuffle=True)

    # Convert Chinese data to tensors
    embeddings_cn = torch.tensor(sbert_doc_embeddings_cn, dtype=torch.float32)
    bow_cn_tensor = torch.tensor(bow_cn, dtype=torch.float32)
    labels_cn_tensor = torch.tensor(labels_cn, dtype=torch.float32)
    labels_e2c = torch.tensor(labels_e2c, dtype=torch.float32)
    dataset_cn = TensorDataset(embeddings_cn, bow_cn_tensor, labels_cn_tensor, labels_e2c)
    dataloader_cn = DataLoader(dataset_cn, batch_size=batch_size_cn, shuffle=True)

    return dataloader_en, dataloader_cn