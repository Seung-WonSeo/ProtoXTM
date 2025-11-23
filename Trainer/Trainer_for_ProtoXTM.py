import time
import torch

def train_ProtoXTM(model, dataloader_en, dataloader_cn, optimizer, num_epochs=500, device='cpu'):
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        start_time = time.time()  # 에폭 시작 시간 기록
        epoch_loss = 0.0

        # Zip the two dataloaders to iterate over them simultaneously
        for (x_en, x_en_bow, labels_en, labels_c2e), (x_cn, x_cn_bow, labels_cn, labels_e2c) in zip(dataloader_en, dataloader_cn):
            # Move data to the specified device
            x_en = x_en.to(device)
            x_en_bow = x_en_bow.to(device)
            labels_en = labels_en.to(device)
            labels_c2e = labels_c2e.to(device)
            x_cn = x_cn.to(device)
            x_cn_bow = x_cn_bow.to(device)
            labels_cn = labels_cn.to(device)
            labels_e2c = labels_e2c.to(device)

            # Forward pass
            outputs = model(x_en, x_cn, x_en_bow, x_cn_bow, labels_en, labels_cn, labels_c2e, labels_e2c)

            # Handle potential keys in the outputs
            tm_loss = outputs.get('topic_modeling_loss', torch.tensor(0.0, device=device))
            dpcl_loss = outputs.get('contrastive_loss', torch.tensor(0.0, device=device))
            total_loss = outputs.get('total_loss', tm_loss + dcl_loss)

            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        end_time = time.time()  # 에폭 종료 시간 기록
        epoch_time = end_time - start_time

        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Loss: {epoch_loss / len(dataloader_en):.4f}, "
              f"Time: {epoch_time:.2f} sec")

    return model