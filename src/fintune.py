
from args import args_define
import clip
import open_clip
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
from datasets import CIRRDataset
from utils import  device,targetpad_transform
import torch
import torch.nn as nn
import torch.nn.functional as F
import umap
from torch.optim import AdamW





# # 加载预训练的 CLIP 模型和处理器
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# # 设置为训练模式
# model.train()

# # 冻结编码器的前几层（可选）
# # for param in model.vision_model.parameters():
# #     param.requires_grad = False  # 只微调后几层

# # 定义优化器
# optimizer = AdamW(model.parameters(), lr=1e-5)

# # 示例训练数据（文本和图像对）
# texts = ["A sample description"] * batch_size
# images = [image_data] * batch_size  # 这里 image_data 是图像张量

# # 训练循环
# for epoch in range(num_epochs):
#     inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
#     outputs = model(**inputs)

#     # 计算损失（根据任务自定义）
#     image_features = outputs.image_embeds
#     text_features = outputs.text_embeds

#     # 对特征进行归一化
#     image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#     text_features = text_features / text_features.norm(dim=-1, keepdim=True)

#     similarity = torch.matmul(text_features, image_features.T)
#     loss = ...  # 自定义损失计算

#     # 反向传播和优化
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()

# print("微调完成")
args = args_define.args
def clip_finetune_cirr(num_epochs: int, clip_model_name: str, learning_rate: float, batch_size: int,
                       val_freq: int, transform: str, save_training: bool, encoder: str, save_best: bool,loss: str,
                       **kwargs):
    """
    Fine-tune CLIP on the CIRR dataset using as combining function the image-text element-wise sum
    :param num_epochs: number of epochs
    :param clip_model_name: CLIP model you want to use: "RN50", "RN101", "RN50x4"...
    :param learning_rate: fine-tuning learning rate
    :param batch_size: batch size
    :param validation_frequency: validation frequency expressed in epoch
    :param transform: preprocess transform you want to use. Should be in ['clip', 'squarepad', 'targetpad']. When
                targetpad is also required to provide `target_ratio` kwarg.
    :param save_training: when True save the weights of the Combiner network
    :param encoder: which CLIP encoder to fine-tune, should be in ['both', 'text', 'image']
    :param save_best: when True save only the weights of the best Combiner wrt three different averages of the metrics
    :param kwargs: if you use the `targetpad` transform you should prove `target_ratio`    :return:
    """

    # Save all the hyperparameters on a file
    clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False)

    if encoder == 'text':
        print('Only the CLIP text encoder will be fine-tuned')
        for param in clip_model.visual.parameters():
            param.requires_grad = False
    elif encoder == 'image':
        print('Only the CLIP image encoder will be fine-tuned')
        for param in clip_model.parameters():
            param.requires_grad = False
        for param in clip_model.visual.parameters():
            param.requires_grad = True
    elif encoder == 'both':
        print('Both CLIP encoders will be fine-tuned')
    else:
        raise ValueError("encoder parameter should be in ['text', 'image', both']")

    clip_model.eval().float()
    input_dim = clip_model.visual.input_resolution

    if transform == "clip":
        preprocess = clip_preprocess
        print('CLIP default preprocess pipeline is used')
    elif transform == "targetpad":
        target_ratio = kwargs['target_ratio']
        preprocess = targetpad_transform(target_ratio, input_dim)
        print(f'Target pad with {target_ratio = } preprocess pipeline is used')
    else:
        raise ValueError("Preprocess transform should be in ['clip', 'squarepad', 'targetpad']")
    dataset_path = args.dataset_path

    # Define the validation datasets
    relative_dataset = CIRRDataset(dataset_path,'test1', 'relative', preprocess)
    # classic_val_dataset = CIRRDataset(dataset_path, 'val', 'classic', preprocess)

    # Define the train dataset and the combining function
    relative_loader = DataLoader(dataset=relative_dataset, batch_size=batch_size,
                                       num_workers=8,pin_memory=False)

    # Define the optimizer, the loss and the grad scaler
    optimizer = AdamW(
        [{'params': filter(lambda p: p.requires_grad, clip_model.parameters()), 'lr': learning_rate,
          'betas': (0.9, 0.999), 'eps': 1e-7}])
    scaler = torch.GradScaler()

    # When save_best == True initialize the best results to zero
    if save_best:
        best_dist = 100
    if args.type == 'G':
        tokenizer = open_clip.get_tokenizer('ViT-g-14')
    else:
        tokenizer = clip.tokenize

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs} begin:     ----------------------------------------------------------------------------")
        
        train_bar = tqdm(relative_loader, ncols=150)
        for idx, batch in enumerate(train_bar):
            reference_image = batch['reference_image']
            multi_gpt_opt = batch['multi_gpt_opt']
            images_in_batch = reference_image.size(0)
            step = len(train_bar) * epoch + idx

            optimizer.zero_grad()

            reference_images = reference_image.to(device, non_blocking=True)
            
            captions = multi_gpt_opt

            # Extract the features, compute the logits and the loss
            with torch.autocast("cuda"):
                reference_features = clip_model.encode_image(reference_images)
                text_features_list = []
                for cap in captions:
                    tokenized_input_captions = tokenizer(cap, context_length=77).to(device,non_blocking=True)
                    text_features = clip_model.encode_text(tokenized_input_captions)
                    
                    text_features_list.append(text_features)
                  
                text_features_list = torch.stack(text_features_list)
                
                text_features = torch.mean(text_features_list, dim=0)

                images = F.normalize(reference_features,dim=1)
                texts = F.normalize(text_features,dim=1)
                
                if loss == 'contrastive_align':
                    loss = contrastive_alignment_loss(texts,images)
                else:
                    loss = contrastive_loss(texts,images,images_in_batch)
                if idx % 20 == 0:
                    print(f"Batch {idx} - Loss: {loss.item():.4f}")


            # Backpropagate and update the weights
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if epoch % val_freq == 0:
                train_bar = relative_loader
                images = []
                texts = []
                for reference_image, multi_gpt_opt in enumerate(train_bar):
                    images_in_batch = reference_images.size(0)
                 

                    with torch.no_grad():

                        reference_images = reference_image.to(device, non_blocking=True)
                        captions = multi_gpt_opt.to(device, non_blocking=True)

                        # Extract the features, compute the logits and the loss
                        with torch.autocast("cuda"):
                            reference_features = clip_model.encode_image(reference_images)
                            text_features_list = []
                            for cap in captions:
                                tokenized_input_captions = tokenizer(cap, context_length=77).to(device,non_blocking=True)
                                text_features = clip_model.encode_text(tokenized_input_captions)
                                text_features_list.append(text_features)
                            text_features_list = torch.stack(text_features_list)
                            text_features = torch.mean(text_features_list, dim=1)

                            images.append(reference_features)
                            texts.append(text_features)

                images = F.normalize(torch.vstack(images),dim=1)
                texts = F.normalize(torch.vstack(texts),dim=1)
                reducer = umap.UMAP()

                umap_image_embeddings = reducer.fit_transform(images)
                umap_text_embeddings = reducer.fit_transform(texts)

                x_image,y_image = torch.mean(umap_image_embeddings,dim=0)
                x_text,y_text = torch.mean( umap_text_embeddings,dim=0 )
                distance = math.sqrt((x_text - x_image) ** 2 + (y_text - y_image) ** 2)

                print(f"Epoch {epoch} - Distance: {distance}")

                if save_training:
                    if save_best and distance < best_dist:
                        best_dist = distance
                        models_path = args.dataset_path / "saved_models"
                        models_path.mkdir(exist_ok=True, parents=True)
                        torch.save({'epoch': epoch, 'clip_model': clip_model.state_dict(),}, str(models_path / f'clip_model_{epoch}.pt'))
                    


def contrastive_alignment_loss(text_features, image_features, temperature=0.07):
    # 归一化特征向量
    text_features = F.normalize(text_features, p=2, dim=1)
    image_features = F.normalize(image_features, p=2, dim=1)
    
    # 计算相似度矩阵
    logits = torch.matmul(text_features, image_features.T) / temperature
    
    # 创建目标标签
    batch_size = text_features.size(0)
    labels = torch.arange(batch_size, device=device)
    
    # 计算交叉熵损失
    loss_t2i = F.cross_entropy(logits, labels)  # 文本到图像的损失
    loss_i2t = F.cross_entropy(logits.T, labels)  # 图像到文本的损失
    
    # 返回平均损失
    return (loss_t2i + loss_i2t) / 2

def contrastive_loss(text_features, image_features, batch_size, margin=1.0):
    """
    实现对比损失函数。
    
    参数：
    text_features: 文本特征的张量 (batch_size, embedding_dim)
    image_features: 图像特征的张量 (batch_size, embedding_dim)
    labels: 标签张量 (batch_size)，值为1表示正样本对，0表示负样本对
    margin: 阈值，用于负样本对的最小距离

    返回：
    损失值
    """
    num_positive = batch_size // 2
    labels = torch.zeros(batch_size)
    labels[:num_positive] = 1  # 前一半是正样本对
    print(type(labels))
    while True:
        shuffled_indices = torch.randperm(batch_size)
        image_features_shuffled = image_features[shuffled_indices]

        # 检查是否有碰巧匹配的情况
        correct_matches = (shuffled_indices == torch.arange(batch_size)).sum().item()
        if correct_matches == 0:
            break  # 没有碰巧匹配，退出循环并使用此打乱结果

    # 前 num_positive 对用原始特征计算损失（正样本对）
    text_features_positive = text_features[:num_positive]
    image_features_positive = image_features[:num_positive]

    # 后半部分用打乱的特征计算损失（负样本对）
    text_features_negative = text_features[num_positive:]
    image_features_negative = image_features_shuffled[num_positive:]

    # 合并特征
    text_features_combined = torch.cat([text_features_positive, text_features_negative], dim=0)
    image_features_combined = torch.cat([image_features_positive, image_features_negative], dim=0)

    # 计算文本和图像特征之间的余弦相似度
    
    cosine_similarity = F.cosine_similarity(text_features_combined.unsqueeze(1), image_features_combined.unsqueeze(0), dim=2)
    ##计算余弦距离
    cosine_distance = 1 - cosine_similarity
    
    # 计算损失
    positive_loss = torch.mean(torch.diag(torch.pow(cosine_distance, 2))*labels.to(device)) # 正样本对的相似度
    negative_loss =  torch.mean(torch.diag(torch.pow(torch.clamp(margin - cosine_distance, min=0.0), 2))*(1-labels).to(device)  ) # 负样本对的相似度
    
    # 总损失
    loss = (positive_loss + negative_loss) *0.5
    
    return loss




def main():
    if args.eval_type in ['LDRE-B', 'LDRE-L', 'LDRE-G']:
        if args.eval_type == 'LDRE-B':
            clip_model_name = 'ViT-B/32'
        elif args.eval_type == 'LDRE-L':
            clip_model_name = 'ViT-L/14'
        else:
            clip_model_name = 'ViT-g-14'
    
    
        
    
    clip_finetune_cirr(10, clip_model_name, learning_rate = 0.01, batch_size =16 ,
                       val_freq = 1 , transform = args.preprocess_type, save_training = True, encoder = 'both', save_best = True,loss='contrastive',**vars(args))


if __name__ == '__main__':
    main()
