import json
import pickle
from args import args_define
from typing import List, Tuple, Dict
import PIL.Image as Image
import clip
import open_clip
import numpy as np
import torch
import torch.nn.functional as F
from clip.model import CLIP
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import CIRRDataset, CIRCODataset
from utils import extract_image_features, device, collate_fn, PROJECT_ROOT, targetpad_transform
import torch
import torch.nn as nn
import torch.nn.functional as F
import umap
import matplotlib.pyplot as plt

from transformers import AutoModelForCausalLM, AutoProcessor

@torch.no_grad()
def cirr_generate_test_submission_file(dataset_path: str, clip_model_name: str, preprocess: callable, submission_name: str) -> None:
    """
    Generate the test submission file for the CIRR dataset given the pseudo tokens
    """

    # Load the CLIP model
    if clip_model_name == 'ViT-g-14':
        clip_model, _, _ = open_clip.create_model_and_transforms('ViT-g-14', device=device, pretrained='laion2b_s34b_b88k')
    else:
        clip_model, _ = clip.load(clip_model_name, device=device, jit=False)
    clip_model = clip_model.float().eval()

    # Compute the index features
    classic_test_dataset = CIRRDataset(dataset_path, 'test1', 'classic', preprocess)
    # if args.xxx:
    #     index_features = torch.load('feature/{}/index_features_G14.pt'.format(args.dataset))
    #     index_names = np.load('feature/{}/index_names_G14.npy'.format(args.dataset))
    #     index_names = index_names.tolist()
    # else:
    
    index_features, index_names, _ = extract_image_features(classic_test_dataset, clip_model)
    # index_features, index_names = extract_image_features_as_textfeatures(classic_test_dataset,clip_model)

    relative_test_dataset = CIRRDataset(dataset_path, 'test1', 'relative', preprocess)

    # Get the predictions dicts
    pairid_to_retrieved_images = \
        cirr_generate_test_dicts(relative_test_dataset, clip_model, index_features, index_names)

    submission = {
        'version': 'rc2',
        'metric': 'recall'
    }
    group_submission = {
        'version': 'rc2',
        'metric': 'recall_subset'
    }

    submission.update(pairid_to_retrieved_images)
    # group_submission.update(pairid_to_group_retrieved_images)

    submissions_folder_path = PROJECT_ROOT / 'data' / "test_submissions" / 'cirr'
    submissions_folder_path.mkdir(exist_ok=True, parents=True)

    with open(submissions_folder_path / f"{submission_name}.json", 'w+') as file:
        json.dump(submission, file, sort_keys=True)

    # with open(submissions_folder_path / f"subset_{submission_name}.json", 'w+') as file:
    #     json.dump(group_submission, file, sort_keys=True)


def cirr_generate_test_dicts(relative_test_dataset: CIRRDataset, clip_model: CLIP, index_features: torch.Tensor,
                             index_names: List[str]) \
        -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Generate the test submission dicts for the CIRR dataset given the pseudo tokens
    """

    # Get the predicted features
    predicted_features, reference_names, pairs_id, captions = \
        cirr_generate_test_predictions(clip_model, relative_test_dataset)
    # predict_image_features, predict_image_names, pairids1 = extract_image_features(relative_test_dataset, clip_model)
    print(f"Compute CIRR prediction dicts")
    



    # Normalize the image features
    index_features = index_features.to(device)
    index_features = F.normalize(index_features, dim=-1).float()
    # predict_image_features = predict_image_features.to(device)
    # predict_image_features = F.normalize(predict_image_features, dim=-1).float()



    # # Compute the distances between refimage&refimage and sort the results
    # dist1 = 1 - predict_image_features @ index_features.T
    # sorted_indices1 = torch.argsort(dist1, dim=-1).cpu()
    # sorted_index_names1 = np.array(index_names)[sorted_indices1]





    # # Delete the reference image from the results
    # reference_mask1 = torch.tensor(
    #     sorted_index_names1 != np.repeat(np.array(predict_image_names), len(index_names)).reshape(len(sorted_index_names1),
    #                                                                                          -1))
    # sorted_index_names1 = sorted_index_names1[reference_mask1].reshape(sorted_index_names1.shape[0],
    #                                                                 sorted_index_names1.shape[1] - 1)
    # sorted_id1 =  torch.argsort(torch.tensor(pairids1))
    # sorted_index_names1 = sorted_index_names1[sorted_id1]
    # sorted_index_names1 = sorted_index_names1.tolist()




    # self_attention_layer1 = SelfAttention(embed_dim=768, heads=8).to(device)
    # output = self_attention_layer1( predicted_features,captions)
    # self_attention_layer2 = SelfAttention(embed_dim=768, heads=4)


    # self_attention_layer1 = SingleHeadAttention(embed_dim=768).to(device)
    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T

    
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    
    sorted_index_names = np.array(index_names)[sorted_indices]
    print(len(sorted_index_names))
    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(sorted_index_names),
                                                                                             -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    
    # sorted_id =  torch.argsort(torch.tensor(pairs_id))


    # sorted_index_names = sorted_index_names[sorted_id]
    sorted_index_names = sorted_index_names.tolist()
    # pairid_to_retrieved_images = {}
    # for (pair_id, prediction, prediction1) in zip(pairs_id, sorted_index_names,sorted_index_names1):
    #     outs = []
    #     for item in prediction1:
    #         if len(outs)==50:
    #             break

    #         if item in prediction1[:1000]:
    #             outs.append(item)
    #     pairid_to_retrieved_images.update({str(int(pair_id)): outs})
    # Generate prediction dicts
    pairid_to_retrieved_images = {str(int(pair_id)): prediction[:50] for (pair_id, prediction) in
                                  zip(pairs_id, sorted_index_names)}
    # pairid_to_group_retrieved_images = {str(int(pair_id)): prediction[:3].tolist() for (pair_id, prediction) in
    #                                     zip(pairs_id, sorted_group_names)}
    # 将predicted_features和index_features拼接成一个大的特征矩阵
    combined_features = torch.cat([predicted_features, index_features], dim=0).cpu().detach().numpy()

    # 使用 UMAP 进行降维
    reducer = umap.UMAP()
    umap_embeddings = reducer.fit_transform(combined_features)

    # 可视化
    # 绘制文本特征的散点图（前4181个数据点）
    plt.scatter(umap_embeddings[:predicted_features.shape[0], 0], 
                umap_embeddings[:predicted_features.shape[0], 1], 
                label='Text', color='blue')

    # 绘制图片特征的散点图（接下来的2174个数据点）
    plt.scatter(umap_embeddings[predicted_features.shape[0]:, 0], 
                umap_embeddings[predicted_features.shape[0]:, 1], 
                label='Image', color='red')

    # 添加图例并显示图形
    plt.legend()
    plt.show()
    plt.savefig('umap_visualization.png')







    return pairid_to_retrieved_images


def cirr_generate_test_predictions(clip_model: CLIP, relative_test_dataset: CIRRDataset,
                                   ) -> \
        Tuple[torch.Tensor, List[str], List[str], List[List[str]]]:
    """
    Generate the test prediction features for the CIRR dataset given the pseudo tokens
    """

    # Create the test dataloader
    relative_test_loader = DataLoader(dataset=relative_test_dataset, batch_size=16, num_workers=8,
                                      pin_memory=False)
    
    predicted_features_list = []
    predicted_imagefeatures_list = []
    reference_names_list = []
    pair_id_list = []
    captions = []
    # group_members_list = []
    if args.type == 'G':
        tokenizer = open_clip.get_tokenizer('ViT-g-14')
    else:
        tokenizer = clip.tokenize
    # with open(f'../CIRR/cirr/image_splits/split.rc2.test1.json') as f:
    #     name_to_relpath = json.load(f)
    # Compute the predictions
    for batch in tqdm(relative_test_loader):
        reference_names = batch['reference_name']
        pairs_id = batch['pair_id']
        # relative_captions = batch['relative_caption']
        caption = batch['relative_caption']
        # reference_img_path = f"../CIRR/{name_to_relpath[reference_names]}"
        # raw_image = Image.open(reference_img_path).convert('RGB')
        # image_features = clip_model.encode_image(raw_image)
        # group_members = batch['group_members']                          #####候选图片
        # multi_caption = batch['multi_{}'.format(args.caption_type)]
        multi_caption = batch['multi_opt']                              ########参考图片描述文本
        # multi_gpt_caption = batch['multi_gpt_{}'.format(args.caption_type)]
        multi_gpt_caption = batch['multi_gpt_opt']                      ######## 目标图片描述文本

        # group_members = np.array(group_members).T.tolist()

        # input_captions = [
        #     f"a photo of $ that {rel_caption}" for rel_caption in relative_captions]
        if args.is_gpt_caption:
            input_captions = multi_gpt_caption

        # else:
        #     if args.is_rel_caption:
        #         input_captions = [f"a photo that {caption}" for caption in relative_captions]
        #     else:
        #         input_captions = multi_caption[0]
        
        if args.multi_caption:
            text_features_list = []
            for cap in input_captions:
                tokenized_input_captions = tokenizer(cap, context_length=77).to(device)
                text_features = clip_model.encode_text(tokenized_input_captions)
                text_features_list.append(text_features)
            text_features_list = torch.stack(text_features_list)
            text_features = torch.mean(text_features_list, dim=0)

        else:
            tokenized_input_captions = tokenizer(input_captions, context_length=77).to(device)
            text_features = clip_model.encode_text(tokenized_input_captions)
            
        predicted_features = F.normalize(text_features)


        caption = tokenizer(caption, context_length=77).to(device)
        caption_features = clip_model.encode_text(caption)
        caption_features = F.normalize(caption_features)
        captions.append(caption_features)
        predicted_features_list.append(predicted_features)
        reference_names_list.extend(reference_names)
        pair_id_list.extend(pairs_id)
        # predicted_imagefeatures_list.append(image_features)
        # group_members_list.extend(group_members)
    captions = torch.vstack(captions)
    predicted_features = torch.vstack(predicted_features_list)
    # predicted_imagefeatures = torch.vstack(predicted_imagefeatures_list)
    return predicted_features, reference_names_list, pair_id_list,captions



def extract_image_features_as_textfeatures(classic_test_dataset,clip_model):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    model_id = 'microsoft/Florence-2-large'
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().cuda()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    if args.type == 'G':
        tokenizer = open_clip.get_tokenizer('ViT-g-14')
    else:
        tokenizer = clip.tokenize
    # Create data loader
    loader = DataLoader(dataset=classic_test_dataset, batch_size=16,
                        num_workers=8, pin_memory=True)


    index_names = []
    
    # Extract features
    for batch in tqdm(loader):
        names = batch.get('image_name')
        index_names.extend(names)
        
    

    
    MULTI_CAPTION = True
    NUM_CAPTION = 1
    prompt = '<CAPTION>'
    

    print("-------------------------------------------------------------------------------begin gene----------------------------------------------------------------------------------------")
    index_features= []
    dic = {}
    for ans in tqdm(index_names):
        ref_img_id = ans
       
        reference_img_path = f"./CIRR/test1/{ref_img_id}.png"
        raw_image = Image.open(reference_img_path).convert('RGB')
        inputs = processor(text=prompt, images=raw_image, return_tensors="pt").to('cuda', torch.float16)
        # vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        
        if MULTI_CAPTION:
            generated_ids = model.generate(
                  input_ids=inputs["input_ids"].cuda(),
                  pixel_values=inputs["pixel_values"].cuda(),
                  max_new_tokens=1024,
                  early_stopping=True,
                  do_sample=False,
                  num_beams=3,
                  num_return_sequences=NUM_CAPTION
        )
            caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
            dic[ans] = caption
            tokenized_input_captions = tokenizer(caption, context_length=77).to(device)
            text_features = clip_model.encode_text(tokenized_input_captions)
            
            index_features.append(text_features)
    index_features = torch.vstack(index_features)
    with open("data.json", "w", encoding="utf-8") as file:
        json.dump(dic, file, ensure_ascii=False, indent=4)
    return index_features, index_names

       
















args = args_define.args 
def main():
    if args.eval_type in ['LDRE-B', 'LDRE-L', 'LDRE-G']:
        if args.eval_type == 'LDRE-B':
            clip_model_name = 'ViT-B/32'
        elif args.eval_type == 'LDRE-L':
            clip_model_name = 'ViT-L/14'
        else:
            clip_model_name = 'ViT-g-14'

        if clip_model_name == 'ViT-g-14':
            clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s34b_b88k')
        else:
            clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False)

        if args.preprocess_type == 'targetpad':
            print('Target pad preprocess pipeline is used')
            preprocess = targetpad_transform(1.25, 224)
        elif args.preprocess_type == 'clip':
            print('CLIP preprocess pipeline is used')
            preprocess = clip_preprocess
        else:
            raise ValueError("Preprocess type not supported")

        # if args.dataset.lower() == 'cirr':
        #     relative_test_dataset = CIRRDataset(args.dataset_path, 'test', 'relative', preprocess, no_duplicates=True)
        # elif args.dataset.lower() == 'circo':
        #     relative_test_dataset = CIRCODataset('../../../../data/circo', 'test', 'relative', preprocess)
        # else:
        #     raise ValueError("Dataset not supported")

        clip_model = clip_model.float().to(device)
        

    # print(f"Eval type = {args.eval_type} \t exp name = {args.exp_name} \t")
    print(f"Eval type = {args.eval_type} ")

    if args.dataset == 'cirr':
        cirr_generate_test_submission_file(args.dataset_path, clip_model_name, preprocess, args.submission_name)
    # if args.dataset == 'circo':
    #     circo_generate_test_submission_file(args.dataset_path, clip_model_name, preprocess, args.submission_name)

    else:
        raise ValueError("Dataset not supported")


if __name__ == '__main__':
    main()
