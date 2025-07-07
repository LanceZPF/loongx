import torch
import clip
import json
import os
import argparse
from tqdm import tqdm
from PIL import Image
from scipy import spatial
from torch import nn
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
import yaml
from transformers import CLIPProcessor, CLIPModel
import pandas as pd


def eval_distance(image_pairs, metric='l1'):
    """
    Using pytorch to evaluate l1 or l2 distance
    """
    if metric == 'l1':
        criterion = nn.L1Loss()
    elif metric == 'l2':
        criterion = nn.MSELoss()
    eval_score = 0
    results = {}
    for img_pair in tqdm(image_pairs):
        gen_img = Image.open(img_pair[0]).convert('RGB')
        gt_img = Image.open(img_pair[1]).convert('RGB')
        # resize to gt size
        gen_img = gen_img.resize(gt_img.size)
        # convert to tensor
        gen_img = transforms.ToTensor()(gen_img)
        gt_img = transforms.ToTensor()(gt_img)
        # calculate distance
        per_score = criterion(gen_img, gt_img).detach().cpu().numpy().item()
        eval_score += per_score
        # Store individual image result
        img_name = os.path.basename(img_pair[0])
        if img_name not in results:
            results[img_name] = {}
        results[img_name][metric] = per_score

    return eval_score / len(image_pairs), results

def eval_dino_i(args, image_pairs, model, processor, metric='dino'):
    """
    Calculate DINO score, the cosine similarity between the generated image and the ground truth image
    using Facebook's DINO model
    """
    def encode(image, model, processor):
        # Process the image using the DINO processor
        # Convert PIL image to tensor and apply transformations
        if isinstance(processor, transforms.Compose):
            image_tensor = processor(image).unsqueeze(0).to(args.device)
        else:
            # Fallback for other processor types
            image_tensor = processor(images=image, return_tensors="pt").to(args.device)
        
        with torch.no_grad():
            # Get image features from the DINO model
            image_features = model(image_tensor).detach().cpu().float()
        return image_features

    eval_score = 0
    results = {}
    for img_pair in tqdm(image_pairs):
        generated_features = encode(Image.open(img_pair[0]).convert('RGB'), model, processor)
        gt_features = encode(Image.open(img_pair[1]).convert('RGB'), model, processor)
        similarity = 1 - spatial.distance.cosine(generated_features.view(generated_features.shape[1]),
                                               gt_features.view(gt_features.shape[1]))
        if similarity > 1 or similarity < -1:
            raise ValueError("strange similarity value")
        eval_score = eval_score + similarity
        
        # Store individual image result
        img_name = os.path.basename(img_pair[0])
        if img_name not in results:
            results[img_name] = {}
        results[img_name][metric] = similarity
        
    return eval_score / len(image_pairs), results


def eval_clip_i(args, image_pairs, model, processor, metric='clip_i'):
    """
    Calculate CLIP-I score, the cosine similarity between the generated image and the ground truth image
    """
    def encode(image, model, processor):
        # Process the image using the CLIP processor
        image_input = processor(images=image, return_tensors="pt").to(args.device)
        with torch.no_grad():
            if metric == 'clip_i':
                # Get image features from the CLIP model
                image_features = model.get_image_features(image_input.pixel_values).detach().cpu().float()
            elif metric == 'dino':
                # For DINO, use the model directly
                image_features = model(image_input.pixel_values).detach().cpu().float()
        return image_features

    eval_score = 0
    results = {}
    for img_pair in tqdm(image_pairs):
        generated_features = encode(Image.open(img_pair[0]).convert('RGB'), model, processor)
        gt_features = encode(Image.open(img_pair[1]).convert('RGB'), model, processor)
        similarity = 1 - spatial.distance.cosine(generated_features.view(generated_features.shape[1]),
                                               gt_features.view(gt_features.shape[1]))
        if similarity > 1 or similarity < -1:
            raise ValueError("strange similarity value")
        eval_score = eval_score + similarity
        
        # Store individual image result
        img_name = os.path.basename(img_pair[0])
        if img_name not in results:
            results[img_name] = {}
        results[img_name][metric] = similarity
        
    return eval_score / len(image_pairs), results

def eval_clip_score(args, image_pairs, model, processor, caption_dict):
    """
    Calculate CLIP score, the cosine similarity between the image and caption
    return gen_clip_score, gt_clip_score
    """
    def clip_score(image_path, caption):
        image = Image.open(image_path).convert('RGB')
        # Process both image and text using the CLIP processor
        inputs = processor(text=caption, images=image, return_tensors="pt", padding=True).to(args.device)
        with torch.no_grad():
            outputs = model(**inputs)
            # Get the logits_per_image which represents the similarity
            similarity = outputs.logits_per_image.detach().cpu().float()
        return similarity.item()
    
    gen_clip_score = 0
    gt_clip_score = 0
    
    for img_pair in tqdm(image_pairs):
        gen_img_path = img_pair[0]
        gt_img_path = img_pair[1]
        gt_img_name = gt_img_path.split('/')[-1]
        gt_caption = caption_dict[gt_img_name]
        gen_clip_score += clip_score(gen_img_path, gt_caption)
        gt_clip_score += clip_score(gt_img_path, gt_caption)

    return gen_clip_score / len(image_pairs), gt_clip_score / len(image_pairs)

def eval_clip_t(args, image_pairs, model, processor, caption_dict):
    """
    Calculate CLIP-T score, the cosine similarity between the image and the text CLIP embedding
    """
    def encode_image(image, model, processor):
        # Process the image using the CLIP processor
        inputs = processor(images=image, return_tensors="pt").to(args.device)
        with torch.no_grad():
            image_features = model.get_image_features(inputs.pixel_values).detach().cpu().float()
        return image_features
    
    def encode_text(text, model, processor):
        # Process the text using the CLIP processor
        # Truncate text to avoid exceeding max position embeddings (77 tokens for CLIP)
        inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=77).to(args.device)
        with torch.no_grad():
            text_features = model.get_text_features(inputs.input_ids).detach().cpu().float()
        return text_features

    gen_clip_t = 0
    gt_clip_t = 0
    results = {}
    
    for img_pair in tqdm(image_pairs):
        gen_img_path = img_pair[0]
        gt_img_path = img_pair[1]
        gt_img_name = gt_img_path.split('/')[-1]
        
        # Get instruction from jsonl file
        img_id = gt_img_name.split('.')[0]  # Extract image ID without extension
        gt_caption = None
        
        # Find the corresponding instruction in caption_dict
        for item in caption_dict:
            if item["target_image"].endswith(f"{img_id}.jpg") or item["target_image"].endswith(f"{img_id}.png"):
                gt_caption = item["instruction"]
                break
        
        if gt_caption is None:
            print(f"Warning: No caption found for {gt_img_name}")
            # continue
            exit()

        gen_img = Image.open(gen_img_path).convert("RGB")
        gt_img = Image.open(gt_img_path).convert("RGB")

        generated_features = encode_image(gen_img, model, processor)
        gt_features = encode_image(gt_img, model, processor)
        
        # Get text CLIP embedding
        text_features = encode_text(gt_caption, model, processor)

        gen_similarity = 1 - spatial.distance.cosine(generated_features.view(generated_features.shape[1]),
                                                text_features.view(text_features.shape[1]))
        gt_similarity = 1 - spatial.distance.cosine(gt_features.view(gt_features.shape[1]),
                                                text_features.view(text_features.shape[1]))
        
        gen_clip_t += gen_similarity
        gt_clip_t += gt_similarity
        
        # Store individual image result
        img_name = os.path.basename(gen_img_path)
        if img_name not in results:
            results[img_name] = {}
        results[img_name]['clip-t'] = gen_similarity
        
    return gen_clip_t / len(image_pairs), gt_clip_t / len(image_pairs), results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--caption_path', type=str, default='/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/test_s2t.jsonl', help='Path to caption JSONL file')
    parser.add_argument('--generated_path', type=str, default='/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/generated-ns-0430', help='Path to generated images')
    parser.add_argument('--gt_path', type=str, default='/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/images', help='Path to ground truth images')
    parser.add_argument('--metric', type=str, default='l1,l2,clip-i,dino,clip-t', help='Metrics to evaluate')
    parser.add_argument('--save_path', type=str, default='results', help='Path to save results')
    args = parser.parse_args()
    args.metric = args.metric.split(',')

    # Set device
    if args.device is None:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)

    # Load config
    # config = get_config()
    
    # Initialize evaluation metrics dictionary
    evaluated_metrics_dict = {}
    per_image_results = {}

    # Load image pairs
    image_pairs = []
    for img_name in os.listdir(args.generated_path):
        if img_name.endswith(('.png', '.jpg')):
            gen_path = os.path.join(args.generated_path, img_name)
            gt_path = os.path.join(args.gt_path, img_name.replace('_0','_1'))
            if os.path.exists(gt_path):
                image_pairs.append((gen_path, gt_path))
                # Initialize results for this image
                per_image_results[img_name] = {}

    print(f"Number of image pairs: {len(image_pairs)}")

    # Load caption dictionary from JSONL file
    caption_dict = []
    with open(args.caption_path, 'r') as f:
        for line in f:
            caption_dict.append(json.loads(line))

    # Evaluate metrics
    if 'l1' in args.metric:
        l1_score, l1_results = eval_distance(image_pairs, 'l1')
        print(f"L1 distance: {l1_score}")
        evaluated_metrics_dict['l1'] = l1_score
        # Update per-image results
        for img_name, metrics in l1_results.items():
            per_image_results[img_name].update(metrics)

    if 'l2' in args.metric:
        l2_score, l2_results = eval_distance(image_pairs, 'l2')
        print(f"L2 distance: {l2_score}")
        evaluated_metrics_dict['l2'] = l2_score
        # Update per-image results
        for img_name, metrics in l2_results.items():
            per_image_results[img_name].update(metrics)

    if 'clip-i' in args.metric:
        temp_path = '/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
        # model, transform =  clip.load("ViT-B/32", device=args.device, download_root=str(temp_path))
        # temp_path = 'openai/clip-vit-base-patch32'
        model = CLIPModel.from_pretrained(temp_path, local_files_only=True).to(args.device)
        transform = CLIPProcessor.from_pretrained(temp_path, local_files_only=True)
        clip_i_score, clip_i_results = eval_clip_i(args, image_pairs, model, transform)
        print(f"CLIP-I score: {clip_i_score}")
        evaluated_metrics_dict['clip-i'] = clip_i_score
        # Update per-image results
        for img_name, metrics in clip_i_results.items():
            per_image_results[img_name].update(metrics)

    if 'dino' in args.metric:
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        model.eval()
        model.to(args.device)
        transform = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        dino_score, dino_results = eval_dino_i(args, image_pairs, model, transform, metric='dino')
        print(f"DINO score: {dino_score}")
        evaluated_metrics_dict['dino'] = dino_score
        # Update per-image results
        for img_name, metrics in dino_results.items():
            per_image_results[img_name].update(metrics)

    if 'clip-t' in args.metric:
        temp_path = '/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
        # model, transform =  clip.load("ViT-B/32", device=args.device, download_root=str(temp_path))
        # temp_path = 'openai/clip-vit-base-patch32'
        model = CLIPModel.from_pretrained(temp_path, local_files_only=True).to(args.device)
        transform = CLIPProcessor.from_pretrained(temp_path, local_files_only=True)
        gen_clip_t, gt_clip_t, clip_t_results = eval_clip_t(args, image_pairs, model, transform, caption_dict)
        print(f"CLIP-T score (generated): {gen_clip_t}")
        print(f"CLIP-T score (ground truth): {gt_clip_t}")
        evaluated_metrics_dict['clip-t_gen'] = gen_clip_t
        evaluated_metrics_dict['clip-t_gt'] = gt_clip_t
        # Update per-image results
        for img_name, metrics in clip_t_results.items():
            per_image_results[img_name].update(metrics)

    # Save results
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        
    # Save overall metrics to txt file
    txt_path = os.path.join(args.save_path, 'evaluation_metrics.txt')
    with open(txt_path, 'w') as f:
        for metric, value in evaluated_metrics_dict.items():
            f.write(f"{metric}: {value}\n")

    # Save per-image metrics to CSV file
    csv_path = os.path.join(args.save_path, 'per_image_metrics.csv')
    df = pd.DataFrame.from_dict(per_image_results, orient='index')
    df.index.name = 'image_name'
    df.to_csv(csv_path)
    print(f"Per-image metrics saved to {csv_path}")

    # Print results in a table format
    print("\nEvaluation Results:")
    print("-" * 50)
    print(f"{'Metric':<15} | {'Score':<10}")
    print("-" * 50)
    for metric, score in evaluated_metrics_dict.items():
        print(f"{metric:<15} | {score:<10.4f}")
    print("-" * 50)

if __name__ == "__main__":
    main()
