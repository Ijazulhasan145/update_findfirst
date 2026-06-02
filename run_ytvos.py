import alphaclip
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model
from cutie.inference.sam2_tracker import SAM2Tracker
from utils import *
import os
import cv2
import json
import numpy as np
from PIL import Image
import torch
import torchvision as tv
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, BitsAndBytesConfig
import warnings
warnings.filterwarnings('ignore')


def test(args=None):

    num_references = args.num_references if args is not None else 3
    min_frame_distance = args.min_frame_distance if args is not None else 15
    multi_reference = args.multi_reference if args is not None else False
    dynamic_recovery = args.dynamic_recovery if args is not None else False
    tracker = args.tracker if args is not None else 'sam2'
    sam2_config = args.sam2_config if args is not None else 'sam2_hiera_l.yaml'
    sam2_checkpoint = args.sam2_checkpoint if args is not None else 'checkpoints/sam2_hiera_large.pt'
    alpha_clip_ckpt = args.alpha_clip_ckpt if (args is not None and args.alpha_clip_ckpt is not None) else 'weights/clip_l14_336_grit_20m_4xe.pth'
    recovery_threshold = 0.5

    # initialize EVF-SAM
    tokenizer, evfsam = init_models()

    # initialize Alpha-CLIP (auto-download weights if not found)
    if not os.path.exists(alpha_clip_ckpt):
        import urllib.request
        print(f'[INFO] Alpha-CLIP weights not found at {alpha_clip_ckpt}. Downloading...')
        os.makedirs(os.path.dirname(alpha_clip_ckpt) if os.path.dirname(alpha_clip_ckpt) else 'weights', exist_ok=True)
        url = 'https://huggingface.co/SunzeY/AlphaClip/resolve/main/clip_l14_336_grit_20m_4xe.pth'
        urllib.request.urlretrieve(url, alpha_clip_ckpt)
        print(f'[INFO] Downloaded Alpha-CLIP weights to {alpha_clip_ckpt}')
    clip, clip_preprocess = alphaclip.load('ViT-L/14@336px', alpha_vision_ckpt_pth=alpha_clip_ckpt, device='cuda')
    clip_preprocess_mask = transforms.Compose([transforms.Resize((336, 336)), transforms.Normalize(0.5, 0.26)])

    # initialize Tracker
    if tracker == 'cutie':
        cutie = get_default_model(config='ytvos_config')
        processor = InferenceCore(cutie, cfg=cutie.cfg)
    elif tracker == 'sam2':
        sam2_tracker = SAM2Tracker()

    # load videos
    output_dir = 'outputs'
    save_path_prefix = os.path.join(output_dir, 'Ref_YTVOS_val')
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
    root = args.data_root if (args is not None and args.data_root is not None) else '../DB/RVOS/YTVOS'
    img_folder = os.path.join(root, 'valid', 'JPEGImages')
    if os.path.exists(os.path.join(img_folder, 'JPEGImages')):
        img_folder = os.path.join(img_folder, 'JPEGImages')
    meta_file = os.path.join(root, 'meta_expressions', 'valid', 'meta_expressions.json')
    with open(meta_file, 'r') as f:
        data = json.load(f)['videos']
    valid_test_videos = set(data.keys())
    test_meta_file = os.path.join(root, 'meta_expressions', 'test', 'meta_expressions.json')
    with open(test_meta_file, 'r') as f:
        test_data = json.load(f)['videos']
    test_videos = set(test_data.keys())
    valid_videos = valid_test_videos - test_videos
    video_list = sorted([video for video in valid_videos])

    # inference
    for idx_, video in enumerate(video_list):
        print(idx_)
        metas = []
        expressions = data[video]['expressions']
        expression_list = list(expressions.keys())
        num_expressions = len(expression_list)
        for i in range(num_expressions):
            meta = {}
            meta['video'] = video
            meta['exp'] = expressions[expression_list[i]]['exp']
            meta['exp_id'] = expression_list[i]
            meta['frames'] = data[video]['frames']
            metas.append(meta)
        meta = metas
        video_name = video
        frames = data[video]['frames']
        video_len = len(frames)

        # input pre-process
        imgs_beit = []
        imgs_sam = []
        imgs_clip = []
        imgs_cutie = []
        frames_np = []
        for i in range(video_len):
            img_path = os.path.join(img_folder, video_name, frames[i] + '.jpg')
            image_np = cv2.imread(img_path)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            original_size_list = [image_np.shape[:2]]
            frames_np.append(image_np)

            # BEiT pre-process
            img_beit = beit3_preprocess(Image.open(img_path), 224)
            imgs_beit.append(img_beit)

            # SAM pre-process
            img_sam, resize_shape = sam_preprocess(image_np)
            imgs_sam.append(img_sam)

            # Alpha-CLIP pre-process
            img_clip = clip_preprocess(Image.open(img_path))
            imgs_clip.append(img_clip)

            # Cutie pre-process
            img_cutie = tv.transforms.ToTensor()(Image.open(img_path))
            imgs_cutie.append(img_cutie)

        # for each language
        for e in range(num_expressions):

            # make files
            video_name = meta[e]['video']
            exp = meta[e]['exp']
            exp_id = meta[e]['exp_id']
            frames = meta[e]['frames']
            save_path = os.path.join(save_path_prefix, video_name, exp_id)
            
            # Resume logic: Check if this expression has already been fully processed
            if os.path.exists(save_path) and len([f for f in os.listdir(save_path) if f.endswith('.png')]) == len(frames):
                print(f"Skipping: video={video_name}, exp_id={exp_id} (Already processed, resuming...)")
                continue

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # per-frame mask prediction
            ref_masks = []
            ref_scores = []
            ref_confidences = []
            ref_alignments = []
            ref_frames_all = []
            
            is_multi_ref = multi_reference and (num_references > 1)
            ref_num = max(5, num_references * 2) if is_multi_ref else 5
            
            for ref_idx in range(ref_num):
                i = int(ref_idx * (video_len - 1) / (ref_num - 1))
                words = tokenizer(exp, return_tensors='pt')['input_ids'].cuda()
                ref_mask, ref_score_conf = evfsam.inference(imgs_sam[i].unsqueeze(0).cuda(), imgs_beit[i].unsqueeze(0).cuda(), words, resize_shape, original_size_list)
                ref_mask = (ref_mask > 0).float()
                ref_masks.append(ref_mask)
                ref_confidences.append(ref_score_conf.item())

                # consider vision-text alignment in addition to segmentation confidence
                w1, w2 = 0.5, 0.5
                clip_text = alphaclip.tokenize([exp]).cuda()
                alpha = clip_preprocess_mask(ref_mask).cuda()
                image_features = clip.visual(imgs_clip[i].unsqueeze(0).cuda(), alpha.unsqueeze(0))
                text_features = clip.encode_text(clip_text)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                alignment_score = torch.matmul(image_features, text_features.transpose(0, 1))[0]
                ref_alignments.append(alignment_score.item())
                
                ref_score_final = w1 * ref_score_conf + w2 * alignment_score
                ref_scores.append(ref_score_final.item())
                
                ref_frames_all.append(
                    ReferenceFrame(
                        frame_idx=i,
                        mask=ref_mask.squeeze(0),
                        confidence=ref_score_conf.item(),
                        alignment_score=alignment_score.item(),
                        score=ref_score_final.item(),
                        image=imgs_cutie[i]
                    )
                )

            if tracker == 'sam2':
                # Initialize predictor with the frames list
                import time
                start_time = time.time()
                torch.cuda.reset_peak_memory_stats()
                
                sam2_tracker.initialize(frames_np, sam2_checkpoint, sam2_config, device='cuda')
                
                # Add mask prompts
                if is_multi_ref:
                    candidates_sorted = sorted(ref_frames_all, key=lambda x: x.score, reverse=True)
                    selected_references = []
                    for cand in candidates_sorted:
                        if all(abs(cand.frame_idx - sel.frame_idx) >= min_frame_distance for sel in selected_references):
                            selected_references.append(cand)
                            if len(selected_references) == num_references:
                                break
                    sum_scores = sum(ref.score for ref in selected_references)
                    for ref in selected_references:
                        ref.weight = ref.score / sum_scores if sum_scores > 0 else 1.0 / len(selected_references)
                    
                    selected_references = sorted(selected_references, key=lambda x: x.frame_idx)
                    for ref in selected_references:
                        sam2_tracker.add_mask(frame_idx=ref.frame_idx, mask=ref.mask)
                    key_frame_idx = selected_references[0].frame_idx
                    print("Tracker: SAM2")
                    for ref in selected_references:
                        print(f"Key Frame: {ref.frame_idx}")
                else:
                    best_ref_idx = torch.argmax(torch.stack([torch.tensor(s) for s in ref_scores], dim=0), dim=0).item()
                    best_i = int(best_ref_idx * (video_len - 1) / (ref_num - 1))
                    sam2_tracker.add_mask(frame_idx=best_i, mask=ref_masks[best_ref_idx].squeeze(0))
                    key_frame_idx = best_i
                    print("Tracker: SAM2")
                    print(f"Key Frame: {key_frame_idx}")
                    
                # Dictionary to accumulate logits/scores per frame
                logits_dict = {}
                conf_dict = {}
                
                # Forward propagation
                print("Propagation Direction: Forward")
                for out_frame_idx, out_obj_ids, out_mask_logits in sam2_tracker.propagate(start_frame_idx=key_frame_idx, reverse=False):
                    logits_dict[out_frame_idx] = out_mask_logits[0]
                    conf = torch.sigmoid(out_mask_logits[0]).max().item()
                    conf_dict[out_frame_idx] = conf
                    
                # Backward propagation
                print("Propagation Direction: Backward")
                for out_frame_idx, out_obj_ids, out_mask_logits in sam2_tracker.propagate(start_frame_idx=key_frame_idx, reverse=True):
                    logits_dict[out_frame_idx] = out_mask_logits[0]
                    conf = torch.sigmoid(out_mask_logits[0]).max().item()
                    conf_dict[out_frame_idx] = conf
                    
                # Calculate metrics
                total_time = time.time() - start_time
                fps_val = video_len / total_time
                max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
                
                print(f"FPS: {fps_val:.2f}")
                print(f"GPU memory usage: {max_mem:.2f} MB")
                print(f"total inference time: {total_time:.2f} s")
                
                avg_conf = np.mean(list(conf_dict.values())) if conf_dict else 0.0
                print(f"Tracking Confidence (mean): {avg_conf:.4f}")
                
                # Apply masks and save
                for i in range(video_len):
                    if i in logits_dict:
                        mask = (logits_dict[i] > 0.0).float()
                    else:
                        mask = torch.zeros((resize_shape[0][0], resize_shape[0][1]), device='cuda')
                        
                    # convert format and save
                    mask = mask.detach().cpu().numpy().astype(np.float32)
                    mask = Image.fromarray(mask * 255).convert('L')
                    save_file = os.path.join(save_path, frames[i] + '.png')
                    mask.save(save_file)
                    
                sam2_tracker.reset()
            elif is_multi_ref:
                # Sort candidates by final score descending
                candidates_sorted = sorted(ref_frames_all, key=lambda x: x.score, reverse=True)
                
                # Apply temporal diversity filtering
                selected_references = []
                for cand in candidates_sorted:
                    if all(abs(cand.frame_idx - sel.frame_idx) >= min_frame_distance for sel in selected_references):
                        selected_references.append(cand)
                        if len(selected_references) == num_references:
                            break
                            
                # Normalize weights
                sum_scores = sum(ref.score for ref in selected_references)
                for ref in selected_references:
                    ref.weight = ref.score / sum_scores if sum_scores > 0 else 1.0 / len(selected_references)
                    
                # Logging
                print("Selected reference frames:")
                for idx, ref in enumerate(selected_references):
                    print(f"Reference {idx+1}: frame={ref.frame_idx} confidence={ref.confidence:.4f} alignment={ref.alignment_score:.4f} score={ref.score:.4f} weight={ref.weight:.4f}")
                    
                # Sort selected references by frame_idx for interval propagation
                selected_references = sorted(selected_references, key=lambda x: x.frame_idx)
                first_ref = selected_references[0]
                
                # Initialize memory with all references
                processor.initialize_memory(selected_references)
                
                # backward pass
                for i in range(first_ref.frame_idx, -1, -1):
                    if i == first_ref.frame_idx:
                        processor.reinject_reference(imgs_cutie[i].cuda(), first_ref.mask, i)
                        mask = first_ref.mask
                    else:
                        mask_prob = processor.step(imgs_cutie[i].cuda())
                        
                        # Dynamic recovery
                        if dynamic_recovery:
                            tracking_confidence = mask_prob[1:].max().item()
                            if tracking_confidence < recovery_threshold:
                                # Find nearest reference frame in time
                                nearest_ref = min(selected_references, key=lambda r: abs(r.frame_idx - i))
                                processor.reinject_reference(imgs_cutie[nearest_ref.frame_idx].cuda(), nearest_ref.mask, nearest_ref.frame_idx)
                                mask_prob = processor.step(imgs_cutie[i].cuda())
                                
                        mask = processor.output_prob_to_mask(mask_prob).float()
                        
                    if i == 0:
                        processor.clear_memory()
                        
                    # convert format and save
                    mask = mask.detach().cpu().numpy().astype(np.float32)
                    mask = Image.fromarray(mask * 255).convert('L')
                    save_file = os.path.join(save_path, frames[i] + '.png')
                    mask.save(save_file)

                # forward pass
                for i in range(first_ref.frame_idx, video_len):
                    ref_found = [ref for ref in selected_references if ref.frame_idx == i]
                    if ref_found:
                        ref = ref_found[0]
                        processor.reinject_reference(imgs_cutie[i].cuda(), ref.mask, i)
                        mask = ref.mask
                    else:
                        mask_prob = processor.step(imgs_cutie[i].cuda())
                        
                        # Dynamic recovery
                        if dynamic_recovery:
                            tracking_confidence = mask_prob[1:].max().item()
                            if tracking_confidence < recovery_threshold:
                                nearest_ref = min(selected_references, key=lambda r: abs(r.frame_idx - i))
                                processor.reinject_reference(imgs_cutie[nearest_ref.frame_idx].cuda(), nearest_ref.mask, nearest_ref.frame_idx)
                                mask_prob = processor.step(imgs_cutie[i].cuda())
                                
                        mask = processor.output_prob_to_mask(mask_prob).float()
                        
                    if i == video_len - 1:
                        processor.clear_memory()
                        
                    # convert format and save
                    mask = mask.detach().cpu().numpy().astype(np.float32)
                    mask = Image.fromarray(mask * 255).convert('L')
                    save_file = os.path.join(save_path, frames[i] + '.png')
                    mask.save(save_file)
            else:
                # select reference frame with highest mask score
                best_ref_idx = torch.argmax(torch.stack([torch.tensor(s) for s in ref_scores], dim=0), dim=0)
                best_i = int(best_ref_idx * (video_len - 1) / (5 - 1))
                
                print("Selected reference frame:")
                print(f"Reference 1: frame={best_i} confidence={ref_confidences[best_ref_idx]:.4f} alignment={ref_alignments[best_ref_idx]:.4f} score={ref_scores[best_ref_idx]:.4f} weight=1.0000")

                # forward pass
                for i in range(best_i, video_len):
                    if i == best_i:
                        mask_prob = processor.step(imgs_cutie[i].cuda(), ref_masks[best_ref_idx].squeeze(0), objects=[1])
                    else:
                        mask_prob = processor.step(imgs_cutie[i].cuda())
                    mask = processor.output_prob_to_mask(mask_prob).float()

                    # clear memory for each sequence
                    if i == video_len - 1:
                        processor.clear_memory()

                    # convert format
                    mask = mask.detach().cpu().numpy().astype(np.float32)
                    mask = Image.fromarray(mask * 255).convert('L')
                    save_file = os.path.join(save_path, frames[i] + '.png')
                    mask.save(save_file)

                # backward pass
                for i in range(best_i, -1, -1):
                    if i == best_i:
                        mask_prob = processor.step(imgs_cutie[i].cuda(), ref_masks[best_ref_idx].squeeze(0), objects=[1])
                    else:
                        mask_prob = processor.step(imgs_cutie[i].cuda())
                    mask = processor.output_prob_to_mask(mask_prob).float()

                    # clear memory for each sequence
                    if i == 0:
                        processor.clear_memory()

                    # convert format
                    mask = mask.detach().cpu().numpy().astype(np.float32)
                    mask = Image.fromarray(mask * 255).convert('L')
                    save_file = os.path.join(save_path, frames[i] + '.png')
                    mask.save(save_file)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_references', type=int, default=3)
    parser.add_argument('--min_frame_distance', type=int, default=15)
    parser.add_argument('--multi_reference', action='store_true', default=False)
    parser.add_argument('--dynamic_recovery', action='store_true', default=False)
    parser.add_argument('--tracker', type=str, choices=['cutie', 'sam2'], default='sam2')
    parser.add_argument('--sam2_config', type=str, default='sam2_hiera_l.yaml')
    parser.add_argument('--sam2_checkpoint', type=str, default='checkpoints/sam2_hiera_large.pt')
    parser.add_argument('--alpha_clip_ckpt', type=str, default=None,
                        help='Absolute path to Alpha-CLIP weights (clip_l14_336_grit_20m_4xe.pth)')
    parser.add_argument('--data_root', type=str, default=None)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        test(args)
