import gradio as gr
import alphaclip
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model
from cutie.inference.sam2_tracker import SAM2Tracker
from utils import *
import cv2
import os
import imageio
import numpy as np
from PIL import Image
import torch
import torchvision as tv
from torchvision import transforms
import warnings
warnings.filterwarnings('ignore')


def segment_video(video_path, prompt, gpu, num_references=3, min_frame_distance=15, multi_reference=True, dynamic_recovery=False, tracker='sam2', sam2_config='sam2_hiera_l.yaml', sam2_checkpoint='checkpoints/sam2_hiera_large.pt'):

    # GPU setting
    torch.cuda.set_device(int(gpu))
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):

        # load data
        reader = imageio.get_reader(video_path)
        fps = reader.get_meta_data().get('fps', 24) 
        frames = [frame for frame in reader]
        reader.close()

        # initialize EVF-SAM
        tokenizer, evfsam = init_models()

        # initialize Alpha-CLIP
        clip, clip_preprocess = alphaclip.load('ViT-L/14@336px', alpha_vision_ckpt_pth='weights/clip_l14_336_grit_20m_4xe.pth', device='cuda')
        clip_preprocess_mask = transforms.Compose([transforms.Resize((336, 336)), transforms.Normalize(0.5, 0.26)])

        # initialize Cutie (only if using cutie)
        if tracker == 'cutie':
            cutie = get_default_model(config='ytvos_config')
            processor = InferenceCore(cutie, cfg=cutie.cfg)

        # input pre-process
        video_len = len(frames)
        imgs_beit = []
        imgs_sam = []
        imgs_clip = []
        imgs_cutie = []
        for i in range(video_len):
            image_np = frames[i]
            original_size_list = [image_np.shape[:2]]

            # BEiT pre-process
            img_beit = beit3_preprocess(Image.fromarray(image_np), 224)
            imgs_beit.append(img_beit)

            # SAM pre-process
            img_sam, resize_shape = sam_preprocess(image_np)
            imgs_sam.append(img_sam)

            # Alpha-CLIP pre-process
            img_clip = clip_preprocess(Image.fromarray(image_np))
            imgs_clip.append(img_clip)

            # Cutie pre-process
            img_cutie = tv.transforms.ToTensor()(Image.fromarray(image_np))
            imgs_cutie.append(img_cutie)

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
            words = tokenizer(prompt, return_tensors='pt')['input_ids'].cuda()
            ref_mask, ref_score_conf = evfsam.inference(imgs_sam[i].unsqueeze(0).cuda(), imgs_beit[i].unsqueeze(0).cuda(), words, resize_shape, original_size_list)
            ref_mask = (ref_mask > 0).float()
            ref_masks.append(ref_mask)
            ref_confidences.append(ref_score_conf.item())

            # consider vision-text alignment in addition to segmentation confidence
            w1, w2 = 0.5, 0.5
            clip_text = alphaclip.tokenize([prompt]).cuda()
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

        # color work
        overlay_color = np.array([0, 255, 0], dtype=np.uint8)

        if tracker == 'sam2':
            # Initialize tracker
            sam2_tracker = SAM2Tracker()
            
            # Record start time
            import time
            start_time = time.time()
            torch.cuda.reset_peak_memory_stats()
            
            # Initialize predictor with the frames list
            sam2_tracker.initialize(frames, sam2_checkpoint, sam2_config, device='cuda')
            
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
                best_i = int(best_ref_idx * (video_len - 1) / (5 - 1))
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
            fps = video_len / total_time
            max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
            
            print(f"FPS: {fps:.2f}")
            print(f"GPU memory usage: {max_mem:.2f} MB")
            print(f"total inference time: {total_time:.2f} s")
            
            avg_conf = np.mean(list(conf_dict.values())) if conf_dict else 0.0
            print(f"Tracking Confidence (mean): {avg_conf:.4f}")
            
            # Apply masks and color frames
            for i in range(video_len):
                if i in logits_dict:
                    mask = (logits_dict[i] > 0.0).float()
                else:
                    mask = torch.zeros((resize_shape[0][0], resize_shape[0][1]), device='cuda')
                    
                mask = mask.detach().cpu().numpy() * 255
                mask = mask.astype(np.uint8)
                
                colored_mask = np.zeros_like(frames[0], dtype=np.uint8)
                colored_mask[mask == 255] = overlay_color
                alpha = 0.6
                frames[i] = cv2.addWeighted(colored_mask, alpha, frames[i], 1 - alpha, 0)
                
            sam2_tracker.reset()
        else:
            if is_multi_ref:
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
                            if tracking_confidence < 0.5:
                                # Find nearest reference frame in time
                                nearest_ref = min(selected_references, key=lambda r: abs(r.frame_idx - i))
                                processor.reinject_reference(imgs_cutie[nearest_ref.frame_idx].cuda(), nearest_ref.mask, nearest_ref.frame_idx)
                                mask_prob = processor.step(imgs_cutie[i].cuda())
                                
                        mask = processor.output_prob_to_mask(mask_prob).float()
                        
                    if i == 0:
                        processor.clear_memory()
                        
                    # convert format and color
                    mask = mask.detach().cpu().numpy() * 255
                    mask = mask.astype(np.uint8)
                    
                    colored_mask = np.zeros_like(frames[0], dtype=np.uint8)
                    colored_mask[mask == 255] = overlay_color
                    alpha = 0.6
                    frames[i] = cv2.addWeighted(colored_mask, alpha, frames[i], 1 - alpha, 0)

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
                            if tracking_confidence < 0.5:
                                nearest_ref = min(selected_references, key=lambda r: abs(r.frame_idx - i))
                                processor.reinject_reference(imgs_cutie[nearest_ref.frame_idx].cuda(), nearest_ref.mask, nearest_ref.frame_idx)
                                mask_prob = processor.step(imgs_cutie[i].cuda())
                                
                        mask = processor.output_prob_to_mask(mask_prob).float()
                        
                    if i == video_len - 1:
                        processor.clear_memory()
                        
                    # convert format and color
                    mask = mask.detach().cpu().numpy() * 255
                    mask = mask.astype(np.uint8)
                    
                    colored_mask = np.zeros_like(frames[0], dtype=np.uint8)
                    colored_mask[mask == 255] = overlay_color
                    alpha = 0.6
                    frames[i] = cv2.addWeighted(colored_mask, alpha, frames[i], 1 - alpha, 0)
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
                    mask = mask.detach().cpu().numpy() * 255
                    mask = mask.astype(np.uint8)

                    # color work
                    colored_mask = np.zeros_like(frames[0], dtype=np.uint8)
                    colored_mask[mask == 255] = overlay_color
                    alpha = 0.6
                    target_frame = frames[i]
                    overlayed_image = cv2.addWeighted(colored_mask, alpha, target_frame, 1 - alpha, 0)
                    frames[i] = overlayed_image
                    
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
                    mask = mask.detach().cpu().numpy() * 255
                    mask = mask.astype(np.uint8)

                    # color work
                    colored_mask = np.zeros_like(frames[0], dtype=np.uint8)
                    colored_mask[mask == 255] = overlay_color
                    alpha = 0.6
                    target_frame = frames[i]
                    overlayed_image = cv2.addWeighted(colored_mask, alpha, target_frame, 1 - alpha, 0)
                    frames[i] = overlayed_image
        
        # save output
        output_filename = 'sample/result.mp4'
        writer = imageio.get_writer(output_filename, fps=fps, codec='libx264')
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        return 'sample/result.mp4'


# gradio setting
demo = gr.Interface(
    fn=segment_video,
    inputs=[
        gr.Video(label='Input Video'),
        gr.Text(label='Text Prompt'),
        gr.Text(label='GPU Number'),
        gr.Slider(minimum=1, maximum=10, step=1, value=3, label='Number of References'),
        gr.Slider(minimum=1, maximum=50, step=1, value=15, label='Minimum Frame Distance'),
        gr.Checkbox(value=True, label='Enable Multi-Reference'),
        gr.Checkbox(value=False, label='Enable Dynamic Recovery'),
        gr.Dropdown(choices=['cutie', 'sam2'], value='sam2', label='Tracker Backend'),
        gr.Text(value='sam2_hiera_l.yaml', label='SAM2 Config'),
        gr.Text(value='checkpoints/sam2_hiera_large.pt', label='SAM2 Checkpoint')
    ],
    outputs=gr.Video(label='Output Mask'),
    title='FindTrack Demo Page',
    examples=[
        ['sample/agility.mp4', 'A dog running on grass.', 0, 3, 15, True, False, 'sam2', 'sam2_hiera_l.yaml', 'checkpoints/sam2_hiera_large.pt'],
        ['sample/elon.mp4', 'Elon Musk dancing in a suit.', 0, 3, 15, True, False, 'sam2', 'sam2_hiera_l.yaml', 'checkpoints/sam2_hiera_large.pt'],
        ['sample/trump.mp4', 'Donald Trump dancing and clapping in front of an audience.', 0, 3, 15, True, False, 'sam2', 'sam2_hiera_l.yaml', 'checkpoints/sam2_hiera_large.pt']
    ],
    allow_flagging="never"
)


if __name__ == '__main__':
    demo.launch(share=True)
