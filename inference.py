import torch
import os
import yaml
import argparse
from PIL import Image
import json
import pickle
from tqdm import tqdm
import torch.multiprocessing as mp
import torch.distributed as dist
from src.flux.condition import Condition
from src.flux.generate import generate
from src.train.model import OminiModel
from accelerate import init_empty_weights, infer_auto_device_map

def get_config():
    config_path = os.environ.get("XFL_CONFIG")
    assert config_path is not None, "Please set the XFL_CONFIG environment variable"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_model(checkpoint_path, config=None, device=None):
    """
    Load a trained OminiModel from checkpoint
    """
    if config is None:
        config = get_config()
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize model
    model = OminiModel(
        flux_pipe_id=config["flux_path"],
        lora_config=config["train"]["lora_config"],
        device="cpu",
        dtype=getattr(torch, config["dtype"]),
        model_config=config.get("model", {}),
    )
    
    if "lora" in checkpoint_path:
        model.load_lora(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=model.device)
        if "state_dict" in checkpoint:
            # Handle the case where the checkpoint contains a state_dict key
            model.load_state_dict(checkpoint["state_dict"])
        else:
            # Handle the case where the checkpoint is the state_dict itself
            model.load_state_dict(checkpoint)
        print(f"Loaded full model weights from {checkpoint_path}")

    model.to("cuda")
    model.flux_pipe.to("cuda")

    model.eval()
    
    return model


def load_brain_data(pkl_path):
    """
    Load EEG and fNIRS data from pickle file
    """
    if not os.path.exists(pkl_path):
        print(f"Warning: Brain data file {pkl_path} not found")
        return {}
    
    with open(pkl_path, 'rb') as f:
        brain_data = pickle.load(f)
    
    return brain_data


def inference_single_image(model, condition_img, prompt, condition_type="SEED", 
                          position_delta=[0, 0], target_size=512, seed=42,
                          eeg_data=None, fnirs_data=None, ppg_data=None, motion_data=None):
    """
    Generate a single image using the trained model
    """
    generator = torch.Generator(device=model.device)
    generator.manual_seed(seed)
    
    condition = Condition(
        condition_type=condition_type,
        condition=condition_img,
        position_delta=position_delta,
        eeg=eeg_data,
        fnirs=fnirs_data,
        ppg=ppg_data,
        motion=motion_data,
    )
    
    # Determine if brain data is being used
    use_brain_condition = eeg_data is not None or fnirs_data is not None
    
    result = generate(
        model,
        model.flux_pipe,
        prompt=prompt,
        conditions=[condition],
        height=target_size,
        width=target_size,
        generator=generator,
        model_config=model.model_config,
        default_lora=True,
        additional_condition1=eeg_data,  # EEG data
        additional_condition2=fnirs_data,  # fNIRS data
        additional_condition3=ppg_data,  # PPG data
        additional_condition4=motion_data,  # Motion data
        use_brain_condition=use_brain_condition,
        fuse_flag=False,
    )
    
    return result.images[0]


def process_image_batch(rank, world_size, model, image_files, input_dir, output_dir, captions, 
                        brain_data, condition_type, position_delta, target_size, seed):
    """
    Process a batch of images assigned to this GPU
    """
    # Calculate the chunk of work for this process
    chunk_size = len(image_files) // world_size
    start_idx = rank * chunk_size
    end_idx = start_idx + chunk_size if rank < world_size - 1 else len(image_files)
    
    # Process assigned images
    for idx in range(start_idx, end_idx):
        img_file = image_files[idx]
        img_path = os.path.join(input_dir, img_file)
        condition_img = Image.open(img_path).convert('RGB')
        
        # Get caption for this image if available
        prompt = captions.get(img_file, "Edit this image")
        
        # Get brain data for this image if available
        eeg_data = None
        fnirs_data = None
        ppg_data = None
        motion_data = None
        if img_file in brain_data:
            if "EEG" in brain_data[img_file]:
                eeg_data = torch.tensor(brain_data[img_file]["EEG"], device=model.device)
            if "FNIRS" in brain_data[img_file]:
                fnirs_data = torch.tensor(brain_data[img_file]["FNIRS"], device=model.device)
            if "PPG" in brain_data[img_file]:
                ppg_data = torch.tensor(brain_data[img_file]["PPG"], device=model.device)
            if "Motion" in brain_data[img_file]:
                motion_data = torch.tensor(brain_data[img_file]["Motion"], device=model.device)
        
        # Generate the image
        result_img = inference_single_image(
            model, 
            condition_img, 
            prompt, 
            condition_type=condition_type,
            position_delta=position_delta,
            target_size=target_size,
            seed=seed,
            eeg_data=eeg_data,
            fnirs_data=fnirs_data,
            ppg_data=ppg_data,
            motion_data=motion_data
        )
        
        # Save the result
        output_path = os.path.join(output_dir, img_file)
        result_img.save(output_path)
        
        if rank == 0 and (idx - start_idx) % 10 == 0:
            print(f"Process {rank}: Completed {idx - start_idx}/{end_idx - start_idx} images")


def setup(rank, world_size):
    """
    Initialize the distributed environment
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """
    Clean up the distributed environment
    """
    dist.destroy_process_group()


def distributed_inference_worker(rank, world_size, args, config, model_loaded_event):
    """
    Run inference in a distributed manner with pre-loaded model
    """
    # Initialize the distributed environment
    setup(rank, world_size)
    
    # Set device for this process
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)
    
    # Wait for the model to be loaded by the main process
    model_loaded_event.wait()
    
    # Load model on this device
    model = load_model(args.checkpoint, config, device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load brain data if provided
    brain_data = {}
    if args.brain_data_path and os.path.exists(args.brain_data_path):
        brain_data = load_brain_data(args.brain_data_path)
        if rank == 0:
            print(f"Loaded brain data for {len(brain_data)} images")
    
    # Load captions if provided
    captions = {}
    if args.caption_path and os.path.exists(args.caption_path):
        with open(args.caption_path, 'r') as f:
            # Load JSONL file line by line
            for line in f:
                item = json.loads(line)
                # Extract image filename from target_image path
                img_filename = os.path.basename(item.get("source_image", ""))
                # Use speech2text as caption
                captions[img_filename] = item.get("speech2text", "Edit this image")

    # Process all images in the input directory
    image_files = [f for f in captions if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    if rank == 0:
        print(f"Processing {len(image_files)} images across {world_size} GPUs")
    
    # Process the batch of images assigned to this GPU
    process_image_batch(
        rank, 
        world_size, 
        model, 
        image_files, 
        args.input_dir, 
        args.output_dir, 
        captions, 
        brain_data, 
        args.condition_type, 
        [args.position_delta_x, args.position_delta_y], 
        args.target_size, 
        args.seed
    )
    
    # Wait for all processes to complete
    dist.barrier()
    
    if rank == 0:
        print(f"Processed {len(image_files)} images. Results saved to {args.output_dir}")
    
    # Clean up
    cleanup()


def batch_inference(model, input_dir, output_dir, caption_path=None, condition_type="SEED", 
                   target_size=512, position_delta=[0, -32], seed=42, brain_data_path=None):
    """
    Process all images in a directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load brain data if provided
    brain_data = {}
    if brain_data_path and os.path.exists(brain_data_path):
        brain_data = load_brain_data(brain_data_path)
        print(f"Loaded brain data for {len(brain_data)} images")
    
    # Load captions if provided
    captions = {}
    if caption_path and os.path.exists(caption_path):
        with open(caption_path, 'r') as f:
            # Load JSONL file line by line
            for line in f:
                item = json.loads(line)
                # Extract image filename from target_image path
                img_filename = os.path.basename(item.get("source_image", ""))
                # Use speech2text as caption
                if "speech2text" in item:
                    captions[img_filename] = item["speech2text"]
                # if "instruction" in item:
                elif "instruction" in item:
                    captions[img_filename] = item["instruction"]
                else:
                    captions[img_filename] = "Edit this image"

    # Process all images in the input directory
    image_files = [f for f in captions if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_file in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(input_dir, img_file)
        condition_img = Image.open(img_path).convert('RGB')
        
        # Get caption for this image if available
        prompt = captions.get(img_file, "Edit this image")
        
        # Get brain data for this image if available
        eeg_data = None
        fnirs_data = None
        ppg_data = None
        motion_data = None
        if img_file in brain_data:
            if "EEG" in brain_data[img_file]:
                eeg_data = torch.tensor(brain_data[img_file]["EEG"])
            if "FNIRS" in brain_data[img_file]:
                fnirs_data = torch.tensor(brain_data[img_file]["FNIRS"])
            if "PPG" in brain_data[img_file]:
                ppg_data = torch.tensor(brain_data[img_file]["PPG"])
            if "Motion" in brain_data[img_file]:
                motion_data = torch.tensor(brain_data[img_file]["Motion"])
        
        # Generate the image
        result_img = inference_single_image(
            model, 
            condition_img, 
            prompt, 
            condition_type=condition_type,
            position_delta=position_delta,
            target_size=target_size,
            seed=seed,
            eeg_data=eeg_data,
            fnirs_data=fnirs_data,
            ppg_data=ppg_data,
            motion_data=motion_data
        )
        
        # Save the result
        output_path = os.path.join(output_dir, img_file)
        result_img.save(output_path)
        
    print(f"Processed {len(image_files)} images. Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run inference with trained LoongX model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint directory")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output images")
    parser.add_argument("--caption_path", type=str, default = "/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/test_s2t.jsonl", help="Path to JSON file with captions")
    parser.add_argument("--condition_type", type=str, default="subject", help="Condition type (SEED, subject, canny, etc.)")
    parser.add_argument("--target_size", type=int, default=512, help="Target image size")
    parser.add_argument("--position_delta_x", type=int, default=0, help="Position delta X")
    parser.add_argument("--position_delta_y", type=int, default=-32, help="Position delta Y")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    parser.add_argument("--single_image", type=str, help="Path to single image for inference")
    parser.add_argument("--prompt", type=str, help="Prompt for single image inference")
    parser.add_argument("--brain_data_path", type=str, default='/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/data_final.pkl', help="Path to brain data pickle file (data_final.pkl)")
    parser.add_argument("--num_gpus", type=int, default=8, help="Number of GPUs to use for distributed inference")
    
    args = parser.parse_args()
    
    # Load config
    config = get_config()
    
    if args.single_image and args.prompt:
        # Single image inference (not distributed)
        model = load_model(args.checkpoint, config)
        
        # Load brain data if provided
        brain_data = {}
        if args.brain_data_path and os.path.exists(args.brain_data_path):
            brain_data = load_brain_data(args.brain_data_path)
        
        condition_img = Image.open(args.single_image).convert('RGB')
        
        # Get brain data for this image if available
        eeg_data = None
        fnirs_data = None
        ppg_data = None
        motion_data = None
        img_filename = os.path.basename(args.single_image)
        if brain_data and img_filename in brain_data:
            if "EEG" in brain_data[img_filename]:
                eeg_data = torch.tensor(brain_data[img_filename]["EEG"])
            if "FNIRS" in brain_data[img_filename]:
                fnirs_data = torch.tensor(brain_data[img_filename]["FNIRS"])
            if "PPG" in brain_data[img_filename]:
                ppg_data = torch.tensor(brain_data[img_filename]["PPG"])
            if "Motion" in brain_data[img_filename]:
                motion_data = torch.tensor(brain_data[img_filename]["Motion"])

        result_img = inference_single_image(
            model,
            condition_img,
            args.prompt,
            condition_type=args.condition_type,
            position_delta=[args.position_delta_x, args.position_delta_y],
            target_size=args.target_size,
            seed=args.seed,
            eeg_data=eeg_data,
            fnirs_data=fnirs_data,
            ppg_data=ppg_data,
            motion_data=motion_data
        )
        
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, os.path.basename(args.single_image))
        result_img.save(output_path)
        print(f"Generated image saved to {output_path}")
    else:
        # Determine number of available GPUs
        available_gpus = torch.cuda.device_count()
        num_gpus = min(args.num_gpus, available_gpus)
        
        if num_gpus <= 1:
            # Single GPU or CPU inference
            model = load_model(args.checkpoint, config)
            batch_inference(
                model,
                args.input_dir,
                args.output_dir,
                args.caption_path,
                condition_type=args.condition_type,
                target_size=args.target_size,
                position_delta=[args.position_delta_x, args.position_delta_y],
                seed=args.seed,
                brain_data_path=args.brain_data_path
            )
        else:
            # Multi-GPU distributed inference with sequential model loading
            print(f"Running distributed inference on {num_gpus} GPUs with sequential model loading")
            
            # Set the start method to 'spawn' to avoid CUDA re-initialization issues
            mp.set_start_method('spawn', force=True)
            
            # Create an event to signal when model loading is complete
            model_loaded_event = mp.Event()
            
            # Start worker processes
            processes = []
            for rank in range(num_gpus):
                p = mp.Process(
                    target=distributed_inference_worker,
                    args=(rank, num_gpus, args, config, model_loaded_event)
                )
                p.start()
                processes.append(p)
            
            # Signal that model loading can begin
            model_loaded_event.set()
            
            # Wait for all processes to complete
            for p in processes:
                p.join()


if __name__ == "__main__":
    main()
