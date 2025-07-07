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

# For audio loading
import numpy as np
import soundfile as sf
try:
    import sounddevice as sd
except ImportError:
    sd = None

import whisper
from transformers import MarianMTModel, MarianTokenizer

# =========================
# 模型加载和初始化相关函数
# =========================

def load_whisper_model(model_size="large"):
    """
    加载 Whisper 语音识别模型
    """
    return whisper.load_model(model_size)

def load_marianmt_model_and_tokenizer(model_name="Helsinki-NLP/opus-mt-zh-en"):
    """
    加载 MarianMT 翻译模型和分词器
    """
    tokenizer = MarianTokenizer.from_pretrained(model_name, local_files_only=True)
    mt_model = MarianMTModel.from_pretrained(model_name, local_files_only=True)
    return tokenizer, mt_model

def get_config():
    config_path = os.environ.get("XFL_CONFIG")
    assert config_path is not None, "Please set the XFL_CONFIG environment variable"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_model(checkpoint_path, config=None, device=None):
    """
    加载训练好的 OminiModel
    """
    if config is None:
        config = get_config()
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 初始化模型
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
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded full model weights from {checkpoint_path}")

    model.to("cuda")
    model.flux_pipe.to("cuda")

    model.eval()
    
    return model

# =========================
# 其他功能函数
# =========================

def extract_text_from_audio(audio, sample_rate=16000, whisper_model=None, tokenizer=None, mt_model=None, target_language="en"):
    """
    Extract text from an audio file or numpy array using Whisper and translate it to English using MarianMT.
    Returns a dict: {"chinese": ..., "english": ...}
    """
    print("Starting to load Whisper and translation models...")
    # 加载 Whisper 模型
    if whisper_model is None:
        whisper_model = load_whisper_model("large")
    # 加载 MarianMT 模型和分词器
    if tokenizer is None or mt_model is None:
        tokenizer, mt_model = load_marianmt_model_and_tokenizer("Helsinki-NLP/opus-mt-zh-en")
    print("Model loading finished.")

    # 判断 audio 是文件路径还是 numpy array
    if isinstance(audio, str) and os.path.exists(audio):
        # 文件路径
        result = whisper_model.transcribe(audio, language="zh")
    elif isinstance(audio, np.ndarray):
        # numpy array
        # whisper 需要 float32, 16kHz, 单声道
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        # whisper 的 transcribe 支持 np.ndarray
        result = whisper_model.transcribe(audio, language="zh", fp16=False, sample_rate=sample_rate)
    else:
        print("Invalid audio input for extract_text_from_audio")
        return {"chinese": "", "english": ""}

    chinese_text = result["text"]

    # Translate Chinese text to English
    inputs = tokenizer(chinese_text, return_tensors="pt", padding=True, truncation=True)
    translated = mt_model.generate(**inputs)
    english_text = tokenizer.decode(translated[0], skip_special_tokens=True)

    return {
        "chinese": chinese_text,
        "english": english_text
    }

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

def load_audio_data(audio_path=None, use_microphone=False, sample_rate=16000, duration=5):
    """
    Load audio data from file or record from microphone.
    Returns: numpy array of audio samples (float32)
    """
    if use_microphone:
        if sd is None:
            raise ImportError("sounddevice is not installed. Please install it to use microphone input.")
        print(f"Recording {duration} seconds of audio from microphone at {sample_rate} Hz...")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        audio = np.squeeze(audio)
        print("Audio recording complete.")
        return audio
    elif audio_path is not None and os.path.exists(audio_path):
        audio, sr = sf.read(audio_path)
        if sr != sample_rate:
            print(f"Warning: audio file sample rate {sr} != {sample_rate}, consider resampling.")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)  # Convert to mono
        return audio.astype(np.float32)
    else:
        print("No audio input provided.")
        return None

def extract_instruction_from_audio(audio_data, sample_rate=16000):
    """
    Convert audio (file path or numpy array) to instruction text using Whisper and MarianMT.
    Returns None if no instruction is extracted or if function is unavailable.
    """
    if audio_data is None:
        return None
    result = extract_text_from_audio(audio_data, sample_rate=sample_rate)
    if isinstance(result, dict):
        # 优先返回英文
        instruction = result.get("english", "") or result.get("chinese", "")
    else:
        instruction = result
    if instruction and isinstance(instruction, str) and instruction.strip():
        return instruction.strip()
    else:
        return None

def inference_single_image(model, condition_img, prompt, condition_type="SEED", 
                          position_delta=[0, 0], target_size=512, seed=42,
                          eeg_data=None, fnirs_data=None, ppg_data=None, motion_data=None,
                          audio_data=None, audio_sample_rate=16000):
    """
    Generate a single image using the trained model
    """
    # 修正音频处理逻辑：audio_data 既可以是文件路径也可以是 numpy array
    instruction_from_audio = None
    if audio_data is not None:
        instruction_from_audio = extract_instruction_from_audio(audio_data, sample_rate=audio_sample_rate)
    prompt_to_use = instruction_from_audio if instruction_from_audio is not None else prompt

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
        audio=audio_data,
    )
    
    # Determine if brain data is being used
    use_brain_condition = eeg_data is not None or fnirs_data is not None

    # Determine if audio data is being used
    use_audio_condition = audio_data is not None

    result = generate(
        model,
        model.flux_pipe,
        prompt=prompt_to_use,
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
        additional_condition5=audio_data,  # Audio data
        use_brain_condition=use_brain_condition,
        use_audio_condition=use_audio_condition,
        fuse_flag=False,
    )
    
    return result.images[0]

def batch_inference(model, input_dir, output_dir, caption_path=None, condition_type="SEED", 
                   target_size=512, position_delta=[0, -32], seed=42, brain_data_path=None,
                   audio_dir=None, audio_sample_rate=16000):
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
        # If audio instruction exists, override prompt
        # 修正音频处理逻辑
        instruction_from_audio = None
        audio_file = os.path.join(audio_dir, img_file.replace(".png", ".mp3"))
        audio_data = load_audio_data(audio_path=audio_file)
        if audio_data is not None:
            instruction_from_audio = extract_instruction_from_audio(audio_data, sample_rate=audio_sample_rate)
        prompt_to_use = instruction_from_audio if instruction_from_audio is not None else prompt
        
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
            prompt_to_use, 
            condition_type=condition_type,
            position_delta=position_delta,
            target_size=target_size,
            seed=seed,
            eeg_data=eeg_data,
            fnirs_data=fnirs_data,
            ppg_data=ppg_data,
            motion_data=motion_data,
            audio_data=audio_data,
            audio_sample_rate=audio_sample_rate
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
    # Audio options
    parser.add_argument("--audio_path", type=str, default=None, help="Path to audio file for audio input")
    parser.add_argument("--use_microphone", action="store_true", help="Use microphone input for audio")
    parser.add_argument("--audio_sample_rate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument("--audio_duration", type=int, default=5, help="Audio duration in seconds (for microphone input)")
    
    args = parser.parse_args()
    
    # Load config
    config = get_config()
    

    if args.single_image and args.prompt:

        ###
        # TODO: 具体demo里可能还需要一个input image展示给用户的逻辑
        ###

        # Load audio data if provided
        audio_data = None
        if args.audio_path or args.use_microphone:
            audio_data = load_audio_data(audio_path=args.audio_path, use_microphone=args.use_microphone,
                                        sample_rate=args.audio_sample_rate, duration=args.audio_duration)
            if audio_data is not None:
                print(f"Loaded audio data, shape: {audio_data.shape}")

        # 修正音频处理逻辑
        instruction_from_audio = None
        if audio_data is not None:
            instruction_from_audio = extract_instruction_from_audio(audio_data, sample_rate=args.audio_sample_rate)

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

        # If instruction_from_audio exists, override prompt
        prompt_to_use = instruction_from_audio if instruction_from_audio is not None else args.prompt

        result_img = inference_single_image(
            model,
            condition_img,
            prompt_to_use,
            condition_type=args.condition_type,
            position_delta=[args.position_delta_x, args.position_delta_y],
            target_size=args.target_size,
            seed=args.seed,
            eeg_data=eeg_data,
            fnirs_data=fnirs_data,
            ppg_data=ppg_data,
            motion_data=motion_data,
            audio_data=audio_data,
            audio_sample_rate=args.audio_sample_rate
        )
        
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, os.path.basename(args.single_image))
        result_img.save(output_path)
        print(f"Generated image saved to {output_path}")

if __name__ == "__main__":
    main()
