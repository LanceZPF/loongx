from torch.utils.data import DataLoader
import torch
import lightning as L
import yaml
import os
import time

from datasets import load_dataset

from .data import ImageConditionDataset, Subject200KDataset, CartoonDataset, SeedDataset
from .model import OminiModel
from .callbacks import TrainingCallback

from lightning.pytorch.strategies import DDPStrategy

import random
import numpy as np

import time

def get_rank():
    try:
        rank = int(os.environ.get("LOCAL_RANK"))
    except:
        rank = 0
    return rank


def get_config():
    config_path = os.environ.get("XFL_CONFIG")
    assert config_path is not None, "Please set the XFL_CONFIG environment variable"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def init_wandb(wandb_config, run_name):
    import wandb

    try:
        assert os.environ.get("WANDB_API_KEY") is not None
        wandb.init(
            project=wandb_config["project"],
            name=run_name,
            config={},
        )
    except Exception as e:
        print("Failed to initialize WanDB:", e)


def main():
    # Initialize
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    # 保证 cudnn 行为可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Lightning 也设一遍
    L.seed_everything(42)

    is_main_process, rank = get_rank() == 0, get_rank()
    torch.cuda.set_device(rank)
    config = get_config()
    training_config = config["train"]
    run_name = time.strftime("%Y%m%d-%H%M%S")

    # Initialize WanDB
    wandb_config = training_config.get("wandb", None)
    if wandb_config is not None and is_main_process:
        init_wandb(wandb_config, run_name)

    print("Rank:", rank)
    if is_main_process:
        print("Config:", config)

    # Initialize dataset and dataloader
    if training_config["dataset"]["type"] == "SEED":
        # Load the SEED dataset
        seed_dataset = SeedDataset(
            jsonl_path=training_config["dataset"]["jsonl_path"],
            image_dir=training_config["dataset"]["image_dir"],
            return_pil_image=training_config["dataset"].get("return_pil_image", False)
        )
        
        # Create dataset based on configuration
        dataset = seed_dataset

    elif training_config["dataset"]["type"] == "subject":
        dataset = load_dataset("LoongX/Subjects200K")

        # Define filter function
        def filter_func(item):
            if not item.get("quality_assessment"):
                return False
            return all(
                item["quality_assessment"].get(key, 0) >= 5
                for key in ["compositeStructure", "objectConsistency", "imageQuality"]
            )

        # Filter dataset
        if not os.path.exists("./cache/dataset"):
            os.makedirs("./cache/dataset")
        data_valid = dataset["train"].filter(
            filter_func,
            num_proc=16,
            cache_file_name="./cache/dataset/data_valid.arrow",
        )
        dataset = Subject200KDataset(
            data_valid,
            condition_size=training_config["dataset"]["condition_size"],
            target_size=training_config["dataset"]["target_size"],
            image_size=training_config["dataset"]["image_size"],
            padding=training_config["dataset"]["padding"],
            condition_type=training_config["condition_type"],
            drop_text_prob=training_config["dataset"]["drop_text_prob"],
            drop_image_prob=training_config["dataset"]["drop_image_prob"],
        )
    elif training_config["dataset"]["type"] == "img":
        # Load dataset text-to-image-2M
        dataset = load_dataset(
            "webdataset",
            data_files={"train": training_config["dataset"]["urls"]},
            split="train",
            cache_dir="cache/t2i2m",
            num_proc=32,
        )
        dataset = ImageConditionDataset(
            dataset,
            condition_size=training_config["dataset"]["condition_size"],
            target_size=training_config["dataset"]["target_size"],
            condition_type=training_config["condition_type"],
            drop_text_prob=training_config["dataset"]["drop_text_prob"],
            drop_image_prob=training_config["dataset"]["drop_image_prob"],
            position_scale=training_config["dataset"].get("position_scale", 1.0),
        )
    elif training_config["dataset"]["type"] == "cartoon":
        dataset = load_dataset("LoongX/oye-cartoon", split="train")
        dataset = CartoonDataset(
            dataset,
            condition_size=training_config["dataset"]["condition_size"],
            target_size=training_config["dataset"]["target_size"],
            image_size=training_config["dataset"]["image_size"],
            padding=training_config["dataset"]["padding"],
            condition_type=training_config["condition_type"],
            drop_text_prob=training_config["dataset"]["drop_text_prob"],
            drop_image_prob=training_config["dataset"]["drop_image_prob"],
        )
    else:
        raise NotImplementedError

    print("Dataset length:", len(dataset))
    train_loader = DataLoader(
        dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=training_config["dataloader_workers"],
    )

    # Initialize model
    trainable_model = OminiModel(
        flux_pipe_id=config["flux_path"],
        lora_config=training_config["lora_config"],
        device=f"cuda",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
    )

    # Callbacks for logging and saving checkpoints
    training_callbacks = (
        [TrainingCallback(run_name, training_config=training_config)]
        if is_main_process
        else []
    )

    # Initialize trainer
    trainer = L.Trainer(
        strategy=DDPStrategy(find_unused_parameters=True),
        accumulate_grad_batches=training_config["accumulate_grad_batches"],
        callbacks=training_callbacks,
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
        max_steps=training_config.get("max_steps", -1),
        # max_epochs=1,
        gradient_clip_val=training_config.get("gradient_clip_val", 0.5),
    )

    setattr(trainer, "training_config", training_config)

    # Save config
    save_path = training_config.get("save_path", "./output")
    if is_main_process:
        os.makedirs(f"{save_path}/{run_name}")
        with open(f"{save_path}/{run_name}/config.yaml", "w") as f:
            yaml.dump(config, f)

    start_time = time.time()
    # Start training
    trainer.fit(trainable_model, train_loader)
    
    # 计算总训练时间
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if is_main_process:
        print(f"{total_time} Total training time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

    if is_main_process:
        model_save_path = f"{save_path}/{run_name}/all_model_weights.pth"
        torch.save(trainable_model.state_dict(), model_save_path)
        print(f"Full model weights saved to {model_save_path}")

if __name__ == "__main__":
    main()
