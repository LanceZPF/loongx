# LoongX Training 🛠️

This directory contains the training code for LoongX. You can train your own LoongX model by customizing any control tasks (3D, multi-view, pose-guided, try-on, etc.) with the FLUX model.

## Quick Start

1. Prepare your dataset and configuration:
   - Place your training data in the appropriate directory
   - Configure your training parameters in `config/` directory

2. Start training:
bash train/script/train_seed_loongx.sh
```

## Configuration

The training configuration is defined in YAML files under the `config/` directory. You can customize the following parameters:

- Model architecture
- Training hyperparameters
- Dataset paths
- Logging and checkpoint settings

## Training Process

1. Data Preparation:
   - Organize your training data according to the specified format
   - Update the data paths in your config file

2. Model Training:
   - The training script will automatically handle model initialization, data loading, and training loop
   - Checkpoints are saved regularly
   - Training progress is logged and can be monitored

3. Evaluation:
   - The model is evaluated on a validation set during training
   - Metrics are logged and can be visualized

## Customization

You can customize the training process by:
1. Modifying the configuration files
2. Adding new data loaders
3. Implementing custom loss functions
4. Adding new evaluation metrics

## Troubleshooting

Common issues and solutions:
1. Out of memory: Reduce batch size or use gradient accumulation
2. Training instability: Adjust learning rate or use different optimizers
3. Poor results: Check data quality and model configuration

## Preparation

### Setup
1. **Environment**
    ```bash
    conda create -n loongx python=3.10
    conda activate loongx
    ```
2. **Requirements**
    ```bash
    pip install -r train/requirements.txt
    ```

### Dataset

    **Note:** Sampled instances from L-Mind are provided.

## Training

### Start training training
**Config file path**: `./train/config`

**Scripts path**: `./train/script`

`bash train/script/train_seed_loongx.sh`

**Note**: Detailed WanDB settings and GPU settings can be found in the script files and the config files.

### Other spatial control tasks
This repository supports 5 spatial control tasks: 
1. Canny edge to image (`canny`)
2. Image colorization (`coloring`)
3. Image deblurring (`deblurring`)
4. Depth map to image (`depth`)
5. Image to depth map  (`depth_pred`)
6. Image inpainting (`fill`)
7. Super resolution (`sr`)

You can modify the `condition_type` parameter in config file `config/canny_512.yaml` to switch between different tasks.

### Customize your own task
You can customize your own task by constructing a new dataset and modifying the training code.

<details>
<summary>Instructions</summary>

1. **Dataset** : 
   
   Construct a new dataset with the following format: (`src/train/data.py`)
    ```python
    class MyDataset(Dataset):
        def __init__(self, ...):
            ...
        def __len__(self):
            ...
        def __getitem__(self, idx):
            ...
            return {
                "image": image,
                "condition": condition_img,
                "condition_type": "your_condition_type",
                "description": description,
                "position_delta": position_delta
            }
    ```
    **Note:** For spatial control tasks, set the `position_delta` to be `[0, 0]`. For non-spatial control tasks, set `position_delta` to be `[0, -condition_width // 16]`.
2. **Condition**:
   
   Add a new condition type in the `Condition` class. (`src/flux/condition.py`)
    ```python
    condition_dict = {
        ...
        "your_condition_type": your_condition_id_number, # Add your condition type here
    }
    ...
    if condition_type in [
        ...
        "your_condition_type", # Add your condition type here
    ]:
        ...
    ```
3. **Test**: 
   
   Add a new test function for your task. (`src/train/callbacks.py`)
    ```python
    if self.condition_type == "your_condition_type":
        condition_img = (
            Image.open("images/vase.jpg")
            .resize((condition_size, condition_size))
            .convert("RGB")
        )
        ...
        test_list.append((condition_img, [0, 0], "A beautiful vase on a table."))
    ```

4. **Import relevant dataset in the training script**
   Update the file in the following section. (`src/train/train.py`)
   ```python
    from .data import (
        ImageConditionDataset,
        Subject200KDateset,
        MyDataset
    )
    ...
   
    # Initialize dataset and dataloader
    if training_config["dataset"]["type"] == "your_condition_type":
       ...
   ```
   
</details>

## Hardware requirement
**Note**: Memory optimization (like dynamic T5 model loading) is pending implementation.

**Recommanded**
- Hardware: 8x NVIDIA H100 GPUs
- Memory: ~80GB GPU memory

**Minimal**
- Hardware: 1x NVIDIA L20 GPU
- Memory: ~80GB GPU memory