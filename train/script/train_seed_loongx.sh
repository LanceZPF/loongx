# Specify the config file path and the GPU devices to use
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Specify the config file path
# export XFL_CONFIG=./train/config/subject_512.yaml
export XFL_CONFIG=./train/config/seed_512.yaml

# # Specify the WANDB API key
# export WANDB_API_KEY='54c44deda10beca581c7d830b5190051e024d7df'

echo $XFL_CONFIG
export TOKENIZERS_PARALLELISM=true

accelerate launch --main_process_port 26353 -m src.train.train