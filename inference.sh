export XFL_CONFIG=/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/train/config/seed_512.yaml
export CUDA_VISIBLE_DEVICES=0

export MASTER_PORT=31600 

python inference.py --checkpoint /inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/runs/20250430-015450/all_model_weights.pth \
    --input_dir /inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/images \
    --output_dir ./generated-ns-0430 \
    --num_gpus 8