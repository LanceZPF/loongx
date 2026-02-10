import lightning as L
from diffusers.pipelines import FluxPipeline
import torch
from peft import LoraConfig, get_peft_model_state_dict
import torch.nn.functional as F

import prodigyopt

from ..flux.transformer import tranformer_forward
from ..flux.condition import Condition
from ..flux.pipeline_tools import encode_images, prepare_text_input


class OminiModel(L.LightningModule):
    def __init__(
        self,
        flux_pipe_id: str,
        lora_path: str = None,
        lora_config: dict = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
        use_brain_condition: bool = True,
        fuse_flag: bool = False,
    ):
        # Initialize the LightningModule
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        # Store dtype as a property for later use without setting it directly
        self._dtype = dtype

        # Load the Flux pipeline
        self.flux_pipe: FluxPipeline = (
            FluxPipeline.from_pretrained(flux_pipe_id).to(dtype=dtype).to(device)
        )
        self.transformer = self.flux_pipe.transformer
        self.transformer.gradient_checkpointing = gradient_checkpointing
        self.transformer.train()

        # Freeze the Flux pipeline
        self.flux_pipe.text_encoder.requires_grad_(False).eval()
        self.flux_pipe.text_encoder_2.requires_grad_(False).eval()
        self.flux_pipe.vae.requires_grad_(False).eval()

        # Initialize LoRA layers
        self.lora_layers = self.init_lora(lora_path, lora_config)

        self.to(device).to(dtype)

        self.fuse_flag = fuse_flag
        self.use_brain_condition = use_brain_condition
        
        # Target fixed lengths after spatial pyramid pooling
        self.eeg_fixed_length = 54
        self.fnirs_fixed_length = 6
        
        # Lightweight projection for EEG data [1,4,variable_length] -> [1,512,4096]
        # Using spatial pyramid pooling to handle variable length input
        self.eeg_projection = torch.nn.Sequential(
            # First flatten the channels
            torch.nn.Flatten(1, 1),  # Keep the last dimension, shape becomes [1,4,variable_length]
            # Apply spatial pyramid pooling to get fixed length
            # The SPP is implemented in the forward pass
            torch.nn.Linear(4 * self.eeg_fixed_length, 2048),  # Using fixed length after SPP
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 4096),  # Project to intermediate dimension
            torch.nn.Unflatten(1, (512, 8)),  # Reshape to [1,512,8]
            torch.nn.Linear(8, 4096)  # Final projection to target dimension
        ).to(device).to(dtype)
        
        # Lightweight projection for fNIRS data [1,6,variable_length] -> [1,768]
        # Using spatial pyramid pooling to handle variable length input
        self.fnirs_projection = torch.nn.Sequential(
            # First flatten the channels
            torch.nn.Flatten(1, 1),  # Keep the last dimension, shape becomes [1,6,variable_length]
            # Apply spatial pyramid pooling to get fixed length
            # The SPP is implemented in the forward pass
            torch.nn.Linear(6 * self.fnirs_fixed_length, 384),  # Using fixed length after SPP
            torch.nn.ReLU(),
            torch.nn.Linear(384, 768)  # Project to target dimension
        ).to(device).to(dtype)
        
        # Initialize fusion layers if fusion is enabled
        if fuse_flag:
            # Projection layer to map additional_condition1 to prompt_embeds dimension
            
            # Attention mechanism for fusion
            self.fusion_attention1 = torch.nn.MultiheadAttention(
                embed_dim=4096, 
                num_heads=4, 
                batch_first=True
            ).to(device).to(dtype)
            # Add these to trainable parameters
            
            # self.lora_layers.extend(list(self.fusion_attention1.parameters()))

            self.fusion_attention2 = torch.nn.MultiheadAttention(
                embed_dim=768, 
                num_heads=4, 
                batch_first=True
            ).to(device).to(dtype)
            # self.lora_layers.extend(list(self.fusion_attention2.parameters()))

    def load_lora(self, checkpoint_path: str):
        """
        Load LoRA weights from a checkpoint directory
        
        Args:
            checkpoint_path (str): Path to the directory containing LoRA weights
        """
        # Load the LoRA weights
        self.flux_pipe.load_lora_weights(checkpoint_path)
        
        # Make sure the transformer is in evaluation mode for inference
        self.transformer.eval()
        
        return self

    def spatial_pyramid_pooling(self, x, output_size):
        """
        Apply spatial pyramid pooling to convert variable length input to fixed length
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, variable_length]
            output_size (int): Desired fixed length after pooling
            
        Returns:
            torch.Tensor: Tensor of shape [batch_size, channels, output_size]
        """
        batch_size, channels, length = x.shape
        
        # If input length is already the desired length, return as is
        if length == output_size:
            return x
            
        # Calculate adaptive pool sizes to achieve the desired output size
        # We'll use a simple approach with equal-sized bins
        result = F.adaptive_avg_pool1d(x, output_size)
        
        return result

    def init_lora(self, lora_path: str, lora_config: dict):
        assert lora_path or lora_config
        if lora_path:
            # TODO: Implement this
            raise NotImplementedError
        else:
            self.transformer.add_adapter(LoraConfig(**lora_config))
            # TODO: Check if this is correct (p.requires_grad)
            lora_layers = filter(
                lambda p: p.requires_grad, self.transformer.parameters()
            )
        return list(lora_layers)

    def save_lora(self, path: str):
        FluxPipeline.save_lora_weights(
            save_directory=path,
            transformer_lora_layers=get_peft_model_state_dict(self.transformer),
            safe_serialization=True,
        )

    def configure_optimizers(self):
        # Freeze the transformer
        self.transformer.requires_grad_(False)
        opt_config = self.optimizer_config

        # Set the trainable parameters
        self.trainable_params = self.lora_layers

        # Unfreeze trainable parameters
        for p in self.trainable_params:
            p.requires_grad_(True)

        # Initialize the optimizer
        if opt_config["type"] == "AdamW":
            optimizer = torch.optim.AdamW(self.trainable_params, **opt_config["params"])
        elif opt_config["type"] == "Prodigy":
            optimizer = prodigyopt.Prodigy(
                self.trainable_params,
                **opt_config["params"],
            )
        elif opt_config["type"] == "SGD":
            optimizer = torch.optim.SGD(self.trainable_params, **opt_config["params"])
        else:
            raise NotImplementedError

        return optimizer

    def training_step(self, batch, batch_idx):
        step_loss = self.step(batch)
        self.log_loss = (
            step_loss.item()
            if not hasattr(self, "log_loss")
            else self.log_loss * 0.95 + step_loss.item() * 0.05
        )
        return step_loss

    def step(self, batch):
        imgs = batch["image"]
        conditions = batch["condition"]
        condition_types = batch["condition_type"]
        prompts = batch["description"]
        position_delta = batch["position_delta"][0]
        position_scale = float(batch.get("position_scale", [1.0])[0])
        
        # Get additional special vector as condition
        additional_condition1 = batch.get("eeg", None)
        additional_condition2 = batch.get("fnirs", None)

        # Prepare inputs
        with torch.no_grad():
            # Prepare image input
            x_0, img_ids = encode_images(self.flux_pipe, imgs)

            # Prepare text input
            prompt_embeds, pooled_prompt_embeds, text_ids = prepare_text_input(
                self.flux_pipe, prompts
            )

            # Prepare t and x_t
            t = torch.sigmoid(torch.randn((imgs.shape[0],), device=self.device))
            x_1 = torch.randn_like(x_0).to(self.device)
            t_ = t.unsqueeze(1).unsqueeze(1)
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self._dtype)

            # Prepare conditions
            condition_latents, condition_ids = encode_images(self.flux_pipe, conditions)

            # Add position delta
            condition_ids[:, 1] += position_delta[0]
            condition_ids[:, 2] += position_delta[1]

            if position_scale != 1.0:
                scale_bias = (position_scale - 1.0) / 2
                condition_ids[:, 1] *= position_scale
                condition_ids[:, 2] *= position_scale
                condition_ids[:, 1] += scale_bias
                condition_ids[:, 2] += scale_bias

            # Prepare condition type
            condition_type_ids = torch.tensor(
                [
                    Condition.get_type_id(condition_type)
                    for condition_type in condition_types
                ]
            ).to(self.device)
            condition_type_ids = (
                torch.ones_like(condition_ids[:, 0]) * condition_type_ids[0]
            ).unsqueeze(1)

            # Process additional condition vector if provided
            if additional_condition1 is not None:
                # Convert to tensor if not already
                if not isinstance(additional_condition1, torch.Tensor):
                    additional_condition1 = torch.tensor(additional_condition1).to(self.device)
                
                # Ensure it's in the right shape and device
                additional_condition1 = additional_condition1.to(self.device).to(self._dtype)
                
                # Apply spatial pyramid pooling to get fixed length
                additional_condition1 = self.spatial_pyramid_pooling(additional_condition1, self.eeg_fixed_length)
            
            if additional_condition2 is not None:
                # Convert to tensor if not already
                if not isinstance(additional_condition2, torch.Tensor):
                    additional_condition2 = torch.tensor(additional_condition2).to(self.device)
                
                # Ensure it's in the right shape and device
                additional_condition2 = additional_condition2.to(self.device).to(self._dtype)
                
                # Apply spatial pyramid pooling to get fixed length
                additional_condition2 = self.spatial_pyramid_pooling(additional_condition2, self.fnirs_fixed_length)

            # Prepare guidance
            guidance = (
                torch.ones_like(t).to(self.device)
                if self.transformer.config.guidance_embeds
                else None
            )
        
        if self.use_brain_condition and additional_condition1 is not None:
            # Project EEG data using the lightweight projection
            # Flatten the tensor for the linear layer
            flattened_eeg = additional_condition1.flatten(1, 2)
            projected_eeg = self.eeg_projection(flattened_eeg)
            
            if self.fuse_flag:
                # Use attention mechanism to fuse the embeddings
                # prompt_embeds shape: [1, 512, 4096]
                # projected_condition shape: [1, 512, 4096]
                
                # Create attention mask to focus on the condition
                fused_embeds, _ = self.fusion_attention1(
                    prompt_embeds,
                    projected_eeg,
                    projected_eeg
                )
                prompt_embeds = prompt_embeds + fused_embeds
            else:
                prompt_embeds = projected_eeg

        if self.use_brain_condition and additional_condition2 is not None:
            # Project fNIRS data using the lightweight projection
            # Flatten the tensor for the linear layer
            flattened_fnirs = additional_condition2.flatten(1, 2)
            projected_fnirs = self.fnirs_projection(flattened_fnirs)

            if self.fuse_flag:
                # Use attention mechanism to fuse the embeddings
                # prompt_embeds shape: [1, 512, 4096]
                # projected_condition shape: [1, 4, 4096]
                
                # Create attention mask to focus on the condition
                fused_embeds, _ = self.fusion_attention2(
                    pooled_prompt_embeds,
                    projected_fnirs,
                    projected_fnirs
                )
                pooled_prompt_embeds = pooled_prompt_embeds + fused_embeds
            else:
                pooled_prompt_embeds = projected_fnirs

        # Forward pass
        transformer_out = tranformer_forward(
            self.transformer,
            # Model config
            model_config=self.model_config,
            # Inputs of the condition (new feature)
            condition_latents=condition_latents,
            condition_ids=condition_ids,
            condition_type_ids=condition_type_ids,
            # Inputs to the original transformer
            hidden_states=x_t,
            timestep=t,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )
        pred = transformer_out[0]

        # Compute loss
        loss = torch.nn.functional.mse_loss(pred, (x_1 - x_0), reduction="mean")
        self.last_t = t.mean().item()
        return loss
