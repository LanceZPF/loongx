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
        self.eeg_fixed_length = 128
        self.fnirs_fixed_length = 16
        self.ppg_fixed_length = 128
        self.motion_fixed_length = 16
        
        # EEG projection
        self.eeg_projection = torch.nn.Sequential(
            torch.nn.Linear(4 * self.eeg_fixed_length, 2048),
            torch.nn.LayerNorm(2048),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(2048, 4096),
            torch.nn.LayerNorm(4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Unflatten(1, (512, 8)),  # Reshape to [1,512,8]
            torch.nn.Linear(8, 4096)  # Final projection to target dimension
        ).to(device).to(dtype)
        
        # PPG projection
        self.ppg_projection = torch.nn.Sequential(
            torch.nn.Linear(4 * self.ppg_fixed_length, 384),
            torch.nn.LayerNorm(384),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(384, 768),
            torch.nn.LayerNorm(768),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Unflatten(1, (512, 8)),  # Reshape to [1,512,8]
            torch.nn.Linear(8, 4096)  # Final projection to target dimension
        ).to(device).to(dtype)
        
        # FNIRS projection
        self.fnirs_projection = torch.nn.Sequential(
            torch.nn.Linear(6 * self.fnirs_fixed_length, 384),
            torch.nn.LayerNorm(384),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(384, 768),
            torch.nn.LayerNorm(768),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
        ).to(device).to(dtype)
        
        # Motion projection
        self.motion_projection = torch.nn.Sequential(
            torch.nn.Linear(6 * self.motion_fixed_length, 384),
            torch.nn.LayerNorm(384),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(384, 768),
            torch.nn.LayerNorm(768),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
        ).to(device).to(dtype)
        
        # Fusion layers for EEG and PPG
        self.fusion1 = torch.nn.Sequential(
            torch.nn.Linear(4096 + 768, 4096),
            torch.nn.LayerNorm(4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
        ).to(device).to(dtype)
        
        # Fusion layers for FNIRS and Motion
        self.fusion2 = torch.nn.Sequential(
            torch.nn.Linear(768 + 768, 768),
            torch.nn.LayerNorm(768),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
        ).to(device).to(dtype)

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
        eeg = batch.get("eeg", None)
        fnirs = batch.get("fnirs", None)
        ppg = batch.get("ppg", None)
        motion = batch.get("motion", None)

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

            # Process brain signals if provided
            if eeg is not None:
                if not isinstance(eeg, torch.Tensor):
                    eeg = torch.tensor(eeg).to(self.device)
                eeg = eeg.to(self.device).to(self._dtype)
                eeg = self.spatial_pyramid_pooling(eeg, self.eeg_fixed_length)
            
            if fnirs is not None:
                if not isinstance(fnirs, torch.Tensor):
                    fnirs = torch.tensor(fnirs).to(self.device)
                fnirs = fnirs.to(self.device).to(self._dtype)
                fnirs = self.spatial_pyramid_pooling(fnirs, self.fnirs_fixed_length)
                
            if ppg is not None:
                if not isinstance(ppg, torch.Tensor):
                    ppg = torch.tensor(ppg).to(self.device)
                ppg = ppg.to(self.device).to(self._dtype)
                ppg = self.spatial_pyramid_pooling(ppg, self.ppg_fixed_length)
                
            if motion is not None:
                if not isinstance(motion, torch.Tensor):
                    motion = torch.tensor(motion).to(self.device)
                motion = motion.to(self.device).to(self._dtype)
                motion = self.spatial_pyramid_pooling(motion, self.motion_fixed_length)

            # Prepare guidance
            guidance = (
                torch.ones_like(t).to(self.device)
                if self.transformer.config.guidance_embeds
                else None
            )
        
        if self.use_brain_condition:
            # Process and fuse EEG and PPG for prompt_embeds
            if eeg is not None:
                eeg_features = self.eeg_projection(eeg.flatten(1))
                
                if ppg is not None:
                    ppg_features = self.ppg_projection(ppg.flatten(1))
                    # Use fusion function to combine EEG and PPG features
                    prompt_embeds = self.fuse_eeg(eeg_features, ppg_features)
                else:
                    prompt_embeds = eeg_features
            
            # Process and fuse FNIRS and Motion for pooled_prompt_embeds
            if fnirs is not None:
                fnirs_features = self.fnirs_projection(fnirs.flatten(1))
                
                if motion is not None:
                    motion_features = self.motion_projection(motion.flatten(1))
                    # Use fusion function to combine FNIRS and Motion features
                    pooled_prompt_embeds = self.fuse_fnirs(fnirs_features, motion_features)
                else:
                    pooled_prompt_embeds = fnirs_features

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
    
    def fuse_eeg(self, eeg_features, ppg_features):
        """
        Fuse EEG and PPG features using a linear layer
        
        Args:
            eeg_features (torch.Tensor): EEG features of shape [batch_size, 4096]
            ppg_features (torch.Tensor): PPG features of shape [batch_size, 768]
            fusion_layer (torch.nn.Module): Fusion layer to combine features
            
        Returns:
            torch.Tensor: Fused features of shape [batch_size, 4096]
        """
        combined = torch.cat([eeg_features, ppg_features], dim=-1)  
        fused_features = self.fusion1(combined)
        return fused_features
    
    def fuse_fnirs(self, fnirs_features, motion_features):
        """
        Fuse FNIRS and Motion features using a linear layer

        Args:
            fnirs_features (torch.Tensor): FNIRS features of shape [batch_size, 768]
            motion_features (torch.Tensor): Motion features of shape [batch_size, 768]
            fusion_layer (torch.nn.Module): Fusion layer to combine features
            
        Returns:
            torch.Tensor: Fused features of shape [batch_size, 768]
        """
        combined = torch.cat([fnirs_features, motion_features], dim=-1)
        fused_features = self.fusion2(combined)
        return fused_features