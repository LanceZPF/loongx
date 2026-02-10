import lightning as L
from diffusers.pipelines import FluxPipeline
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model_state_dict
import torch.nn.functional as F

import prodigyopt

from ..flux.transformer import tranformer_forward
from ..flux.condition import Condition
from ..flux.pipeline_tools import encode_images, prepare_text_input

from s4torch import S4Model

class EEGEncoder(nn.Module):
    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        self.eeg_fixed_length = 4096
        
        # S4 models for sequence modeling
        d_input1 = 4
        d_model1 = 64
        n_blocks1 = 2
        l_max1 = self.eeg_fixed_length
        self.s41 = S4Model(
            d_input1,
            d_model=d_model1,
            d_output=d_model1,
            n_blocks=n_blocks1,
            n=d_model1,
            l_max=l_max1 or 100
        ).to(device)

        self.pool1 = nn.AdaptiveAvgPool1d(4).to(device).to(dtype)

        d_input2 = 4
        d_model2 = 4
        n_blocks2 = 2
        l_max2 = self.eeg_fixed_length
        self.s42 = S4Model(
            d_input2,
            d_model=d_model2,
            d_output=d_model2,
            n_blocks=n_blocks2,
            n=d_model2,
            l_max=l_max2 or 100
        ).to(device)

        self.pool2 = nn.AdaptiveAvgPool1d(64).to(device).to(dtype)
        
        # Feature Pyramid Pooling for multi-scale feature extraction
        self.fpp = FeaturePyramidPooling(output_sizes=[128, 256, 512, 1024, 2048]).to(device).to(dtype)
        
        self.projection = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(4 * 4096, 2048),
            torch.nn.LayerNorm(2048),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(2048, 4096),
            torch.nn.LayerNorm(4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Unflatten(1, (512, 8)),
            torch.nn.Linear(8, 4096)
        ).to(device).to(dtype)

    def forward(self, x):
        # Apply S4 models and pooling
        z1 = x.permute(0, 2, 1).contiguous()
        # Convert to float32 for S4Model
        # z1_float32 = z1.to(torch.float32)
        z1 = self.s41(z1)
        # Convert back to original dtype
        # z1 = z1.to(x.dtype)
        z1 = z1.permute(0, 2, 1).contiguous()
        z1 = self.pool1(z1)
        z1 = z1.permute(0, 2, 1).contiguous()

        z2 = x.permute(0, 2, 1).contiguous()
        # Convert to float32 for S4Model
        # z2_float32 = z2.to(torch.float32)
        z2 = self.s42(z2)
        # Convert back to original dtype
        # z2 = z2.to(x.dtype)
        z2 = z2.permute(0, 2, 1).contiguous()
        z2 = self.pool2(z2)

        # Apply feature pyramid pooling
        x_fpp = self.fpp(x)
        
        # Concatenate features
        x_combined = torch.cat([z1, x_fpp, z2], dim=-1)

        # Project to final representation
        x_out = self.projection(x_combined)

        '''
        # Apply S4 models and pooling
        z1 = x.permute(0, 2, 1).contiguous()
        # Convert to float32 for S4Model
        # z1_float32 = z1.to(torch.float32)
        z1 = self.s41(z1)
        # Convert back to original dtype
        # z1 = z1.to(x.dtype)
        z1 = z1.permute(0, 2, 1).contiguous()
        z1 = self.pool1(z1)
        z1 = z1.permute(0, 2, 1).contiguous()

        z2 = x.permute(0, 2, 1).contiguous()
        # Convert to float32 for S4Model
        # z2_float32 = z2.to(torch.float32)
        z2 = self.s42(z2)
        # Convert back to original dtype
        # z2 = z2.to(x.dtype)
        z2 = z2.permute(0, 2, 1).contiguous()
        z2 = self.pool2(z2)

        # Apply feature pyramid pooling
        x_fpp = self.fpp(x)
        
        # Concatenate features
        x_combined = torch.cat([z1, x_fpp, z2], dim=-1)

        # Project to final representation
        x_out = self.projection(x_combined)
        '''
        return x_out


class PPGEncoder(nn.Module):
    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        
        self.ppg_fixed_length = 256
        
        # S4 model for PPG signals
        d_input = 4
        d_model = 4
        n_blocks = 2
        l_max = self.ppg_fixed_length
        
        self.s4 = S4Model(
            d_input,
            d_model=d_model,
            d_output=d_model,
            n_blocks=n_blocks,
            n=d_model,
            l_max=l_max or 100
        ).to(device)  # Set S4Model to float32
        
        self.pool = nn.AdaptiveAvgPool1d(16).to(device).to(dtype)
        
        # Feature pyramid pooling for multi-scale features
        self.fpp = FeaturePyramidPooling(output_sizes=[64, 128, 256]).to(device).to(dtype)
        
        # Projection layers
        self.projection = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(d_model * 16 + 448 * 4, 1024),  # 448 = 64+128+256 from FPP
            torch.nn.LayerNorm(1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(1024, 4096),
            torch.nn.LayerNorm(4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Unflatten(1, (512, 8)),
            torch.nn.Linear(8, 4096)
        ).to(device).to(dtype)
        
    def forward(self, x):
        # Apply S4 model
        z = x.permute(0, 2, 1).contiguous()
        # Convert to float32 for S4Model
        # z_float32 = z.to(torch.float32)
        z = self.s4(z)
        # Convert back to original dtype
        # z = z.to(x.dtype)
        z = z.permute(0, 2, 1).contiguous()
        z = self.pool(z)
        
        # Apply feature pyramid pooling
        x_fpp = self.fpp(x)
        
        # Flatten and concatenate
        z_flat = z.flatten(1)
        x_fpp_flat = x_fpp.flatten(1)
        
        # Concatenate features
        combined = torch.cat([z_flat, x_fpp_flat], dim=1)
        
        # Project to final representation
        output = self.projection(combined)
        return output


class FNIRSEncoder(nn.Module):
    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        
        self.fnirs_fixed_length = 512
        
        # S4 model for fNIRS signals
        d_input = 6  # Typically fNIRS has 2 channels (HbO and HbR)
        d_model = 6
        n_blocks = 2
        l_max = self.fnirs_fixed_length
        
        self.s4 = S4Model(
            d_input,
            d_model=d_model,
            d_output=d_model,
            n_blocks=n_blocks,
            n=d_model,
            l_max=l_max or 100
        ).to(device)  # Set S4Model to float32
        
        self.pool = nn.AdaptiveAvgPool1d(32).to(device).to(dtype)
        
        # Feature pyramid pooling
        self.fpp = FeaturePyramidPooling(output_sizes=[128, 256, 448]).to(device).to(dtype)
        
        # Projection layers
        self.projection = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(d_model * 32 + 832 * 6, 1024),  # 768 = 128+256+384 from FPP
            torch.nn.LayerNorm(1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(1024, 768),
            torch.nn.LayerNorm(768),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
        ).to(device).to(dtype)
        
    def forward(self, x):
        # Apply S4 model
        z = x.permute(0, 2, 1).contiguous()
        # Convert to float32 for S4Model
        # z_float32 = z.to(torch.float32)
        z = self.s4(z)
        # Convert back to original dtype
        # z = z.to(x.dtype)
        z = z.permute(0, 2, 1).contiguous()
        z = self.pool(z)
        
        # Apply feature pyramid pooling
        x_fpp = self.fpp(x)
        
        # Flatten and concatenate
        z_flat = z.flatten(1)
        x_fpp_flat = x_fpp.flatten(1)
        
        # Concatenate features
        combined = torch.cat([z_flat, x_fpp_flat], dim=1)
        
        # Project to final representation
        output = self.projection(combined)
        return output


class MotionEncoder(nn.Module):
    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        
        self.motion_fixed_length = 128
        
        # S4 model for motion signals (typically 6 channels: 3 for accelerometer, 3 for gyroscope)
        d_input = 6
        d_model = 6
        n_blocks = 2
        l_max = self.motion_fixed_length
        
        self.s4 = S4Model(
            d_input,
            d_model=d_model,
            d_output=d_model,
            n_blocks=n_blocks,
            n=d_model,
            l_max=l_max or 100
        ).to(device)  # Set S4Model to float32
        
        self.pool = nn.AdaptiveAvgPool1d(6).to(device).to(dtype)
        
        # Feature pyramid pooling
        self.fpp = FeaturePyramidPooling(output_sizes=[32, 64, 124]).to(device).to(dtype)
        
        # Projection layers
        self.projection = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(d_model * 6 + 220 * 6, 512),  # 192 = 32+64+96 from FPP
            torch.nn.LayerNorm(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 768),
            torch.nn.LayerNorm(768),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
        ).to(device).to(dtype)

    def forward(self, x):
        # Apply S4 model
        z = x.permute(0, 2, 1).contiguous()
        # Convert to float32 for S4Model
        # z_float32 = z.to(torch.float32)
        z = self.s4(z)
        # Convert back to original dtype
        # z = z.to(x.dtype)
        z = z.permute(0, 2, 1).contiguous()
        z = self.pool(z)
        
        # Apply feature pyramid pooling
        x_fpp = self.fpp(x)
        
        # Flatten and concatenate
        z_flat = z.flatten(1)
        x_fpp_flat = x_fpp.flatten(1)
        
        # Concatenate features
        combined = torch.cat([z_flat, x_fpp_flat], dim=1)
        
        # Project to final representation
        output = self.projection(combined)
        return output

class FeaturePyramidPooling(nn.Module):
    """
    Feature Pyramid Pooling module for biosignals.
    This module applies multiple adaptive average pooling operations with different output sizes
    to create a multi-scale feature representation.
    """
    def __init__(self, output_sizes=[256, 512, 1024]):
        super(FeaturePyramidPooling, self).__init__()
        self.output_sizes = output_sizes
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool1d(size) for size in output_sizes
        ])
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, C, L] where B is batch size, 
               C is number of channels, and L is sequence length
        
        Returns:
            Concatenated multi-scale features of shape [B, C, sum(output_sizes)]
        """
        # Apply different pooling operations
        features = []
        for pool in self.pools:
            features.append(pool(x))
        
        # Concatenate along the sequence dimension
        return torch.cat(features, dim=-1)


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
        fuse_flag: bool = True,
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

        self.fuse_flag = fuse_flag
        self.use_brain_condition = use_brain_condition
        
        # Target fixed lengths after spatial pyramid pooling
        self.eeg_fixed_length = 4096
        self.fnirs_fixed_length = 512
        self.ppg_fixed_length = 256
        self.motion_fixed_length = 128

        # Fusion layers for EEG and PPG
        # self.fusion1 = torch.nn.Sequential(
        #     torch.nn.Linear(4096 + 768, 4096),
        #     torch.nn.LayerNorm(4096),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(0.3),
        # ).to(device).to(dtype)

        self.fusion1 = torch.nn.Sequential(
            torch.nn.Linear(512*2, 512),
        ).to(device).to(dtype)

        # # Fusion layers for FNIRS and Motion
        self.fusion2 = torch.nn.Sequential(
            torch.nn.Linear(768 + 768, 768),
        ).to(device).to(dtype)

        self.duan_norm1 = DUAN(channels=512, device=device, dtype=dtype)
        self.duan_norm2 = DUAN(channels=1, device=device, dtype=dtype)
        
        self.fusion3 = torch.nn.Sequential(
            torch.nn.Linear(512*2, 512),
        ).to(device).to(dtype)
        
        self.fusion4 = torch.nn.Sequential(
            torch.nn.Linear(768*2, 768),
        ).to(device).to(dtype)
        
        # Add DUAN for prompt_embeds fusion
        self.duan_norm_prompt = DUAN(channels=512, device=device, dtype=dtype)
        
        # Add DUAN for pooled_prompt_embeds fusion
        self.duan_norm_pooled = DUAN(channels=1, device=device, dtype=dtype)

        self.to(device).to(dtype)
                
        # Initialize encoders for different biosignals
        self.eeg_projection = EEGEncoder(device=device, dtype=dtype)
        self.ppg_projection = PPGEncoder(device=device, dtype=dtype)
        self.fnirs_projection = FNIRSEncoder(device=device, dtype=dtype)
        self.motion_projection = MotionEncoder(device=device, dtype=dtype)

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

    def spatial_pyramid_pooling(self, x, output_size, adaptive=False):
        """
        Apply spatial pyramid pooling to convert variable length input to fixed length
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, variable_length]
            output_size (int): Desired fixed length after pooling
            adaptive (bool): Whether to use adaptive pooling or padding/truncation
            
        Returns:
            torch.Tensor: Tensor of shape [batch_size, channels, output_size]
        """
        batch_size, channels, length = x.shape
        
        # If input length is already the desired length, return as is
        if length == output_size:
            return x
        
        if adaptive:
            # Calculate adaptive pool sizes to achieve the desired output size
            # We'll use a simple approach with equal-sized bins
            result = F.adaptive_avg_pool1d(x, output_size)
        else:
            # Use padding approach to handle different input lengths
            if length < output_size:
                # Pad with zeros if input is shorter than desired output
                padding = torch.zeros(batch_size, channels, output_size - length, device=x.device, dtype=x.dtype)
                result = torch.cat([x, padding], dim=2)
            else:
                # Truncate if input is longer than desired output
                result = x[:, :, :output_size]
            
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
                eeg_features = self.eeg_projection(eeg)
                
                if ppg is not None:
                    ppg_features = self.ppg_projection(ppg)
                    # Use fusion function to combine EEG and PPG features
                    prompt_embeds_brain = self.fuse_eeg(eeg_features, ppg_features)
                else:
                    prompt_embeds_brain = eeg_features
            
            # Process and fuse FNIRS and Motion for pooled_prompt_embeds
            if fnirs is not None:
                fnirs_features = self.fnirs_projection(fnirs)
                
                if motion is not None:
                    motion_features = self.motion_projection(motion)
                    # Use fusion function to combine FNIRS and Motion features
                    pooled_prompt_embeds_brain = self.fuse_fnirs(fnirs_features, motion_features)
                else:
                    pooled_prompt_embeds_brain = fnirs_features
            
            # Fuse original embeddings with brain embeddings when fuse_flag is True
            if self.fuse_flag:
                # Fuse prompt_embeds with prompt_embeds_brain
                prompt_embeds_fused = self.duan_norm_prompt(prompt_embeds_brain, prompt_embeds)
                prompt_embeds_cat = torch.cat([prompt_embeds, prompt_embeds_fused], dim=1)
                prompt_embeds_cat = prompt_embeds_cat.transpose(1, 2).contiguous()
                prompt_embeds_cat =  self.fusion3(prompt_embeds_cat)
                prompt_embeds = prompt_embeds + prompt_embeds_cat.transpose(1, 2).contiguous()
                
                # Fuse pooled_prompt_embeds with pooled_prompt_embeds_brain
                # Reshape for DUAN which expects [B,C,L] format
                pooled_prompt_embeds_reshaped = pooled_prompt_embeds.unsqueeze(1)  # [B,1,D]
                pooled_prompt_embeds_brain_reshaped = pooled_prompt_embeds_brain.unsqueeze(1)  # [B,1,D]
                
                fused_pooled = self.duan_norm_pooled(
                    pooled_prompt_embeds_brain_reshaped,
                    pooled_prompt_embeds_reshaped
                )
                pooled_prompt_embeds_cat = torch.cat([pooled_prompt_embeds, fused_pooled.squeeze(1)], dim=-1)  # Back to [B,D]
                pooled_prompt_embeds = pooled_prompt_embeds + self.fusion4(pooled_prompt_embeds_cat)
            else:
                prompt_embeds = prompt_embeds_brain
                pooled_prompt_embeds = pooled_prompt_embeds_brain
        

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
        # combined = torch.cat([eeg_features, ppg_features], dim=-1)  
        # fused_features = self.fusion1(combined)
        # fused_features += eeg_features
        # ppg_features = ppg_features.transpose(1, 2)
        # eeg_features = eeg_features.transpose(1, 2)

        fused_features = self.duan_norm1(ppg_features, eeg_features)
        fused_features = torch.cat([eeg_features, fused_features], dim=1)
        fused_features = fused_features.transpose(1, 2).contiguous()
        fused_features = self.fusion1(fused_features)
        fused_features = fused_features.transpose(1, 2).contiguous()
        
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
        # combined = torch.cat([fnirs_features, motion_features], dim=-1)
        
        # fused_features = self.fusion2(combined)
        fnirs_features = fnirs_features.unsqueeze(1)
        motion_features = motion_features.unsqueeze(1)
        fused_features = self.duan_norm2(fnirs_features, motion_features)
        fused_features = torch.cat([fnirs_features, fused_features], dim=-1)
        fused_features = self.fusion2(fused_features)
        fused_features = fused_features.squeeze(1)
        # fused_features = fnirs_features
        return fused_features
    def save_custom_weights(self, save_path):
        """
        Save only the custom weights (non-Flux weights) to a specified path.
        This includes LoRA weights, encoders, DUAN modules, and fusion layers.
        
        Args:
            save_path (str): Directory path to save the weights
        """
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Create a state dict with only the custom components
        custom_state_dict = {}
        
        # Save encoder weights if they exist
        if hasattr(self, 'eeg_encoder'):
            custom_state_dict['eeg_encoder'] = self.eeg_encoder.state_dict()
        
        if hasattr(self, 'ppg_encoder'):
            custom_state_dict['ppg_encoder'] = self.ppg_encoder.state_dict()
            
        if hasattr(self, 'fnirs_encoder'):
            custom_state_dict['fnirs_encoder'] = self.fnirs_encoder.state_dict()
            
        if hasattr(self, 'motion_encoder'):
            custom_state_dict['motion_encoder'] = self.motion_encoder.state_dict()
        
        # Save DUAN modules
        if hasattr(self, 'duan_norm1'):
            custom_state_dict['duan_norm1'] = self.duan_norm1.state_dict()
            
        if hasattr(self, 'duan_norm2'):
            custom_state_dict['duan_norm2'] = self.duan_norm2.state_dict()
            
        if hasattr(self, 'duan_norm_prompt'):
            custom_state_dict['duan_norm_prompt'] = self.duan_norm_prompt.state_dict()
            
        if hasattr(self, 'duan_norm_pooled'):
            custom_state_dict['duan_norm_pooled'] = self.duan_norm_pooled.state_dict()
        
        # Save fusion layers
        if hasattr(self, 'fusion1'):
            custom_state_dict['fusion1'] = self.fusion1.state_dict()
            
        if hasattr(self, 'fusion2'):
            custom_state_dict['fusion2'] = self.fusion2.state_dict()
            
        if hasattr(self, 'fusion3'):
            custom_state_dict['fusion3'] = self.fusion3.state_dict()
            
        if hasattr(self, 'fusion4'):
            custom_state_dict['fusion4'] = self.fusion4.state_dict()
        
        # Save projection layers
        if hasattr(self, 'eeg_projection'):
            custom_state_dict['eeg_projection'] = self.eeg_projection.state_dict()
            
        if hasattr(self, 'ppg_projection'):
            custom_state_dict['ppg_projection'] = self.ppg_projection.state_dict()
            
        if hasattr(self, 'fnirs_projection'):
            custom_state_dict['fnirs_projection'] = self.fnirs_projection.state_dict()
            
        if hasattr(self, 'motion_projection'):
            custom_state_dict['motion_projection'] = self.motion_projection.state_dict()
        
        # Save LoRA weights if they exist
        if hasattr(self, 'transformer') and hasattr(self.transformer, 'get_lora_state_dict'):
            lora_state_dict = self.transformer.get_lora_state_dict()
            if lora_state_dict:
                custom_state_dict['lora'] = lora_state_dict
        
        # Save model configuration for proper loading
        custom_state_dict['model_config'] = {
            'use_brain_condition': self.use_brain_condition,
            'fuse_flag': self.fuse_flag,
            'dtype': str(self._dtype)
        }
        
        # Save the custom state dict
        torch.save(custom_state_dict, os.path.join(save_path, 'custom_weights.pt'))
        print(f"Custom weights saved to {os.path.join(save_path, 'custom_weights.pt')}")
    
    def load_custom_weights(self, load_path):
        """
        Load custom weights (non-Flux weights) from a specified path.
        
        Args:
            load_path (str): Path to the saved weights file
        """
        import os
        weights_path = os.path.join(load_path, 'custom_weights.pt') if os.path.isdir(load_path) else load_path
        
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found at {weights_path}")
        
        # Load the state dict
        custom_state_dict = torch.load(weights_path, map_location=self.device)
        
        # Load model configuration if available
        if 'model_config' in custom_state_dict:
            config = custom_state_dict['model_config']
            print(f"Loading model with configuration: {config}")
        
        # Load encoder weights if they exist
        if 'eeg_encoder' in custom_state_dict and hasattr(self, 'eeg_encoder'):
            self.eeg_encoder.load_state_dict(custom_state_dict['eeg_encoder'])
            
        if 'ppg_encoder' in custom_state_dict and hasattr(self, 'ppg_encoder'):
            self.ppg_encoder.load_state_dict(custom_state_dict['ppg_encoder'])
            
        if 'fnirs_encoder' in custom_state_dict and hasattr(self, 'fnirs_encoder'):
            self.fnirs_encoder.load_state_dict(custom_state_dict['fnirs_encoder'])
            
        if 'motion_encoder' in custom_state_dict and hasattr(self, 'motion_encoder'):
            self.motion_encoder.load_state_dict(custom_state_dict['motion_encoder'])
        
        # Load DUAN modules
        if 'duan_norm1' in custom_state_dict and hasattr(self, 'duan_norm1'):
            self.duan_norm1.load_state_dict(custom_state_dict['duan_norm1'])
            
        if 'duan_norm2' in custom_state_dict and hasattr(self, 'duan_norm2'):
            self.duan_norm2.load_state_dict(custom_state_dict['duan_norm2'])
            
        if 'duan_norm_prompt' in custom_state_dict and hasattr(self, 'duan_norm_prompt'):
            self.duan_norm_prompt.load_state_dict(custom_state_dict['duan_norm_prompt'])
            
        if 'duan_norm_pooled' in custom_state_dict and hasattr(self, 'duan_norm_pooled'):
            self.duan_norm_pooled.load_state_dict(custom_state_dict['duan_norm_pooled'])
        
        # Load fusion layers
        if 'fusion1' in custom_state_dict and hasattr(self, 'fusion1'):
            self.fusion1.load_state_dict(custom_state_dict['fusion1'])
            
        if 'fusion2' in custom_state_dict and hasattr(self, 'fusion2'):
            self.fusion2.load_state_dict(custom_state_dict['fusion2'])
            
        if 'fusion3' in custom_state_dict and hasattr(self, 'fusion3'):
            self.fusion3.load_state_dict(custom_state_dict['fusion3'])
            
        if 'fusion4' in custom_state_dict and hasattr(self, 'fusion4'):
            self.fusion4.load_state_dict(custom_state_dict['fusion4'])
        
        # Load projection layers
        if 'eeg_projection' in custom_state_dict and hasattr(self, 'eeg_projection'):
            self.eeg_projection.load_state_dict(custom_state_dict['eeg_projection'])
            
        if 'ppg_projection' in custom_state_dict and hasattr(self, 'ppg_projection'):
            self.ppg_projection.load_state_dict(custom_state_dict['ppg_projection'])
            
        if 'fnirs_projection' in custom_state_dict and hasattr(self, 'fnirs_projection'):
            self.fnirs_projection.load_state_dict(custom_state_dict['fnirs_projection'])
            
        if 'motion_projection' in custom_state_dict and hasattr(self, 'motion_projection'):
            self.motion_projection.load_state_dict(custom_state_dict['motion_projection'])
        
        # Load LoRA weights if they exist
        if 'lora' in custom_state_dict and hasattr(self, 'transformer'):
            if hasattr(self.transformer, 'load_lora_state_dict'):
                self.transformer.load_lora_state_dict(custom_state_dict['lora'])
            else:
                print("Warning: Transformer does not have load_lora_state_dict method")
        
        print(f"Custom weights loaded successfully from {weights_path}")



class DUAN(nn.Module):
    """
    Dynamic Unified Adaptive Normalization
    Args:
        channels (int):   C in [B,C,L]
        hidden_dim (int): 隐藏维度，用于产生门控与 γ/β
        keep_ratio (float):0~1，每个样本保留的通道比例
        eps (float):      数值稳定项
    """
    def __init__(
        self,
        channels: int,
        hidden_dim: int = 128,
        keep_ratio: float = 0.7,
        eps: float = 1e-3,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_dim = hidden_dim
        self.keep_ratio = keep_ratio
        self.eps = eps

        # 门控网络：c ─► g_mix ∈ [0,1]^(B,C,1)
        self.gate = nn.Sequential(
            nn.Conv1d(channels, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, channels, 1),
            nn.Sigmoid(),
        )

        # γ / β 生成器：c ─► [γ,β] ∈ ℝ^(B,2C,1)
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, channels * 2, 1),
        )

        if device is not None:
            self.to(device=device, dtype=dtype)

    def forward(
        self,
        x16: torch.Tensor,  # [B,C,L] 内容特征
        c16: torch.Tensor,  # [B,C,L] 条件特征
        keep_ratio: float | None = None,
    ) -> torch.Tensor:
        x, c = x16.float(), c16.float()

        assert x.shape == c.shape, "x, c must have identical shape [B,C,L]"
        B, C, L = x.shape
        if keep_ratio is None:
            keep_ratio = self.keep_ratio

        # ---------- 1. 统计 ----------
        # Instance-level（每通道）
        mu_c = x.mean(dim=2, keepdim=True)                         # [B,C,1]
        var_c = x.var(dim=2, unbiased=False, keepdim=True)
        sigma_c = torch.sqrt(var_c + self.eps)

        # Layer-level（整层）
        mu_l = x.mean(dim=(1, 2), keepdim=True).expand(B, C, 1)    # [B,C,1]
        var_l = x.var(dim=(1, 2), unbiased=False, keepdim=True).expand(B, C, 1)
        sigma_l = torch.sqrt(var_l + self.eps)

        # ---------- 2. 门控融合 ----------
        # g_mix 取 c 经过 GAP 后的门控（每通道同一权重）
        g_mix = self.gate(c).mean(dim=2, keepdim=True)             # [B,C,1]
        mu    = g_mix * mu_c + (1 - g_mix) * mu_l
        sigma = g_mix * sigma_c + (1 - g_mix) * sigma_l
        x_hat = (x - mu) / sigma

        # ---------- 3. γ / β 调制 ----------
        cond_pool = c.mean(dim=2, keepdim=True)                    # [B,C,1]
        gamma_beta = self.mlp(cond_pool)                           # [B,2C,1]
        gamma, beta = gamma_beta.chunk(2, dim=1)                   # [B,C,1]×2
        y = (1 + gamma) * x_hat + beta                             # [B,C,L]

        # ---------- 4. 动态掩码 ----------
        # importance: 通道平均幅值
        imp = y.abs().mean(dim=2)                                  # [B,C]
        k = max(1, int(C * keep_ratio))
        topk = torch.topk(imp, k, dim=1).indices                   # [B,k]
        mask = torch.zeros_like(imp, dtype=y.dtype, device=y.device)
        mask.scatter_(1, topk, 1.0)                                # [B,C]
        y = y * mask.unsqueeze(2)                                  # [B,C,L]

        return y.to(x16.dtype)