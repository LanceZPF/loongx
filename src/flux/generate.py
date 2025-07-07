import torch
import yaml, os
from diffusers.pipelines import FluxPipeline
from typing import List, Union, Optional, Dict, Any, Callable
from .transformer import tranformer_forward
from .condition import Condition

from diffusers.pipelines.flux.pipeline_flux import (
    FluxPipelineOutput,
    calculate_shift,
    retrieve_timesteps,
    np,
)


def get_config(config_path: str = None):
    config_path = config_path or os.environ.get("XFL_CONFIG")
    if not config_path:
        return {}
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def prepare_params(
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = 512,
    width: Optional[int] = 512,
    num_inference_steps: int = 28,
    timesteps: List[int] = None,
    guidance_scale: float = 3.5,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
    **kwargs: dict,
):
    return (
        prompt,
        prompt_2,
        height,
        width,
        num_inference_steps,
        timesteps,
        guidance_scale,
        num_images_per_prompt,
        generator,
        latents,
        prompt_embeds,
        pooled_prompt_embeds,
        output_type,
        return_dict,
        joint_attention_kwargs,
        callback_on_step_end,
        callback_on_step_end_tensor_inputs,
        max_sequence_length,
    )


def seed_everything(seed: int = 42):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)
@torch.no_grad()
def generate(
    model,
    pipeline: FluxPipeline,
    conditions: List[Condition] = None,
    config_path: str = None,
    model_config: Optional[Dict[str, Any]] = {},
    condition_scale: float = 1.0,
    default_lora: bool = False,
    additional_condition1: Optional[torch.FloatTensor] = None,  # EEG data
    additional_condition2: Optional[torch.FloatTensor] = None,  # fNIRS data
    additional_condition3: Optional[torch.FloatTensor] = None,  # PPG data
    additional_condition4: Optional[torch.FloatTensor] = None,  # Motion data
    use_brain_condition: bool = True,
    fuse_flag: bool = True,
    **params: dict,
):
    model_config = model_config or get_config(config_path).get("model", {})
    if condition_scale != 1:
        for name, module in pipeline.transformer.named_modules():
            if not name.endswith(".attn"):
                continue
            module.c_factor = torch.ones(1, 1) * condition_scale

    self = pipeline
    (
        prompt,
        prompt_2,
        height,
        width,
        num_inference_steps,
        timesteps,
        guidance_scale,
        num_images_per_prompt,
        generator,
        latents,
        prompt_embeds,
        pooled_prompt_embeds,
        output_type,
        return_dict,
        joint_attention_kwargs,
        callback_on_step_end,
        callback_on_step_end_tensor_inputs,
        max_sequence_length,
    ) = prepare_params(**params)

    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    lora_scale = (
        self.joint_attention_kwargs.get("scale", None)
        if self.joint_attention_kwargs is not None
        else None
    )
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )
    
    # Process additional brain conditions if provided
    if use_brain_condition:
        # Process EEG data
        if additional_condition1 is not None:
            additional_condition1 = additional_condition1.unsqueeze(0)
            if not isinstance(additional_condition1, torch.Tensor):
                additional_condition1 = torch.tensor(additional_condition1).to(device)
            
            eeg = additional_condition1.to(device).to(prompt_embeds.dtype)
            eeg = model.spatial_pyramid_pooling(eeg, model.eeg_fixed_length)
        else:
            eeg = None
            
        # Process fNIRS data
        if additional_condition2 is not None:
            additional_condition2 = additional_condition2.unsqueeze(0)
            if not isinstance(additional_condition2, torch.Tensor):
                additional_condition2 = torch.tensor(additional_condition2).to(device)
            
            fnirs = additional_condition2.to(device).to(pooled_prompt_embeds.dtype)
            fnirs = model.spatial_pyramid_pooling(fnirs, model.fnirs_fixed_length)
        else:
            fnirs = None
            
        # Process PPG data
        if additional_condition3 is not None:
            additional_condition3 = additional_condition3.unsqueeze(0)
            if not isinstance(additional_condition3, torch.Tensor):
                additional_condition3 = torch.tensor(additional_condition3).to(device)
            
            ppg = additional_condition3.to(device).to(prompt_embeds.dtype)
            ppg = model.spatial_pyramid_pooling(ppg, model.ppg_fixed_length)
        else:
            ppg = None
            
        # Process Motion data
        if additional_condition4 is not None:
            additional_condition4 = additional_condition4.unsqueeze(0)
            if not isinstance(additional_condition4, torch.Tensor):
                additional_condition4 = torch.tensor(additional_condition4).to(device)
            
            motion = additional_condition4.to(device).to(pooled_prompt_embeds.dtype)
            motion = model.spatial_pyramid_pooling(motion, model.motion_fixed_length)
        else:
            motion = None
            
        # Process and fuse EEG and PPG for prompt_embeds
        if eeg is not None:
            eeg_features = model.eeg_projection(eeg.flatten(1))
            
            if ppg is not None:
                ppg_features = model.ppg_projection(ppg.flatten(1))
                # Use fusion function to combine EEG and PPG features
                prompt_embeds_brain = model.fuse_eeg(eeg_features, ppg_features)
            else:
                prompt_embeds_brain = eeg_features
        else:
            prompt_embeds_brain = None
        
        # Process and fuse FNIRS and Motion for pooled_prompt_embeds
        if fnirs is not None:
            fnirs_features = model.fnirs_projection(fnirs.flatten(1))
            
            if motion is not None:
                motion_features = model.motion_projection(motion.flatten(1))
                # Use fusion function to combine FNIRS and Motion features
                pooled_prompt_embeds_brain = model.fuse_fnirs(fnirs_features, motion_features)
            else:
                pooled_prompt_embeds_brain = fnirs_features
        else:
            pooled_prompt_embeds_brain = None
        
        # Fuse original embeddings with brain embeddings when fuse_flag is True
        if fuse_flag and prompt_embeds_brain is not None and pooled_prompt_embeds_brain is not None:
            # Fuse prompt_embeds with prompt_embeds_brain
            prompt_embeds_fused = model.duan_norm_prompt(prompt_embeds, prompt_embeds_brain)
            
            # Fuse pooled_prompt_embeds with pooled_prompt_embeds_brain
            # Reshape for DUAN which expects [B,C,L] format
            pooled_prompt_embeds_reshaped = pooled_prompt_embeds.unsqueeze(1)  # [B,1,D]
            pooled_prompt_embeds_brain_reshaped = pooled_prompt_embeds_brain.unsqueeze(1)  # [B,1,D]
            
            fused_pooled = model.duan_norm_pooled(
                pooled_prompt_embeds_reshaped, 
                pooled_prompt_embeds_brain_reshaped
            )
            
            prompt_embeds = prompt_embeds_fused
            pooled_prompt_embeds = fused_pooled.squeeze(1)
        elif prompt_embeds_brain is not None and pooled_prompt_embeds_brain is not None:
            prompt_embeds = prompt_embeds_brain
            pooled_prompt_embeds = pooled_prompt_embeds_brain

    # 4. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels // 4
    latents, latent_image_ids = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 4.1. Prepare conditions
    condition_latents, condition_ids, condition_type_ids = ([] for _ in range(3))
    use_condition = conditions is not None or []
    if use_condition:
        assert len(conditions) <= 1, "Only one condition is supported for now."
        if not default_lora:
            pipeline.set_adapters(conditions[0].condition_type)
        for condition in conditions:
            tokens, ids, type_id = condition.encode(self)
            condition_latents.append(tokens)  # [batch_size, token_n, token_dim]
            condition_ids.append(ids)  # [token_n, id_dim(3)]
            condition_type_ids.append(type_id)  # [token_n, 1]
        condition_latents = torch.cat(condition_latents, dim=1)
        condition_ids = torch.cat(condition_ids, dim=0)
        condition_type_ids = torch.cat(condition_type_ids, dim=0)

    # 5. Prepare timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.base_image_seq_len,
        self.scheduler.config.max_image_seq_len,
        self.scheduler.config.base_shift,
        self.scheduler.config.max_shift,
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        timesteps,
        sigmas,
        mu=mu,
    )
    num_warmup_steps = max(
        len(timesteps) - num_inference_steps * self.scheduler.order, 0
    )
    self._num_timesteps = len(timesteps)

    # 6. Denoising loop
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            # handle guidance
            if self.transformer.config.guidance_embeds:
                guidance = torch.tensor([guidance_scale], device=device)
                guidance = guidance.expand(latents.shape[0])
            else:
                guidance = None
            noise_pred = tranformer_forward(
                self.transformer,
                model_config=model_config,
                # Inputs of the condition (new feature)
                condition_latents=condition_latents if use_condition else None,
                condition_ids=condition_ids if use_condition else None,
                condition_type_ids=condition_type_ids if use_condition else None,
                # Inputs to the original transformer
                hidden_states=latents,
                # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
                timestep=timestep / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]

            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            # call the callback, if provided
            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()

    if output_type == "latent":
        image = latents

    else:
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (
            latents / self.vae.config.scaling_factor
        ) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    if condition_scale != 1:
        for name, module in pipeline.transformer.named_modules():
            if not name.endswith(".attn"):
                continue
            del module.c_factor

    if not return_dict:
        return (image,)

    return FluxPipelineOutput(images=image)
