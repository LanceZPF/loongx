import lightning as L
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
from transformers import pipeline
import cv2
import torch
import os

try:
    import wandb
except ImportError:
    wandb = None

from ..flux.condition import Condition
from ..flux.generate import generate
import json
import pickle

class TrainingCallback(L.Callback):
    def __init__(self, run_name, training_config: dict = {}):
        self.run_name, self.training_config = run_name, training_config

        self.print_every_n_steps = training_config.get("print_every_n_steps", 10)
        self.save_interval = training_config.get("save_interval", 1000)
        self.sample_interval = training_config.get("sample_interval", 1000)
        self.save_path = training_config.get("save_path", "./output")

        self.wandb_config = training_config.get("wandb", None)
        self.use_wandb = (
            wandb is not None and os.environ.get("WANDB_API_KEY") is not None
        )

        self.total_steps = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        gradient_size = 0
        max_gradient_size = 0
        count = 0
        for _, param in pl_module.named_parameters():
            if param.grad is not None:
                gradient_size += param.grad.norm(2).item()
                max_gradient_size = max(max_gradient_size, param.grad.norm(2).item())
                count += 1
        if count > 0:
            gradient_size /= count

        self.total_steps += 1

        # Print training progress every n steps
        if self.use_wandb:
            report_dict = {
                "steps": batch_idx,
                "steps": self.total_steps,
                "epoch": trainer.current_epoch,
                "gradient_size": gradient_size,
            }
            loss_value = outputs["loss"].item() * trainer.accumulate_grad_batches
            report_dict["loss"] = loss_value
            report_dict["t"] = pl_module.last_t
            wandb.log(report_dict)

        if self.total_steps % self.print_every_n_steps == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps}, Batch: {batch_idx}, Loss: {pl_module.log_loss:.4f}, Gradient size: {gradient_size:.4f}, Max gradient size: {max_gradient_size:.4f}"
            )

        # Save LoRA weights at specified intervals
        if self.total_steps % self.save_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Saving LoRA weights"
            )
            pl_module.save_lora(
                f"{self.save_path}/{self.run_name}/ckpt/{self.total_steps}"
            )

        # Generate and save a sample image at specified intervals
        if self.total_steps % self.sample_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Generating a sample"
            )
            self.generate_a_sample(
                trainer,
                pl_module,
                f"{self.save_path}/{self.run_name}/output",
                f"lora_{self.total_steps}",
                batch["condition_type"][
                    0
                ],  # Use the condition type from the current batch
            )

    @torch.no_grad()
    def generate_a_sample(
        self,
        trainer,
        pl_module,
        save_path,
        file_name,
        condition_type="super_resolution",
    ):
        # TODO: change this two variables to parameters
        condition_size = trainer.training_config["dataset"]["condition_size"]
        target_size = trainer.training_config["dataset"]["target_size"]
        position_scale = trainer.training_config["dataset"].get("position_scale", 1.0)

        generator = torch.Generator(device=pl_module.device)
        generator.manual_seed(42)

        test_list = []

        # Load EEG/fNIRS data from pkl file
        pkl_path = os.path.join('/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/data_final.pkl')
        with open(pkl_path, 'rb') as f:
            bio_data = pickle.load(f)

        if condition_type == "subject":
            # For SEED dataset, use source and target images from the dataset
            # Sample a few entries from the test dataset
            test_list.extend(
                [
                    (
                        Image.open("/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/images/4104_0.jpg").resize((condition_size, condition_size)),
                        [0, -condition_size // 16],
                        "Enlarge the mouse, shrink the character, and swap the positions of the mouse and the character.",
                        {"eeg": torch.tensor(bio_data['4104_0.jpg']["EEG"]), "fnirs": torch.tensor(bio_data['4104_0.jpg']["FNIRS"]),
                         "ppg": torch.tensor(bio_data['4104_0.jpg']["PPG"]),
                         "motion": torch.tensor(bio_data['4104_0.jpg']["Motion"])},
                    ),
                    (
                        Image.open("/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/images/3102_0.jpg").resize((condition_size, condition_size)),
                        [0, -condition_size // 16],
                        "Add personnel, desks, signs, and some other things to the office, increase the brightness of the picture, and make the office more lively.",
                        {"eeg": torch.tensor(bio_data['3102_0.jpg']["EEG"]), "fnirs": torch.tensor(bio_data['3102_0.jpg']["FNIRS"]),
                         "ppg": torch.tensor(bio_data['3102_0.jpg']["PPG"]),
                         "motion": torch.tensor(bio_data['3102_0.jpg']["Motion"])},
                    ),
                    (
                        Image.open("data/images/22004_0.jpg").resize((condition_size, condition_size)),
                        [0, -condition_size // 16],
                        "Remove all other people except the boy and girl from the background, making the boy and girl stand out more in the picture.",
                        {"eeg": torch.tensor(bio_data['22004_0.jpg']["EEG"]), "fnirs": torch.tensor(bio_data['22004_0.jpg']["FNIRS"]),
                         "ppg": torch.tensor(bio_data['22004_0.jpg']["PPG"]),
                         "motion": torch.tensor(bio_data['22004_0.jpg']["Motion"])},
                    ),
                    (
                        Image.open("data/images/22006_0.jpg").resize((condition_size, condition_size)),
                        [0, -condition_size // 16],
                        "Add wings and a halo to the jumping cat, and adjust the lighting to enhance the angel effect.",
                        {"eeg": torch.tensor(bio_data['22006_0.jpg']["EEG"]), "fnirs": torch.tensor(bio_data['22006_0.jpg']["FNIRS"]),
                         "ppg": torch.tensor(bio_data['22006_0.jpg']["PPG"]),
                         "motion": torch.tensor(bio_data['22006_0.jpg']["Motion"])},
                    )
                ]
            )
        # elif condition_type == "subject":
        #     test_list.extend(
        #         [
        #             (
        #                 Image.open("/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/zhangkaipeng-24043/zhoupengfei/OminiControl/imagedataset/images/5901_0.jpg").resize((condition_size, condition_size)),
        #                 [0, -condition_size // 16],
        #                 "Retain the part of the original cat's head, replace the background with a combat scene of a person holding a sword, adjust the angle of the cat's head, make it look like a catman holding a sword, and scale it according to the size of the scene",
        #             ),
        #             (
        #                 Image.open("/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/zhangkaipeng-24043/zhoupengfei/OminiControl/imagedataset/images/5915_0.jpg").resize((condition_size, condition_size)),
        #                 [0, -condition_size // 16],
        #                 "Remove the girl from the photo while maintaining the integrity of the remaining elements.",
        #             ),
        #             (
        #                 Image.open("/inspire/hdd/ws-c6f77a66-a5f5-45dc-a4ce-1e856fe7a7b4/project/zhangkaipeng-24043/zhoupengfei/OminiControl/imagedataset/images/5942_0.jpg").resize((condition_size, condition_size)),
        #                 [0, -condition_size // 16],
        #                 "Replace the original background with a blue sky and waves scene, make some adjustments to the inflatable duck-shaped ride and the woman on it, including the angle and size, so that the image looks like a woman riding a duck surfing on the ocean.",
        #             ),
        #         ]
        #     )
        elif condition_type == "canny":
            condition_img = Image.open("assets/vase_hq.jpg").resize(
                (condition_size, condition_size)
            )
            condition_img = np.array(condition_img)
            condition_img = cv2.Canny(condition_img, 100, 200)
            condition_img = Image.fromarray(condition_img).convert("RGB")
            test_list.append(
                (
                    condition_img,
                    [0, 0],
                    "A beautiful vase on a table.",
                    {"position_scale": position_scale} if position_scale != 1.0 else {},
                )
            )
        elif condition_type == "coloring":
            condition_img = (
                Image.open("assets/vase_hq.jpg")
                .resize((condition_size, condition_size))
                .convert("L")
                .convert("RGB")
            )
            test_list.append((condition_img, [0, 0], "A beautiful vase on a table."))
        elif condition_type == "depth":
            if not hasattr(self, "deepth_pipe"):
                self.deepth_pipe = pipeline(
                    task="depth-estimation",
                    model="LiheYoung/depth-anything-small-hf",
                    device="cpu",
                )
            condition_img = (
                Image.open("assets/vase_hq.jpg")
                .resize((condition_size, condition_size))
                .convert("RGB")
            )
            condition_img = self.deepth_pipe(condition_img)["depth"].convert("RGB")
            test_list.append(
                (
                    condition_img,
                    [0, 0],
                    "A beautiful vase on a table.",
                    {"position_scale": position_scale} if position_scale != 1.0 else {},
                )
            )
        elif condition_type == "depth_pred":
            condition_img = (
                Image.open("assets/vase_hq.jpg")
                .resize((condition_size, condition_size))
                .convert("RGB")
            )
            test_list.append((condition_img, [0, 0], "A beautiful vase on a table."))
        elif condition_type == "deblurring":
            blur_radius = 5
            image = Image.open("./assets/vase_hq.jpg")
            condition_img = (
                image.convert("RGB")
                .resize((condition_size, condition_size))
                .filter(ImageFilter.GaussianBlur(blur_radius))
                .convert("RGB")
            )
            test_list.append(
                (
                    condition_img,
                    [0, 0],
                    "A beautiful vase on a table.",
                    {"position_scale": position_scale} if position_scale != 1.0 else {},
                )
            )
        elif condition_type == "fill":
            condition_img = (
                Image.open("./assets/vase_hq.jpg")
                .resize((condition_size, condition_size))
                .convert("RGB")
            )
            mask = Image.new("L", condition_img.size, 0)
            draw = ImageDraw.Draw(mask)
            a = condition_img.size[0] // 4
            b = a * 3
            draw.rectangle([a, a, b, b], fill=255)
            condition_img = Image.composite(
                condition_img, Image.new("RGB", condition_img.size, (0, 0, 0)), mask
            )
            test_list.append((condition_img, [0, 0], "A beautiful vase on a table."))
        elif condition_type == "sr":
            condition_img = (
                Image.open("assets/vase_hq.jpg")
                .resize((condition_size, condition_size))
                .convert("RGB")
            )
            test_list.append((condition_img, [0, -16], "A beautiful vase on a table."))
        elif condition_type == "cartoon":
            condition_img = (
                Image.open("assets/cartoon_boy.png")
                .resize((condition_size, condition_size))
                .convert("RGB")
            )
            test_list.append(
                (
                    condition_img,
                    [0, -16],
                    "A cartoon character in a white background. He is looking right, and running.",
                )
            )
        else:
            raise NotImplementedError

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for i, (condition_img, position_delta, prompt, *others) in enumerate(test_list):
            condition = Condition(
                condition_type=condition_type,
                condition=condition_img.resize(
                    (condition_size, condition_size)
                ).convert("RGB"),
                position_delta=position_delta,
                **(others[0] if others else {}),
            )
            
            # Extract EEG and fNIRS data if available
            additional_condition1 = None
            additional_condition2 = None
            additional_condition3 = None
            additional_condition4 = None
            
            if others and "eeg" in others[0]:
                additional_condition1 = others[0]["eeg"]
            
            if others and "fnirs" in others[0]:
                additional_condition2 = others[0]["fnirs"]
            
            if others and "ppg" in others[0]:
                additional_condition3 = others[0]["ppg"]
            
            if others and "motion" in others[0]:
                additional_condition4 = others[0]["motion"]
            
            res = generate(
                pl_module,
                pl_module.flux_pipe,
                prompt=prompt,
                conditions=[condition],
                height=target_size,
                width=target_size,
                generator=generator,
                model_config=pl_module.model_config,
                default_lora=True,
                additional_condition1=additional_condition1,  # EEG data
                additional_condition2=additional_condition2,  # fNIRS data
                additional_condition3=additional_condition3,  # PPG data
                additional_condition4=additional_condition4,  # Motion data
            )

            res.images[0].save(
                os.path.join(save_path, f"{file_name}_{condition_type}_{i}.jpg")
            )
