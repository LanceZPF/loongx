from PIL import Image, ImageFilter, ImageDraw
import cv2
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
import random
import json
import os
import pickle

class SeedDataset(Dataset):
    def __init__(
        self,
        jsonl_path,
        condition_size: int = 512,
        condition_type: str = "subject", 
        image_dir="",
        transform=None,
        return_pil_image=False,
    ):
        """
        Dataset for loading testset.jsonl data and EEG/fNIRS data
        
        Args:
            jsonl_path: Path to the jsonl file
            image_dir: Directory containing the images (if empty, assumes paths in jsonl are absolute)
            transform: Optional transform to be applied on the images
            return_pil_image: If True, return PIL images instead of tensors
        """
        
        self.samples = []
        self.image_dir = image_dir
        self.transform = transform
        self.return_pil_image = return_pil_image

        self.condition_type = condition_type
        self.condition_size = condition_size

        # Load EEG/fNIRS data from pkl file
        pkl_path = os.path.join(os.path.dirname(jsonl_path), 'data_final.pkl')

        with open(pkl_path, 'rb') as f:
            self.bio_data = pickle.load(f)

        # Load the jsonl file
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                aline = json.loads(line)
                if aline['source_image'].split("/")[-1] in self.bio_data:
                    self.samples.append(aline)
        
        # Default transform if none provided
        if self.transform is None:
            self.transform = T.Compose([
                T.Resize((512, 512)),
                T.ToTensor(),
            ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # Get image paths
        source_path = os.path.join(self.image_dir, item["source_image"])
        target_path = os.path.join(self.image_dir, item["target_image"])
        
        # Load images
        source_image = Image.open(source_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")
        
        # Apply transforms
        if not self.return_pil_image:
            source_image = self.transform(source_image)
            target_image = self.transform(target_image)

        # Get EEG/fNIRS data
        bio_data = self.bio_data[item["source_image"].split("/")[-1]]
        eeg_data = np.array(bio_data['EEG'])
        fnirs_data = np.array(bio_data['FNIRS']) if 'FNIRS' in bio_data else None

        ppg_data = np.array(bio_data['PPG']) if 'PPG' in bio_data else None
        motion_data = np.array(bio_data['Motion']) if 'Motion' in bio_data else None
        
        # Return data
        return {
            "image": source_image,
            "condition": target_image,
            "description": item["speech2text"] if "speech2text" in item else item["instruction"],
            # "description": item["instruction"],
            "condition_type": self.condition_type,
            "position_delta": np.array([0, -self.condition_size // 16]),
            "eeg": eeg_data,
            "fnirs": fnirs_data,
            'ppg': ppg_data,
            'motion': motion_data
        }


class Subject200KDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        condition_size: int = 512,
        target_size: int = 512,
        image_size: int = 512,
        padding: int = 0,
        condition_type: str = "subject",
        drop_text_prob: float = 0.1,
        drop_image_prob: float = 0.1,
        return_pil_image: bool = False,
    ):
        self.base_dataset = base_dataset
        self.condition_size = condition_size
        self.target_size = target_size
        self.image_size = image_size
        self.padding = padding
        self.condition_type = condition_type
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.return_pil_image = return_pil_image

        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.base_dataset) * 2

    def __getitem__(self, idx):
        # If target is 0, left image is target, right image is condition
        target = idx % 2
        item = self.base_dataset[idx // 2]

        # Crop the image to target and condition
        image = item["image"]
        left_img = image.crop(
            (
                self.padding,
                self.padding,
                self.image_size + self.padding,
                self.image_size + self.padding,
            )
        )
        right_img = image.crop(
            (
                self.image_size + self.padding * 2,
                self.padding,
                self.image_size * 2 + self.padding * 2,
                self.image_size + self.padding,
            )
        )

        # Get the target and condition image
        target_image, condition_img = (
            (left_img, right_img) if target == 0 else (right_img, left_img)
        )

        # Resize the image
        condition_img = condition_img.resize(
            (self.condition_size, self.condition_size)
        ).convert("RGB")
        target_image = target_image.resize(
            (self.target_size, self.target_size)
        ).convert("RGB")

        # Get the description
        description = item["description"][
            "description_0" if target == 0 else "description_1"
        ]

        # Randomly drop text or image
        drop_text = random.random() < self.drop_text_prob
        drop_image = random.random() < self.drop_image_prob
        if drop_text:
            description = ""
        if drop_image:
            condition_img = Image.new(
                "RGB", (self.condition_size, self.condition_size), (0, 0, 0)
            )

        return {
            "image": self.to_tensor(target_image),
            "condition": self.to_tensor(condition_img),
            "condition_type": self.condition_type,
            "description": description,
            # 16 is the downscale factor of the image
            "position_delta": np.array([0, -self.condition_size // 16]),
            **({"pil_image": image} if self.return_pil_image else {}),
        }


class ImageConditionDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        condition_size: int = 512,
        target_size: int = 512,
        condition_type: str = "canny",
        drop_text_prob: float = 0.1,
        drop_image_prob: float = 0.1,
        return_pil_image: bool = False,
        position_scale=1.0,
    ):
        self.base_dataset = base_dataset
        self.condition_size = condition_size
        self.target_size = target_size
        self.condition_type = condition_type
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.return_pil_image = return_pil_image
        self.position_scale = position_scale

        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.base_dataset)

    @property
    def depth_pipe(self):
        if not hasattr(self, "_depth_pipe"):
            from transformers import pipeline

            self._depth_pipe = pipeline(
                task="depth-estimation",
                model="LiheYoung/depth-anything-small-hf",
                device="cpu",
            )
        return self._depth_pipe

    def _get_canny_edge(self, img):
        resize_ratio = self.condition_size / max(img.size)
        img = img.resize(
            (int(img.size[0] * resize_ratio), int(img.size[1] * resize_ratio))
        )
        img_np = np.array(img)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(img_gray, 100, 200)
        return Image.fromarray(edges).convert("RGB")

    def __getitem__(self, idx):
        image = self.base_dataset[idx]["jpg"]
        image = image.resize((self.target_size, self.target_size)).convert("RGB")
        description = self.base_dataset[idx]["json"]["prompt"]

        enable_scale = random.random() < 1
        if not enable_scale:
            condition_size = int(self.condition_size * self.position_scale)
            position_scale = 1.0
        else:
            condition_size = self.condition_size
            position_scale = self.position_scale

        # Get the condition image
        position_delta = np.array([0, 0])
        if self.condition_type == "canny":
            condition_img = self._get_canny_edge(image)
        elif self.condition_type == "coloring":
            condition_img = (
                image.resize((condition_size, condition_size))
                .convert("L")
                .convert("RGB")
            )
        elif self.condition_type == "deblurring":
            blur_radius = random.randint(1, 10)
            condition_img = (
                image.convert("RGB")
                .filter(ImageFilter.GaussianBlur(blur_radius))
                .resize((condition_size, condition_size))
                .convert("RGB")
            )
        elif self.condition_type == "depth":
            condition_img = self.depth_pipe(image)["depth"].convert("RGB")
            condition_img = condition_img.resize((condition_size, condition_size))
        elif self.condition_type == "depth_pred":
            condition_img = image
            image = self.depth_pipe(condition_img)["depth"].convert("RGB")
            description = f"[depth] {description}"
        elif self.condition_type == "fill":
            condition_img = image.resize((condition_size, condition_size)).convert(
                "RGB"
            )
            w, h = image.size
            x1, x2 = sorted([random.randint(0, w), random.randint(0, w)])
            y1, y2 = sorted([random.randint(0, h), random.randint(0, h)])
            mask = Image.new("L", image.size, 0)
            draw = ImageDraw.Draw(mask)
            draw.rectangle([x1, y1, x2, y2], fill=255)
            if random.random() > 0.5:
                mask = Image.eval(mask, lambda a: 255 - a)
            condition_img = Image.composite(
                image, Image.new("RGB", image.size, (0, 0, 0)), mask
            )
        elif self.condition_type == "sr":
            condition_img = image.resize((condition_size, condition_size)).convert(
                "RGB"
            )
            position_delta = np.array([0, -condition_size // 16])

        else:
            raise ValueError(f"Condition type {self.condition_type} not implemented")

        # Randomly drop text or image
        drop_text = random.random() < self.drop_text_prob
        drop_image = random.random() < self.drop_image_prob
        if drop_text:
            description = ""
        if drop_image:
            condition_img = Image.new(
                "RGB", (condition_size, condition_size), (0, 0, 0)
            )

        return {
            "image": self.to_tensor(image),
            "condition": self.to_tensor(condition_img),
            "condition_type": self.condition_type,
            "description": description,
            "position_delta": position_delta,
            **({"pil_image": [image, condition_img]} if self.return_pil_image else {}),
            **({"position_scale": position_scale} if position_scale != 1.0 else {}),
        }


class CartoonDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        condition_size: int = 1024,
        target_size: int = 1024,
        image_size: int = 1024,
        padding: int = 0,
        condition_type: str = "cartoon",
        drop_text_prob: float = 0.1,
        drop_image_prob: float = 0.1,
        return_pil_image: bool = False,
    ):
        self.base_dataset = base_dataset
        self.condition_size = condition_size
        self.target_size = target_size
        self.image_size = image_size
        self.padding = padding
        self.condition_type = condition_type
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.return_pil_image = return_pil_image

        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data = self.base_dataset[idx]
        condition_img = data["condition"]
        target_image = data["target"]

        # Tag
        tag = data["tags"][0]

        target_description = data["target_description"]

        description = {
            "lion": "lion like animal",
            "bear": "bear like animal",
            "gorilla": "gorilla like animal",
            "dog": "dog like animal",
            "elephant": "elephant like animal",
            "eagle": "eagle like bird",
            "tiger": "tiger like animal",
            "owl": "owl like bird",
            "woman": "woman",
            "parrot": "parrot like bird",
            "mouse": "mouse like animal",
            "man": "man",
            "pigeon": "pigeon like bird",
            "girl": "girl",
            "panda": "panda like animal",
            "crocodile": "crocodile like animal",
            "rabbit": "rabbit like animal",
            "boy": "boy",
            "monkey": "monkey like animal",
            "cat": "cat like animal",
        }

        # Resize the image
        condition_img = condition_img.resize(
            (self.condition_size, self.condition_size)
        ).convert("RGB")
        target_image = target_image.resize(
            (self.target_size, self.target_size)
        ).convert("RGB")

        # Process datum to create description
        description = data.get(
            "description",
            f"Photo of a {description[tag]} cartoon character in a white background. Character is facing {target_description['facing_direction']}. Character pose is {target_description['pose']}.",
        )

        # Randomly drop text or image
        drop_text = random.random() < self.drop_text_prob
        drop_image = random.random() < self.drop_image_prob
        if drop_text:
            description = ""
        if drop_image:
            condition_img = Image.new(
                "RGB", (self.condition_size, self.condition_size), (0, 0, 0)
            )

        return {
            "image": self.to_tensor(target_image),
            "condition": self.to_tensor(condition_img),
            "condition_type": self.condition_type,
            "description": description,
            # 16 is the downscale factor of the image
            "position_delta": np.array([0, -16]),
        }