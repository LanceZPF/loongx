import gradio as gr
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers.pipelines import FluxPipeline
from diffusers import FluxTransformer2DModel
import numpy as np

from ..flux.condition import Condition
from ..flux.generate import seed_everything, generate

pipe = None
use_int8 = False


def get_gpu_memory():
    return torch.cuda.get_device_properties(0).total_memory / 1024**3


def init_pipeline():
    global pipe
    if use_int8 or get_gpu_memory() < 33:
        transformer_model = FluxTransformer2DModel.from_pretrained(
            "sayakpaul/flux.1-schell-int8wo-improved",
            torch_dtype=torch.bfloat16,
            use_safetensors=False,
        )
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            transformer=transformer_model,
            torch_dtype=torch.bfloat16,
        )
    else:
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
        )
    pipe = pipe.to("cuda")
    pipe.load_lora_weights(
        "Yuanshi/LoongX",
        weight_name="omini/subject_512.safetensors",
        adapter_name="subject",
    )


def process_image_and_text(image, text):
    # center crop image
    w, h, min_size = image.size[0], image.size[1], min(image.size)
    image = image.crop(
        (
            (w - min_size) // 2,
            (h - min_size) // 2,
            (w + min_size) // 2,
            (h + min_size) // 2,
        )
    )
    image = image.resize((512, 512))

    condition = Condition("subject", image, position_delta=(0, 32))

    if pipe is None:
        init_pipeline()

    result_img = generate(
        pipe,
        prompt=text.strip(),
        conditions=[condition],
        num_inference_steps=8,
        height=512,
        width=512,
    ).images[0]

    return result_img


def get_samples():
    sample_list = [
        {
            "image": "assets/oranges.jpg",
            "text": "A very close up view of this item. It is placed on a wooden table. The background is a dark room, the TV is on, and the screen is showing a cooking show. With text on the screen that reads 'Omini Control!'",
        },
        {
            "image": "assets/penguin.jpg",
            "text": "On Christmas evening, on a crowded sidewalk, this item sits on the road, covered in snow and wearing a Christmas hat, holding a sign that reads 'Omini Control!'",
        },
        {
            "image": "assets/rc_car.jpg",
            "text": "A film style shot. On the moon, this item drives across the moon surface. The background is that Earth looms large in the foreground.",
        },
        {
            "image": "assets/clock.jpg",
            "text": "In a Bauhaus style room, this item is placed on a shiny glass table, with a vase of flowers next to it. In the afternoon sun, the shadows of the blinds are cast on the wall.",
        },
        {
            "image": "assets/tshirt.jpg",
            "text": "On the beach, a lady sits under a beach umbrella with 'Omini' written on it. She's wearing this shirt and has a big smile on her face, with her surfboard hehind her.",
        },
    ]
    return [[Image.open(sample["image"]), sample["text"]] for sample in sample_list]


demo = gr.Interface(
    fn=process_image_and_text,
    inputs=[
        gr.Image(type="pil"),
        gr.Textbox(lines=2),
    ],
    outputs=gr.Image(type="pil"),
    title="LoongX / Subject driven generation",
    examples=get_samples(),
)

if __name__ == "__main__":
    init_pipeline()
    demo.launch(
        debug=True,
    )
