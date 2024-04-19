import os
import sys
import random

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from diffusers.utils import load_image
import torch
from torchvision import transforms

mask_path = "path/to/inference/masks"  # the folder containing the segmentation maps
base_model_path = "./stable-diffusion-v1-5"  # path base SD model
controlnet_path = "path/to/cnet"  # this the path to your controlnet checkpoint
save_path_image = "path/to/save/images"  # the path for saving the synthetic images
save_path_mask = "path/to/save/masks"  # the path for saving the corresponding maps

images = sorted(os.listdir(mask_path))

if not os.path.exists(save_path_image):
    os.mkdir(save_path_image)

if not os.path.exists(save_path_mask):
    os.mkdir(save_path_mask)

prompts = {
    "ch2": "an ultrasound image of heart in two-chamber view",
    "ch4": "an ultrasound image of heart in four-chamber view",
}

controlnet = ControlNetModel.from_pretrained(
    controlnet_path,
    torch_dtype=torch.float16,
    use_safetensors=True,
    safety_checker=None,
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    use_safetensors=True,
    safety_checker=None,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

random.seed(0)
torch.manual_seed(0)

for ii, image in enumerate(images):
    print(f"\n***Image: {image} *** \n")
    for i in range(5):
        control_image = load_image(mask_path + image)
        control_image.save(save_path_mask + image[:-4] + f"_mask_id{i}.png")
        prompt = prompts[image[:3]]
        syn_image = pipe(prompt, num_inference_steps=50, image=control_image).images[0]
        syn_image.save(save_path_image + image[:-4] + f"_generated_id{i}.png")
