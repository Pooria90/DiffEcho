from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from diffusers.utils import load_image
import torch
from torchvision import transforms

base_model_path = "path/to/sd"  # this is the path to stable diffusion model. Mine was "./stable-diffusion-v1-5"
controlnet_path = "path/to/cnet"  # this the path to your controlnet checkpoint
image_path = "path/to/input/image"  # this is the path to the input segmentation map
save_path = "path/to/save/outputs"  # this is the path to save generated images

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

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# xformers was not installed for me. So I removed commented the following line.
# pipe.enable_xformers_memory_efficient_attention()

pipe.enable_model_cpu_offload()

control_image = load_image(image_path)
prompt = (
    "an ultrasound image of heart in four-chamber view"  # or replace "four" with "two"
)

# generates 5 synthetic images from the control_image (segmentation map)
generator = torch.manual_seed(0)
for i in range(5):
    image = pipe(prompt, num_inference_steps=50, image=control_image).images[0]
    image.save(f"{save_path}/generated_{i}.png")
