import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
import argparse
import os


def generate_data(classes,numbers_of_augmentation):
    # find the maximum number of images per class
    max_images_per_class = numbers_of_augmentation
    # chekc for cuda avalibility
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    
    pipeline = StableDiffusionPipeline.from_pretrained("results/", use_safetensors=True).to(device)
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    for class_Ultrasound in classes:

        if "brain" in class_Ultrasound:
            prompt = "Ultrasound image of the "+class_Ultrasound[12:]+" plane of the fetal brain"
        else:
            prompt = "Ultrasound image of the "+class_Ultrasound+" plane"
        num_of_augmentations = max_images_per_class
        print("load model for",class_Ultrasound)
        

        os.makedirs(f"generated_texual/{class_Ultrasound}",exist_ok=True)
        # use batch size of 2
        batch_size = 2
        for i in range(num_of_augmentations//batch_size+1):

            print(f"Augmenting {class_Ultrasound} {i+1}/{num_of_augmentations//batch_size+1}")
            
            augmented_image = pipeline(num_images_per_prompt=batch_size,prompt=prompt,num_inference_steps=800).images
            for j in range(batch_size):
                augmented_image[j].save(f"generated_texual/{class_Ultrasound}/{i*batch_size+j}.jpg")

def generate_data_mapper(classes,numbers_of_augmentation,mapping):
    # find the maximum number of images per class
    max_images_per_class = numbers_of_augmentation
    # chekc for cuda avalibility
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    
    pipeline = StableDiffusionPipeline.from_pretrained("results/", use_safetensors=True,safety_checker=None).to(device)
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    for class_Ultrasound in classes:
        if "ch2" in class_Ultrasound:
            if "ed" in class_Ultrasound:
                prompt = f"An {mapping['ultrasound image']} shows the {mapping['heart']} in a {mapping['two-chamber']} view during the {mapping['ed']} phase."
            else:
                prompt = f"An {mapping['ultrasound image']} shows the {mapping['heart']} in a {mapping['two-chamber']} view during the {mapping['es']} phase."
        else:
            if "ed" in class_Ultrasound:
                prompt = f"An {mapping['ultrasound image']} shows the {mapping['heart']} in a {mapping['four-chamber']} view during the {mapping['ed']} phase."
            else:
                prompt = f"An {mapping['ultrasound image']} shows the {mapping['heart']} in a {mapping['four-chamber']} view during the {mapping['es']} phase."
        num_of_augmentations = max_images_per_class
        print("load model for",class_Ultrasound)     

        os.makedirs(f"generated/{class_Ultrasound}",exist_ok=True)
        # use batch size of 4
        batch_size = 8
        for i in range(num_of_augmentations//batch_size+1):
            print(f"Augmenting {class_Ultrasound} {i+1}/{num_of_augmentations//batch_size+1}")
            augmented_image = pipeline(num_images_per_prompt=batch_size,prompt=prompt,num_inference_steps=40).images
            print(len(augmented_image))
            for j in range(batch_size):
                augmented_image[j].save(f"generated/{class_Ultrasound}/{i*batch_size+j}.jpg")       

def parse_args():
    parser = argparse.ArgumentParser(description='Data Generation')
    parser.add_argument('--number_of_images', type=int, help='Number to get')
    parser.add_argument('--do_random_string', action='store_true', help='Whether to do random string')
    return parser.parse_args()

# add main function
if __name__ == '__main__':
    args = parse_args()
    classes = ['class_ch2_ed',
    'class_ch2_es',
    'class_ch4_ed',
    'class_ch4_es',]
    number_of_images = args.number_of_images

    if args.do_random_string:
        # load the dictionary mapper
        import json
        with open("data/mapping.json", "r") as outfile: 
            mapper = json.load(outfile)
        generate_data_mapper(classes,number_of_images,mapper)
    else:
        generate_data(classes,number_of_images)




    
    
