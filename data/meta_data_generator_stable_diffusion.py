import pandas as pd
import random
import argparse
import string
import os

# Create the argument parser
parser = argparse.ArgumentParser()

# Add the boolean argument
parser.add_argument(
    "--do_random_string", action="store_true", help="Whether to do random string"
)
args = parser.parse_args()


def random_string(length=6):
    return "".join(random.choice(string.ascii_uppercase) for _ in range(length))


# Access the value of the boolean argument
do_random_string = args.do_random_string

# set the random seed
random.seed(0)
# Define mappings for keywords
mapping = {
    "two-chamber": random_string(),
    "four-chamber": random_string(),
    "ed": random_string(),
    "es": random_string(),
    "ultrasound image": random_string(),
    "heart": random_string(),
}

base_directory = "."
# List of class folders (adjust these folder names if they are different)
class_folders = ["class_ch2_ed", "class_ch2_es", "class_ch4_ed", "class_ch4_es"]

# List to store image metadata
metadata = []

for folder in class_folders:
    # Construct the full path to the folder
    folder_path = os.path.join(base_directory, folder)

    # Check if the folder exists and is a directory
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # Iterate through the files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".png"):
                # Determine the chamber and phase based on the folder name
                chamber = "two-chamber" if "ch2" in folder else "four-chamber"
                phase = "ed" if "ed" in folder else "es"

                # Create the description using the mapping
                description = f"An {mapping['ultrasound image']} shows the {mapping['heart']} in a {mapping[chamber]} view during the {mapping[phase]} phase."

                # Append the metadata for this image
                metadata.append(
                    {
                        "image": os.path.join("data", folder, filename),
                        "text": description,
                    }
                )

    else:
        print(f"Folder {folder} does not exist or is not a directory.")

# Create a DataFrame
df = pd.DataFrame(metadata)

# Save the DataFrame to a CSV file
df.to_csv("metadata_stable_diffusion.csv", index=False)

print("Metadata CSV file has been created.")

# save mapping to a file as json
import json

with open("mapping.json", "w") as f:
    json.dump(mapping, f)

