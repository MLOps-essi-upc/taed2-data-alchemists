
import os
import numpy as np
import pandas as pd
from PIL import Image

# Folder path where the images are stored
image_folder = 'data/test'

# Initialize an empty list to store normalized pixel values
normalized_pixel_data = []

# Iterate through the images in the folder
for image_name in sorted(os.listdir(image_folder)):
    image_path = os.path.join(image_folder, image_name)

    # Open the image using Pillow (PIL)
    with Image.open(image_path) as img:
        # Convert the image to grayscale (if not already)
        img = img.convert('L')

        # Convert the image to a NumPy array
        img_array = np.array(img)

        # Normalize pixel values to [0, 255]
        img_normalized = (img_array / 255.0 * 255).astype(int)

        # Flatten the 2D array into a 1D array
        img_flat = img_normalized.flatten()

        # Get the 'id' (last character of the image name) and convert it to an integer
        image_id = [int(image_name[-5])]

        image_id.extend(img_flat)
        print(len(image_id))
        # Append the 'id' and the flattened, normalized pixel values to the list
        normalized_pixel_data.append(image_id)
        print(len(normalized_pixel_data[0]))

# Create a DataFrame from the list of normalized pixel values
pixel_df = pd.DataFrame(normalized_pixel_data, columns=['id'] + ['pixel{}'.format(i) for i in range(len(normalized_pixel_data[0])-1)])

# Save the DataFrame to a CSV file
pixel_df.to_csv('data/test.csv', index=False)
