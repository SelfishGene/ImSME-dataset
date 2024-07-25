import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import matplotlib.font_manager as fm
from utils import create_char_image
from config import CHAR_REPLACEMENT_DICT, CHARACTERS_LIST, NAME_SHORT_PATH_DICT

def list_available_fonts():
    font_paths = fm.findSystemFonts()
    available_fonts = {}
    for font_path in font_paths:
        font_name = fm.FontProperties(fname=font_path).get_name()
        available_fonts[font_name] = font_path
    return available_fonts


def create_char_images_dataset(output_folder, fonts_to_use, image_size=(128, 128), return_tight_bbox=False):
    os.makedirs(output_folder, exist_ok=True)
    
    total_images = len([char for char in CHARACTERS_LIST]) * len(fonts_to_use)
    print(f'Total images to save: {total_images}')
    
    with tqdm(total=total_images, desc="Generating character images") as pbar:
        for char in CHARACTERS_LIST:
            for font_name, font_path in fonts_to_use.items():
                char_name = CHAR_REPLACEMENT_DICT.get(char, char)
                # use the shortened font path as the font name
                font_name_to_use = os.path.basename(font_path) if os.path.exists(font_path) else font_path.split('.')[0]
                filename = f'char_{char_name}_font_{font_name_to_use}.png'
                try:
                    image = create_char_image(char, font_path, final_image_size=image_size, return_tight_bbox=return_tight_bbox)
                    image.save(os.path.join(output_folder, filename))
                    pbar.update(1)
                except Exception as e:
                    print(f'Error saving image: "{filename}". Error: {str(e)}')
    
    print(f'Character images dataset created in: {output_folder}')


def display_sample_images(dataset_folder, num_samples=36):
    files = os.listdir(dataset_folder)
    sample_files = np.random.choice(files, num_samples, replace=False)
    
    num_rows = int(np.sqrt(num_samples))
    num_cols = int(np.ceil(num_samples / num_rows))
    
    plt.figure(figsize=(18, 20))
    
    for i, file in enumerate(sample_files):
        image = Image.open(os.path.join(dataset_folder, file))
        char = file.split('_font_')[0].split('char_')[1]
        font = file.split('_font_')[1].split('.')[0]
        
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(f'Char: "{char}"\nFont: "{font}"')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Flag to choose between using all available fonts or just NAME_SHORT_PATH_DICT
    USE_ALL_AVAILABLE_FONTS = False

    print('-' * 80)
    print("Creating character images dataset")
    print('-' * 80)

    # Set output folders
    output_folder = "data/char_images_dataset"
    output_folder_tight = "data/char_images_dataset_tight"
    
    # Get available fonts
    available_fonts = list_available_fonts()
    
    if USE_ALL_AVAILABLE_FONTS:
        fonts_to_use = available_fonts
        print(f"Using all {len(fonts_to_use)} available fonts")
    else:
        fonts_to_use = {}
        for font_name, short_path in NAME_SHORT_PATH_DICT.items():
            full_path = next((path for name, path in available_fonts.items() if path.lower().endswith(short_path.lower())), None)
            if full_path:
                fonts_to_use[font_name] = full_path
            else:
                print(f"Warning: Font {font_name} ({short_path}) not found in available fonts.")
        print(f"Using {len(fonts_to_use)} fonts from NAME_SHORT_PATH_DICT")
    
    print('-' * 80)
    
    # Create datasets
    print("Creating regular character images dataset:")
    create_char_images_dataset(output_folder, fonts_to_use, return_tight_bbox=False)
    print('-' * 80)
    
    print("Creating tight bounding box character images dataset:")
    create_char_images_dataset(output_folder_tight, fonts_to_use, return_tight_bbox=True)
    print('-' * 80)
    
    # Display sample images from both datasets
    print("Displaying sample images from regular dataset:")
    display_sample_images(output_folder)
    
    print("Displaying sample images from tight bounding box dataset:")
    display_sample_images(output_folder_tight)

    print('-' * 80)
    print("Character images dataset creation completed.")
    print('-' * 80)