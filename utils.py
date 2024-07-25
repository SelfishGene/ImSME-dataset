import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from config import DIGIT_CHARS, SYMBOL_CHARS, CHAR_REPLACEMENT_DICT, SECTION_ORDER

OUTPUT_HEIGHT = 128
RESIZE_HEIGHT_DIGIT = 112
RESIZE_HEIGHT_SYMBOL = 72
WIDTH_MULT_FACTORS = [0.25, 1.0]


def create_char_image(char, selected_font, final_image_size=(128, 128), return_tight_bbox=False):
    image_size = (final_image_size[0] * 2, final_image_size[1] * 2)
    font_size = 100
    image = Image.new("L", image_size, "black")
    draw = ImageDraw.Draw(image)
    
    font = ImageFont.truetype(selected_font, font_size)
    text_bbox = draw.textbbox((0, 0), char, font=font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    
    scale_H = (image_size[1] * 0.45) / text_height
    scale_W = (image_size[0] * 0.45) / text_width
    scale = np.array([scale_H, scale_W]).min()
    scaled_font = ImageFont.truetype(selected_font, int(font_size * scale))
    
    text_bbox = draw.textbbox((0, 0), char, font=scaled_font)
    position = (image_size[0] // 2, image_size[1] // 2)
    
    draw.text(position, char, font=scaled_font, fill="white", anchor="mm")

    if return_tight_bbox:
        non_zero_pixels = np.array(image).nonzero()
        top_left = (max(0, non_zero_pixels[0].min() - 10), max(0, non_zero_pixels[1].min() - 10))
        bottom_right = (min(image_size[0], non_zero_pixels[0].max() + 10), min(image_size[1], non_zero_pixels[1].max() + 10))
        image = image.crop((top_left[1], top_left[0], bottom_right[1], bottom_right[0]))
    else:
        crop_bbox = (image_size[0] // 32, image_size[1] // 32, 
                     31 * image_size[0] // 32, 31 * image_size[1] // 32)
        image = image.crop(crop_bbox)

    image = image.resize(final_image_size)
    return image


def equation_to_image(equation_string, char_images_folder, space_padding=48, output_height=OUTPUT_HEIGHT, 
                      resize_height_digit=RESIZE_HEIGHT_DIGIT, resize_height_symbol=RESIZE_HEIGHT_SYMBOL, 
                      width_mult_factors=WIDTH_MULT_FACTORS, use_random_augmentations=True, fonts_pool=None):
    char_images = []
    char_widths = []
    char_fonts = []
    char_offsets = []
    
    all_available_fonts = set(os.path.splitext(file)[0].split('_font_')[1] for file in os.listdir(char_images_folder) if file.startswith('char_'))
    
    if fonts_pool is not None:
        fonts_pool = set(fonts_pool).intersection(all_available_fonts)
        if not fonts_pool:
            print("Warning: None of the specified fonts in fonts_pool are available. Using all available fonts.")
            fonts_pool = all_available_fonts
    else:
        fonts_pool = all_available_fonts
    
    for char in equation_string:
        if char == ' ':
            space_width = space_padding if not use_random_augmentations else max(24, int(space_padding * random.uniform(*width_mult_factors)))
            space_image = Image.new('L', (space_width, output_height), color=0)
            char_images.append(space_image)
            char_widths.append(space_width)
            char_fonts.append('space')
            char_offsets.append(0)
        else:
            char_name = CHAR_REPLACEMENT_DICT.get(char, char)
            char_files = [f for f in os.listdir(char_images_folder) 
                          if f.startswith(f'char_{char_name}_font_') and 
                          f.split('_font_')[1].split('.')[0] in fonts_pool]
            
            if not char_files:
                char_files = [f for f in os.listdir(char_images_folder) 
                              if f.startswith(f'char_{char_name}_font_')]
                if char_files:
                    print(f"Warning: No image found for character '{char}' ('{char_name}') with the specified fonts. Falling back to all available fonts.")
            
            if char_files:
                chosen_file = random.choice(char_files)
                img_path = os.path.join(char_images_folder, chosen_file)
                char_img = Image.open(img_path).convert('L')
                
                resize_height = resize_height_digit if char in DIGIT_CHARS else resize_height_symbol if char in SYMBOL_CHARS else resize_height_digit
                
                aspect_ratio = char_img.width / char_img.height
                new_width = int(resize_height * aspect_ratio)
                if use_random_augmentations:
                    new_width = int(new_width * random.uniform(*width_mult_factors))
                char_img_resized = char_img.resize((new_width, resize_height), Image.LANCZOS)
                
                full_height_img = Image.new('L', (new_width, output_height), color=0)
                
                if char in SYMBOL_CHARS and use_random_augmentations:
                    offset = random.randint(int(0.25 * (output_height - resize_height)), int(0.75 * (output_height - resize_height)))
                elif char in DIGIT_CHARS and use_random_augmentations:
                    offset = random.randint(int(0.05 * (output_height - resize_height)), int(0.95 * (output_height - resize_height)))
                else:
                    offset = (output_height - resize_height) // 2

                full_height_img.paste(char_img_resized, (0, offset))
                
                char_images.append(full_height_img)
                char_widths.append(new_width)
                char_fonts.append(chosen_file.split('_font_')[1].split('.')[0])
                char_offsets.append(offset)
            else:
                print(f"Warning: No image found for character '{char}' ('{char_name}') in any available font.")
                char_widths.append(0)
                char_fonts.append('not_found')
                char_offsets.append(0)
    
    total_width = sum(img.width for img in char_images)
    equation_image = Image.new('L', (total_width, output_height), color=0)
    
    x_offset = 0
    for img in char_images:
        equation_image.paste(img, (x_offset, 0))
        x_offset += img.width
    
    char_details = {
        'chars': list(equation_string),
        'widths': char_widths,
        'fonts': char_fonts,
        'offsets': char_offsets
    }
    
    return equation_image, char_details


def generate_indicators(char_details):
    chars = char_details['chars']
    widths = char_details['widths']
    fonts = char_details['fonts']

    image_width = sum(widths)

    char_indicators = {}
    font_indicators = {}
    section_indicators = {section: np.zeros(image_width, dtype=int) for section in SECTION_ORDER}

    current_position = 0
    current_section = "argument 1"
    for char, width, font in zip(chars, widths, fonts):
        char_indicator = np.zeros(image_width, dtype=int)
        char_indicator[current_position:current_position + width] = 1
        
        if char not in char_indicators:
            char_indicators[char] = np.zeros(image_width, dtype=int)
        char_indicators[char] |= char_indicator

        if font not in font_indicators:
            font_indicators[font] = np.zeros(image_width, dtype=int)
        font_indicators[font] |= char_indicator

        if char in ['+', '-', '*', '/']:
            current_section = "operation"
            section_indicators["symbol"] |= char_indicator
        elif char == '=':
            current_section = "equal sign"
            section_indicators["symbol"] |= char_indicator
        elif current_section == "equal sign":
            current_section = "result"
        elif current_section == "operation":
            current_section = "argument 2"

        if char.isdigit():
            section_indicators["digit"] |= char_indicator
        elif char == ' ':
            section_indicators["space"] |= char_indicator

        section_indicators[current_section] |= char_indicator

        current_position += width

    space_mask = ~section_indicators["space"].astype(bool)
    for key in ["argument 1", "argument 2", "result", "operation", "equal sign"]:
        section_indicators[key] &= space_mask

    return char_indicators, font_indicators, section_indicators


def indicators_to_images(char_indicators, font_indicators, section_indicators, 
                         char_order, section_order, font_order):
    image_width = len(next(iter(char_indicators.values())))
    
    char_array = np.zeros((len(char_order), image_width), dtype=np.int32)
    font_array = np.zeros((len(font_order) + 1, image_width), dtype=np.int32)
    section_array = np.zeros((len(section_order), image_width), dtype=np.int32)
    
    for i, char in enumerate(char_order):
        if char in char_indicators:
            char_array[i] = char_indicators[char]
    
    for i, font in enumerate(font_order):
        if font in font_indicators:
            font_array[i] = font_indicators[font]
    
    other_fonts = set(font_indicators.keys()) - set(font_order)
    other_indicator = np.zeros(image_width, dtype=np.int32)
    for font in other_fonts:
        other_indicator |= font_indicators[font]
    font_array[-1] = other_indicator
    
    for i, section in enumerate(section_order):
        if section in section_indicators:
            section_array[i] = section_indicators[section]
    
    char_image = Image.fromarray((char_array * 255).astype(np.uint8))
    font_image = Image.fromarray((font_array * 255).astype(np.uint8))
    section_image = Image.fromarray((section_array * 255).astype(np.uint8))
    
    return char_image, font_image, section_image


def generate_equation_descriptions(row, char_details, image_height, image_width):
    equation_str = row['full string']
    is_canonical = row['is_canonical']
    font_usage = row['font_usage']
    
    operation_map = {'+': 'addition', '-': 'subtraction', '*': 'multiplication', '/': 'division'}
    operation = next(op for op in ['+', '-', '*', '/'] if op in equation_str)
    operation_name = operation_map[operation]

    simple_description = f'An image of size {image_height}x{image_width} of the equation "{equation_str}" (with the {operation_name} operation)'
    
    if is_canonical:
        simple_description += ", the digits and characters are in an aligned canonical form"
    else:
        simple_description += ", the digits and characters have varied vertical positions and have non-uniform widths"
    
    if font_usage == 'mixed_full' or font_usage == 'mixed_limited':
        unique_fonts = set(font for font in char_details['fonts'] if font != 'other' and font != 'space' and font != 'not_found')
        num_unique_fonts = len(unique_fonts)
        if font_usage == 'mixed_full':
            simple_description += f", using many different mixed fonts ({num_unique_fonts})"
        else:
            simple_description += f", using a limited set of {num_unique_fonts} mixed fonts"
    else:
        simple_description += f", using a single font ({font_usage})"

    args = equation_str.split()
    additional_description = f"First argument has {len(args[0])} digits, "
    additional_description += f"second argument has {len(args[2])} digits, "
    additional_description += f"result has {len(args[4])} digits. "
    
    section_orig_names = ['argument1', 'operation', 'argument2', 'equal_sign', 'result']
    section_names_to_use = ['1st argument', 'operation symbol', '2nd argument', 'equal sign', 'result']

    for k, (section, name_to_use) in enumerate(zip(section_orig_names, section_names_to_use)):
        start = row[f'{section}_start']
        end = row[f'{section}_end']
        if k == 0:
            additional_description += f"{name_to_use} is at timepoints [{start}, {end}], "
        elif k == len(section_names_to_use) - 1:
            additional_description += f"and the {name_to_use} at [{start}, {end}]"
        else:
            additional_description += f"{name_to_use} is at [{start}, {end}], "

    return simple_description, additional_description
