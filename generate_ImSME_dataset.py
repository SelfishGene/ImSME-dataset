import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from utils import equation_to_image, generate_indicators, indicators_to_images, generate_equation_descriptions
from config import (CHAR_ORDER, SECTION_ORDER, FONT_ORDER, OPERATIONS, SECTION_NAME_MAPPING)


def generate_and_save_images(df, num_equations, num_samples_per_equation, output_folder, char_images_folder, 
                             equations_df_name, space_padding_range, fraction_canonical, 
                             fraction_single_font, fraction_limited_font_set):
    
    # Create output folders
    equation_images_folder = os.path.join(output_folder, 'equation_images')
    label_images_folder = os.path.join(output_folder, 'label_images')
    os.makedirs(equation_images_folder, exist_ok=True)
    os.makedirs(label_images_folder, exist_ok=True)

    # Perform stratified sampling
    sampled_rows = pd.DataFrame()
    samples_per_operation = num_equations // len(OPERATIONS)
    for operation in OPERATIONS:
        operation_df = df[df['operation'] == operation]
        if len(operation_df) <= samples_per_operation:
            sampled_rows = pd.concat([sampled_rows, operation_df])
        else:
            sampled_rows = pd.concat([sampled_rows, operation_df.sample(n=samples_per_operation, replace=False)])

    sampled_rows = sampled_rows.sample(frac=1).reset_index(drop=True)

    new_rows = []
    num_digits_equation = len(str(len(sampled_rows) - 1))
    num_digits_sample = len(str(num_samples_per_equation))

    for equation_index, row in tqdm(sampled_rows.iterrows(), total=len(sampled_rows), desc="Generating equation images"):
        for sample_index in range(1, num_samples_per_equation + 1):
            new_row = row.copy()
            new_row['equation_index'] = equation_index
            new_row['sample_index'] = sample_index
            
            current_fraction_canonical = fraction_canonical * 2 if sample_index in [1, 2, 3] else fraction_canonical
            current_fraction_single_font = fraction_single_font * 2 if sample_index in [1, 2, 3] else fraction_single_font
            
            is_canonical = random.random() < current_fraction_canonical
            new_row['is_canonical'] = is_canonical

            rand_value = random.random()
            if rand_value < current_fraction_single_font:
                selected_font = random.choice(FONT_ORDER)
                fonts_pool = [selected_font]
                new_row['font_usage'] = selected_font
            elif rand_value < (current_fraction_single_font + fraction_limited_font_set):
                num_limited_fonts = random.randint(4, len(FONT_ORDER))
                fonts_pool = FONT_ORDER[:num_limited_fonts]
                new_row['font_usage'] = 'mixed_limited'
            else:
                fonts_pool = None
                new_row['font_usage'] = 'mixed_full'

            equation_string = row['full string']
            if is_canonical:
                equation_string = f"    {equation_string}    "
            else:
                equation_string = f"{' ' * random.randint(*space_padding_range)}{equation_string}{' ' * random.randint(*space_padding_range)}"

            equation_image, char_details = equation_to_image(equation_string, char_images_folder, 
                                                             use_random_augmentations=not is_canonical,
                                                             fonts_pool=fonts_pool)

            char_indicators, font_indicators, section_indicators = generate_indicators(char_details)

            for original_name, new_name in SECTION_NAME_MAPPING.items():
                indicator = section_indicators[original_name]
                non_zero = np.nonzero(indicator)[0]
                if len(non_zero) > 0:
                    start = non_zero[0]
                    end = non_zero[-1] + 1
                    new_row[f'{new_name}_start'] = start
                    new_row[f'{new_name}_end'] = end

            char_image, font_image, section_image = indicators_to_images(char_indicators, font_indicators, section_indicators,
                                                                         CHAR_ORDER, SECTION_ORDER, FONT_ORDER)

            total_height = char_image.height + section_image.height + font_image.height
            combined_label_image = Image.new('L', (equation_image.width, total_height))
            combined_label_image.paste(char_image, (0, 0))
            combined_label_image.paste(section_image, (0, char_image.height))
            combined_label_image.paste(font_image, (0, char_image.height + section_image.height))

            file_name = f"equation_{equation_index:0{num_digits_equation}d}_sample_{sample_index:0{num_digits_sample}d}.png"
            equation_image.save(os.path.join(equation_images_folder, file_name), 'PNG', compress_level=8)
            combined_label_image.save(os.path.join(label_images_folder, file_name), 'PNG', compress_level=8)
            
            new_row['image_filename'] = file_name

            simple_desc, additional_desc = generate_equation_descriptions(new_row, char_details, equation_image.height, equation_image.width)
            new_row['simple_description'] = simple_desc
            new_row['additional_description'] = additional_desc

            new_rows.append(new_row)

    new_df = pd.DataFrame(new_rows)

    columns = new_df.columns.tolist()
    for col in ['is_canonical', 'font_usage', 'sample_index', 'equation_index', 'image_filename']:
        columns.remove(col)
    new_column_order = ['image_filename', 'equation_index', 'sample_index', 'font_usage', 'is_canonical'] + columns
    new_df = new_df[new_column_order]

    actual_samples = len(new_df)
    new_filename = f"simple_math_equation_images__{equations_df_name.split('__')[1]}__{actual_samples}_rows.csv"
    new_df.to_csv(os.path.join(output_folder, new_filename), index=False)

    print('-' * 80)
    print(f'Image generation completed! \nNew dataframe saved as "{new_filename}"')
    print(f"Total number of equations processed: {len(sampled_rows)}")
    print(f"Total number of samples generated: {actual_samples}")
    print('-' * 40)
    print(f"Operation distribution in the generated dataset:")
    print(new_df['operation'].value_counts())
    print('-' * 40)
    print(f"Canonical vs Augmented distribution:")
    print(new_df['is_canonical'].value_counts(normalize=True))
    print('-' * 40)
    print(f"Font usage distribution:")
    print(new_df['font_usage'].value_counts(normalize=True))
    print('-' * 80)

    return new_df


if __name__ == "__main__":
    # if just_a_test is True, the script will generate a small dataset for testing purposes
    just_a_test = True

    # Dataset generation parameters
    OUTPUT_HEIGHT = 128
    RESIZE_HEIGHT_DIGIT = 112
    RESIZE_HEIGHT_SYMBOL = 72
    WIDTH_MULT_FACTORS = [0.25, 1.0]
    SPACE_PADDING_RANGE = (4, 7)
    FRACTION_CANONICAL = 0.2
    FRACTION_SINGLE_FONT = 0.2
    FRACTION_LIMITED_FONT_SET = 0.2

    # Folder paths
    equations_dataframe_folder = "data/simple_math_equations_dataset"
    char_images_folder = "data/char_images_dataset_tight"
    output_folder = "data/math_equations_images_dataset"

    # Choose the equations database to use
    equations_df_name = "simple_math_equations__4d_op_3d_eq_3d__1574610_rows.csv"
    
    # Load the equations dataframe
    df = pd.read_csv(os.path.join(equations_dataframe_folder, equations_df_name))

    # Generate datasets of different sizes
    if just_a_test:
        dataset_sizes = {
            "tiny": (4, 4),   # 4 equations, 4 samples each
            "small": (8, 4),  # 8 equations, 4 samples each
            "medium": (16, 4) # 16 equations, 4 samples each
        }
    else:
        dataset_sizes = {
            "tiny": (2048, 8),   # 2048 equations, 8 samples each
            "small": (8192, 8),  # 8192 equations, 8 samples each
            "medium": (32768, 8) # 32768 equations, 8 samples each
        }

    for size_name, (num_equations, num_samples) in dataset_sizes.items():
        print('-' * 80)
        print(f"\nGenerating {size_name} dataset...")
        print('-' * 30)
        size_output_folder = f"{output_folder}_{size_name}"
        generate_and_save_images(df, num_equations, num_samples, size_output_folder, char_images_folder, equations_df_name,
                                 space_padding_range=SPACE_PADDING_RANGE, fraction_canonical=FRACTION_CANONICAL, 
                                 fraction_single_font=FRACTION_SINGLE_FONT, fraction_limited_font_set=FRACTION_LIMITED_FONT_SET)
        print('-' * 80)

    print("\nImSME dataset generation completed.")
