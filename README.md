# ImSME - Images of Simple Math Equations Dataset

ImSME is a dataset of wide images containing simple math equations with detailed annotations. The annotations are both detailed textual descriptions as well as precise pixel coordinates of each equation part (at the digits/symbols, and arguments/operations hirarchies).  
This repository contains the scripts used to generate the dataset and perform basic exploratory data analysis.

## Dataset Description

- Images of simple math equations with varied widths and a fixed height of 128 pixels.
- Equations are of the form "{argument1} {operation} {argument2} {equal_sign} {result}".
- Characters are rendered using random fonts, sometimes with different fonts for each character.
- Detailed textual descriptions are provided for each equation, including information about fonts and character locations.
- Multiple versions of the dataset are available: tiny (~250MB), small (~1GB), and medium (~4GB).

## Repository Contents

1. `config.py`: Configuration parameters and fixed data dictionaries.
2. `utils.py`: Utility functions used across multiple scripts.
3. `create_char_images_dataset.py`: Script to create the character images dataset.
4. `create_equations_dataframe.py`: Script to create the equations dataframe.
5. `generate_ImSME_dataset.py`: Script to generate equation images from the equations dataframe.
6. `download_ImSME_dataset.py`: Script to download the ImSME dataset from Kaggle.
7. `explore_ImSME_dataset.ipynb`: Jupyter notebook for basic exploratory data analysis of the dataset.
8. `requirements.txt`: List of required Python packages.

## Requirements

To install the required packages, run:

```
pip install -r requirements.txt
```

## Usage

You have two options: download the pre-generated dataset or generate the dataset from scratch.

### Option 1: Download and Explore the Pre-generated Dataset

1. Download the ImSME dataset:
   ```
   python download_ImSME_dataset.py
   ```

2. Explore the dataset:
   Open and run the `explore_ImSME_dataset.ipynb` notebook in Jupyter or any compatible environment.

### Option 2: Generate the Dataset from Scratch

1. Create the character images dataset:
   ```
   python create_char_images_dataset.py
   ```

2. Create the equations dataframe:
   ```
   python create_equations_dataframe.py
   ```

3. Generate ImSME dataset:
   ```
   python generate_ImSME_dataset.py
   ```

4. Explore the dataset:
   Open and run the `explore_ImSME_dataset.ipynb` notebook in Jupyter or any compatible environment.

## Dataset Structure

- `math_equations_images_dataset_{tiny/small/medium}/`
  - `equation_images/`: Folder containing equation images
  - `label_images/`: Folder containing label images with pixel-precise annotations
  - `simple_math_equation_images__*.csv`: CSV file with detailed information about each image

- `simple_math_equations_dataset/`: Contains CSV files with equations in a convenient format
- `char_images_dataset/`: Contains 10,664 black and white character images (128x128)
- `char_images_dataset_tight/`: Similar to `char_images_dataset`, but with tighter cropping around the characters

## Citation

If you use this dataset in your research, please cite it as follows:

```
@dataset{ImSME2024,
  title = {ImSME: Images of Simple Math Equations Dataset},
  author = {David Beniaguev},
  year = {2024},
  url={https://github.com/SelfishGene/ImSME-dataset},
  DOI={10.34740/KAGGLE/DSV/9026395},
  publisher = {GitHub},
}
```

## License

MIT License