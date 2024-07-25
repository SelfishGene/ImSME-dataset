import os
from kaggle.api.kaggle_api_extended import KaggleApi
from urllib.parse import urlparse

# instructions for setting up Kaggle API:
# https://www.kaggle.com/docs/api

# 1. pip install kaggle
# 2. Go to your kaggle account settings, click on 'Create New API Token'
# 3. Save the kaggle.json file to C:\Users\<username>\.kaggle\kaggle.json (Windows)
# 3. Save the kaggle.json file to ~/.kaggle/kaggle.json (Linux)

def download_ImSME_dataset(output_dir):
    # Initialize and authenticate Kaggle API
    api = KaggleApi()
    api.authenticate()

    # ImSME dataset URL
    dataset_url = "https://www.kaggle.com/datasets/selfishgene/imsme-images-of-simple-math-equations"

    try:
        # Extract dataset info from URL
        path = urlparse(dataset_url).path
        parts = path.strip('/').split('/')
        if len(parts) < 3 or parts[0] != 'datasets':
            raise ValueError(f"Invalid Kaggle dataset URL: {dataset_url}")
        
        owner, dataset_name = parts[1], parts[2]
        dataset = f"{owner}/{dataset_name}"
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Downloading dataset: {dataset}")
        api.dataset_download_files(dataset, path=output_dir, unzip=True)
        print(f"Dataset {dataset} downloaded successfully to {output_dir}")
    
    except Exception as e:
        print(f"Error downloading {dataset_url}: {str(e)}")

if __name__ == "__main__":
    # Output directory
    output_dir = "data"
    
    download_ImSME_dataset(output_dir)
