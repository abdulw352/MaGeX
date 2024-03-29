import os
import shutil
import requests
import pandas as pd
from tqdm import tqdm
from pymatgen import Structure, Composition

# Data collection
def download_data(url, destination):
    """Download data from the given URL and save it to the destination folder."""
    filename = url.split('/')[-1]
    file_path = os.path.join(destination, filename)

    if not os.path.exists(destination):
        os.makedirs(destination)

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kilobyte

    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(file_path, 'wb') as file:
        for data in response.iter_content(block_size):
            if data:
                progress_bar.update(len(data))
                file.write(data)
    progress_bar.close()

    if total_size != 0 and progress_bar.n != total_size:
        print("Error: Download interrupted!")

# Data sources
materials_project_url = "https://example.com/materials_project_data.zip"
oqmd_url = "https://example.com/oqmd_data.csv"
aflow_url = "https://example.com/aflow_data.tar.gz"

# Download data
download_data(materials_project_url, "data")
download_data(oqmd_url, "data")
download_data(aflow_url, "data")

# Data preprocessing
def preprocess_data(data_dir):
    """Preprocess the data from various sources."""
    all_data = []

    # Process Materials Project data
    materials_project_data = pd.read_csv(os.path.join(data_dir, "materials_project_data.csv"))
    materials_project_data = materials_project_data[["formula", "space_group", "stability"]]
    all_data.append(materials_project_data)

    # Process OQMD data
    oqmd_data = pd.read_csv(os.path.join(data_dir, "oqmd_data.csv"))
    oqmd_data = oqmd_data[["composition", "space_group_number", "formation_energy_per_atom"]]
    all_data.append(oqmd_data)

    # Process AFLOW data
    aflow_data = pd.read_csv(os.path.join(data_dir, "aflow_data.csv"))
    aflow_data = aflow_data[["compound", "space_group_number", "energy_per_atom"]]
    all_data.append(aflow_data)

    # Combine and clean data
    combined_data = pd.concat(all_data, ignore_index=True)
    combined_data.dropna(inplace=True)
    combined_data.reset_index(drop=True, inplace=True)

    # Save preprocessed data
    combined_data.to_csv("preprocessed_data.csv", index=False)

preprocess_data("data")
