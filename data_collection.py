import os
import requests
from tqdm import tqdm
import tarfile
import zipfile
import pandas as pd

def download_file(url, destination_folder):
    """
    Download a file from the given URL and save it to the specified destination folder.
    """
    filename = url.split('/')[-1]
    file_path = os.path.join(destination_folder, filename)

    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Download the file
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

    with open(file_path, 'wb') as file:
        for data in response.iter_content(block_size):
            if data:
                progress_bar.update(len(data))
                file.write(data)

    progress_bar.close()

    if total_size != 0 and progress_bar.n != total_size:
        print(f"Error: Download of {filename} was interrupted.")

def extract_file(file_path, destination_folder):
    """
    Extract a compressed file (ZIP or TAR.GZ) to the specified destination folder.
    """
    filename = os.path.basename(file_path)
    if filename.endswith('.zip'):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(destination_folder)
    elif filename.endswith('.tar.gz'):
        with tarfile.open(file_path, 'r:gz') as tar_ref:
            tar_ref.extractall(destination_folder)
    else:
        print(f"Unsupported file format: {filename}")

def collect_materials_project_data(destination_folder):
    """
    Collect data from the Materials Project database.
    """
    materials_project_url = "https://example.com/materials_project_data.zip"
    download_file(materials_project_url, destination_folder)
    file_path = os.path.join(destination_folder, materials_project_url.split('/')[-1])
    extract_file(file_path, destination_folder)

def collect_oqmd_data(destination_folder):
    """
    Collect data from the OQMD (Open Quantum Materials Database).
    """
    oqmd_url = "https://example.com/oqmd_data.csv"
    download_file(oqmd_url, destination_folder)

def collect_aflow_data(destination_folder):
    """
    Collect data from the AFLOW (Automatic Flow) database.
    """
    aflow_url = "https://example.com/aflow_data.tar.gz"
    download_file(aflow_url, destination_folder)
    file_path = os.path.join(destination_folder, aflow_url.split('/')[-1])
    extract_file(file_path, destination_folder)

def collect_matmatch_data(destination_folder):
    """
    Collect data from the Matmatch.com database.
    """
    matmatch_url = "https://example.com/matmatch_data.zip"
    download_file(matmatch_url, destination_folder)
    file_path = os.path.join(destination_folder, matmatch_url.split('/')[-1])
    extract_file(file_path, destination_folder)

if __name__ == "__main__":
    data_folder = "raw_data"

    # Collect data from different sources
    collect_materials_project_data(data_folder)
    collect_oqmd_data(data_folder)
    collect_aflow_data(data_folder)
    collect_matmatch_data(data_folder)

    # Optionally, you can load and inspect the collected data
    materials_project_data = pd.read_csv(os.path.join(data_folder, "materials_project_data.csv"))
    print(f"Materials Project data shape: {materials_project_data.shape}")
