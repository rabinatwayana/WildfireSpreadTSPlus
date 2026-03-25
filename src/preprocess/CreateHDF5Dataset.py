"""
Command to execute this script:

python src/preprocess/CreateHDF5Dataset.py \
    --data_dir /Users/rabinatwayana/2_CDE_MT/CDE_Master_Thesis_Wildfire_CL/data/raw/WSTS_subset \
    --target_dir /Users/rabinatwayana/2_CDE_MT/CDE_Master_Thesis_Wildfire_CL/data/raw/WSTS_subset_hdf5 \
    --years 2016 2017 2018 2019 2020 2021 2022 2023

Note: latlng is currently disabled
"""

import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# from src.dataloader.FireSpreadDataset import FireSpreadDataset
from pathlib import Path
import h5py
from tqdm import tqdm
import rasterio
import numpy as np
import glob
import time

def hdf5_dataset_generator(data_dir, year):
    fire_dirs = sorted(glob.glob(f"{data_dir}/{year}/*/"))
    for fire_dir in fire_dirs:
        fire_name = Path(fire_dir).name
        img_files = sorted(glob.glob(f"{fire_dir}/*.tif"))

        imgs = []
        lnglat = None
        for img_path in img_files:
            with rasterio.open(img_path, "r") as ds:
                imgs.append(ds.read())
                # if lnglat is None:
                #     lnglat = ds.lnglat()

        x = np.stack(imgs, axis=0)
        img_dates = [Path(p).name.split("_")[0].replace(".tif", "") for p in img_files]

        x[:, -1, ...] = np.nan_to_num(x[:, -1, ...], nan=0)
        x[:, -1, ...] = np.floor_divide(x[:, -1, ...], 100)

        yield year, fire_name, img_dates, lnglat, x

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str,
                    help="Path to dataset directory", required=True)
parser.add_argument("--target_dir", type=str,
                    help="Path to directory where the HDF5 files should be stored", required=True)
parser.add_argument("--years", type=int, nargs='+', required=True,
                    help="List of years to process, e.g. --years 2016 2017 2018")
args = parser.parse_args()

# Need to prevent some error with HDF5 files being locked and thereby inaccessible
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# years = [2016,2017,2018, 2019, 2020, 2021,2022,2023]
years = args.years

# dataset = FireSpreadDataset(data_dir=args.data_dir,
#                             included_fire_years=years,
#                             # the following args are irrelevant here, but need to be set
#                             n_leading_observations=1, crop_side_length=128, load_from_hdf5=False, is_train=True,
#                             remove_duplicate_features=False, stats_years=(2018, 2020))
# data_gen = dataset.get_generator_for_hdf5()

# base_dir = "/tmp"
# for y in [2018, 2019, 2020, 2021]:
for y in years:
    start = time.perf_counter()
    target_dir = f"{args.target_dir}/{y}"
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    data_gen = hdf5_dataset_generator(args.data_dir, y) # RT: discarding get_generator_for_hdf5 of FireSpreadDataset

    for year, fire_name, img_dates, lnglat, imgs in tqdm(data_gen):
        # print(year)
        # For some reason creating HDF5 files directly where we want them doesn't work
        # year_dir = f"{base_dir}/{year}/"
        # Path(year_dir).mkdir(parents=True, exist_ok=True)
        # h5_path = year_dir + f"{fire_name}.hdf5"
        target_dir = f"{args.target_dir}/{year}"
        h5_path = f"{target_dir}/{fire_name}.hdf5"

        if Path(h5_path).is_file():
            print(f"File {h5_path} already exists, skipping...")
            continue

        with h5py.File(h5_path, "w") as f:
            dset = f.create_dataset("data", imgs.shape, data=imgs)
            dset.attrs["year"] = year
            dset.attrs["fire_name"] = fire_name
            dset.attrs["img_dates"] = img_dates
            # dset.attrs["lnglat"] = lnglat
    end = time.perf_counter()
    elapsed_time=end-start
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"Time taken for year {y}: {int(minutes)}m {seconds:.2f}s")