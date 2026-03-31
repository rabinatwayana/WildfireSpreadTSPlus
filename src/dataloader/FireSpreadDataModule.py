from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Subset, DataLoader
import glob
from .FireSpreadDataset import FireSpreadDataset
from typing import List, Optional, Union
import json

class FireSpreadDataModule(LightningDataModule):

    def __init__(self, data_dir: str, batch_size: int, n_leading_observations: int, n_leading_observations_test_adjustment: int,
                 crop_side_length: int,
                 load_from_hdf5: bool, num_workers: int, remove_duplicate_features: bool, 
                 is_pad: Optional[bool] = False,
                 features_to_keep: Union[Optional[List[int]], str] = None, return_doy: bool = False,
                 data_fold_id: int = 0, non_outlier_indices_path: Optional[str] = None, filter_ignition_train: Optional[bool] = False, filter_ignition_val_test: Optional[bool] = False,
                 ignition_only_train: Optional[bool] = False, ignition_only_val_test: Optional[bool] = False, additional_data: Optional[bool] = False, do_cross_year_experiment: Optional[bool] = False, cross_year_split_json_path: str  = "", cross_year_train_id:int = "", cross_year_eval_id: Optional[int] = None, predict_split: str = "val", *args, **kwargs):
        """_summary_ Data module for loading the WildfireSpreadTS dataset.

        Args:
            data_dir (str): _description_ Path to the directory containing the data.
            batch_size (int): _description_ Batch size for training and validation set. Test set uses batch size 1, because images of different sizes can not be batched together.
            n_leading_observations (int): _description_ Number of days to use as input observation. 
            n_leading_observations_test_adjustment (int): _description_ When increasing the number of leading observations, the number of samples per fire is reduced.
              This parameter allows to adjust the number of samples in the test set to be the same across several different values of n_leading_observations, 
              by skipping some initial fires. For example, if this is set to 5, and n_leading_observations is set to 1, the first four samples that would be 
              in the test set are skipped. This way, the test set is the same as it would be for n_leading_observations=5, thereby retaining comparability 
              of the test set.
            crop_side_length (int): _description_ The side length of the random square crops that are computed during training and validation.
            load_from_hdf5 (bool): _description_ If True, load data from HDF5 files instead of TIF. 
            num_workers (int): _description_ Number of workers for the dataloader.
            remove_duplicate_features (bool): _description_ Remove duplicate static features from all time steps but the last one. Requires flattening the temporal dimension, since after removal, the number of features is not the same across time steps anymore.
            features_to_keep (Union[Optional[List[int]], str], optional): _description_. List of feature indices from 0 to 39, indicating which features to keep. Defaults to None, which means using all features.
            return_doy (bool, optional): _description_. Return the day of the year per time step, as an additional feature. Defaults to False.
            data_fold_id (int, optional): _description_. Which data fold to use, i.e. splitting years into train/val/test set. Defaults to 0.
        """
        super().__init__()

        self.n_leading_observations_test_adjustment = n_leading_observations_test_adjustment
        self.data_fold_id = data_fold_id
        self.return_doy = return_doy
        # wandb apparently can't pass None values via the command line without turning them into a string, so we need this workaround
        self.features_to_keep = features_to_keep if type(
            features_to_keep) != str else None
        self.remove_duplicate_features = remove_duplicate_features
        self.num_workers = num_workers
        self.load_from_hdf5 = load_from_hdf5
        self.crop_side_length = crop_side_length
        self.n_leading_observations = n_leading_observations
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        self.is_pad=is_pad
        self.non_outlier_indices_path = non_outlier_indices_path
        self.filter_ignition_train = filter_ignition_train
        self.filter_ignition_val_test = filter_ignition_val_test
        self.ignition_only_train = ignition_only_train
        self.ignition_only_val_test = ignition_only_val_test
        self.additional_data = additional_data
        self.do_cross_year_experiment = do_cross_year_experiment
        self.cross_year_split_json_path = cross_year_split_json_path
        self.cross_year_train_id = cross_year_train_id
        self.cross_year_eval_id = cross_year_eval_id
        self.predict_split = predict_split
        self.predict_dataset = None


    def keep_ignition(self, dataset):
        ignition_indices = []
        total_samples = len(dataset)
        kept = 0
        
        for idx in range(total_samples):
            sample = dataset[idx]
            inputs = sample[0]  # Shape: [1, 7, 128, 128]
            x_af = inputs[:, -1, :, :]  # Active fire mask
            
            # Check current fire presence
            if torch.sum(x_af == 1) < 1:  # Original filtering condition
                ignition_indices.append(idx)
                kept += 1
    
        # Print detailed statistics
        print(f"Total samples: {total_samples}")
        print(f"Kept samples (ignition): {kept} ({kept/total_samples:.2%})")
        print(f"Discarded samples: {total_samples - kept} ({(total_samples - kept)/total_samples:.2%})")
        return Subset(dataset, ignition_indices)

    def filter_dataset(self, dataset):
        valid_indices = []
        total_samples = len(dataset)
        kept = 0
        for idx in range(total_samples):
            sample = dataset[idx]
            inputs = sample[0]  # Shape: [1, 7, 128, 128] if T=1; but [5*N, 128, 128] if T=5, where N is the number of features
            if len(inputs.shape) == 3:
                x_af = inputs[-1, :, :]
            else:
                x_af = inputs[:, -1, :, :]  # Active fire mask
            
            # Check current fire presence
            if torch.sum(x_af == 1) > 1:  # Original filtering condition
                valid_indices.append(idx)
                kept += 1
        
        # Print detailed statistics
        print(f"Total samples: {total_samples}")
        print(f"Kept samples (current fire): {kept} ({kept/total_samples:.2%})")
        print(f"Discarded samples: {total_samples - kept} ({(total_samples - kept)/total_samples:.2%})")
        
        return Subset(dataset, valid_indices)
        
    def _get_split_spec(self):
        train_event_ids, val_event_ids, test_event_ids=  None, None, None
        if self.do_cross_year_experiment:
            eval_year = self.cross_year_eval_id if self.cross_year_eval_id is not None else self.cross_year_train_id
            train_event_ids, val_event_ids, _ = self.split_within_year_fires(self.cross_year_train_id, self.cross_year_split_json_path)
            _, _, test_event_ids = self.split_within_year_fires(eval_year, self.cross_year_split_json_path)
            train_years=[self.cross_year_train_id]
            val_years= train_years
            test_years=[eval_year]
            print(
                "Using cross-year split:\n"
                f"Train year: {self.cross_year_train_id}, "
                f"Eval year: {eval_year}\n"
                f"Train events: {len(train_event_ids)}, "
                f"Val events: {len(val_event_ids)}, "
                f"Test events: {len(test_event_ids)}"
            )

        else:
            train_years, val_years, test_years = self.split_fires(
                self.data_fold_id, self.additional_data)
        return {
            "train_years": train_years,
            "val_years": val_years,
            "test_years": test_years,
            "train_event_ids": train_event_ids,
            "val_event_ids": val_event_ids,
            "test_event_ids": test_event_ids,
        }

    def _build_dataset(self, years, event_ids, is_train, test_adjustment, stats_years):
        return FireSpreadDataset(
            data_dir=self.data_dir,
            included_fire_years=years,
            include_event_ids=event_ids,
            n_leading_observations=self.n_leading_observations,
            n_leading_observations_test_adjustment=test_adjustment,
            crop_side_length=self.crop_side_length,
            load_from_hdf5=self.load_from_hdf5,
            is_train=is_train,
            remove_duplicate_features=self.remove_duplicate_features,
            features_to_keep=self.features_to_keep,
            return_doy=self.return_doy,
            stats_years=stats_years,
            is_pad=self.is_pad,
            do_cross_year_experiment=self.do_cross_year_experiment,
        )

    def setup(self, stage=None):
        split_spec = self._get_split_spec()
        train_years = split_spec["train_years"]
        val_years = split_spec["val_years"]
        test_years = split_spec["test_years"]
        train_event_ids = split_spec["train_event_ids"]
        val_event_ids = split_spec["val_event_ids"]
        test_event_ids = split_spec["test_event_ids"]

        self.predict_dataset = None

        need_train = stage in (None, "fit")
        need_val = stage in (None, "fit", "validate")
        need_test = stage in (None, "test")
        need_predict = stage in (None, "predict")

        if need_train:
            self.train_dataset = self._build_dataset(
                train_years, train_event_ids, True, None, train_years
            )
            if self.non_outlier_indices_path is not None:
                non_outlier_indices = np.load(self.non_outlier_indices_path).tolist()
                print(f"Subsetting train_loader using {self.non_outlier_indices_path}")
                self.train_dataset = Subset(self.train_dataset, non_outlier_indices)
            if self.filter_ignition_train:
                self.train_dataset = self.filter_dataset(self.train_dataset)
            if self.ignition_only_train:
                self.train_dataset = self.keep_ignition(self.train_dataset)

        if need_val:
            self.val_dataset = self._build_dataset(
                val_years, val_event_ids, False, None, train_years
            )
            if self.filter_ignition_val_test:
                self.val_dataset = self.filter_dataset(self.val_dataset)
            if self.ignition_only_val_test:
                self.val_dataset = self.keep_ignition(self.val_dataset)

        if need_test or (need_predict and self.predict_split == "test"):
            self.test_dataset = self._build_dataset(
                test_years,
                test_event_ids,
                False,
                self.n_leading_observations_test_adjustment,
                train_years,
            )
            if self.filter_ignition_val_test:
                self.test_dataset = self.filter_dataset(self.test_dataset)
            if self.ignition_only_val_test:
                self.test_dataset = self.keep_ignition(self.test_dataset)

        if need_predict:
            if self.predict_split == "train":
                if self.train_dataset is None:
                    self.train_dataset = self._build_dataset(
                        train_years, train_event_ids, False, None, train_years
                    )
                self.predict_dataset = self.train_dataset
            elif self.predict_split == "test":
                if self.test_dataset is None:
                    self.test_dataset = self._build_dataset(
                        test_years,
                        test_event_ids,
                        False,
                        self.n_leading_observations_test_adjustment,
                        train_years,
                    )
                self.predict_dataset = self.test_dataset
            else:
                if self.val_dataset is None:
                    self.val_dataset = self._build_dataset(
                        val_years, val_event_ids, False, None, train_years
                    )
                self.predict_dataset = self.val_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def predict_dataloader(self):
        if self.predict_dataset is None:
            raise RuntimeError(
                f"predict_dataset is not initialized. Call setup('predict') first. Current predict_split={self.predict_split}"
            )
        return DataLoader(self.predict_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    @staticmethod
    def split_fires(data_fold_id, additional_data):
        """_summary_ Split the years into train/val/test set.

        Args:
            data_fold_id (_type_): _description_ Index of the respective split to choose, see method body for details.

        Returns:
            _type_: _description_
        """
        if not additional_data:

            folds = [(2018, 2019, 2020, 2021),
                 (2018, 2019, 2021, 2020),
                 (2018, 2020, 2019, 2021),
                 (2018, 2020, 2021, 2019),
                 (2018, 2021, 2019, 2020),
                 (2018, 2021, 2020, 2019),
                 (2019, 2020, 2018, 2021),
                 (2019, 2020, 2021, 2018),
                 (2019, 2021, 2018, 2020),
                 (2019, 2021, 2020, 2018),
                 (2020, 2021, 2018, 2019),
                 (2020, 2021, 2019, 2018)]
            train_years = list(folds[data_fold_id][:2])
            val_years = list(folds[data_fold_id][2:3])
            test_years = list(folds[data_fold_id][3:4])
        
        else:
            folds = [(2016, 2017, 2020, 2021, 2018, 2019, 2022, 2023),
                 (2018, 2019, 2022, 2023, 2020, 2021, 2016, 2017),
                 (2016, 2017, 2020, 2021, 2022, 2023, 2018, 2019),
                 (2018, 2019, 2022, 2023, 2016, 2017, 2020, 2021)]
            train_years = list(folds[data_fold_id][:4])
            val_years = list(folds[data_fold_id][4:6])
            test_years = list(folds[data_fold_id][6:8])

        print(
            f"Using the following dataset split:\nTrain years: {train_years}, Val years: {val_years}, Test years: {test_years}")

        return train_years, val_years, test_years
    
    @staticmethod
    
    def split_within_year_fires(year_id, json_path):
        """
        Load train/val/test event IDs from a JSON file for a specific year.

        Args:
            year_id (int or str): The year to get splits for, e.g., 2016
            json_path (str or Path): Path to the JSON file

        Returns:
            train_event_ids (list): List of training event IDs for the year
            val_event_ids (list): List of validation event IDs for the year
            test_event_ids (list): List of test event IDs for the year
        """
        json_path = Path(json_path)
        if not json_path.is_file():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        with open(json_path, "r") as f:
            data = json.load(f)

        # Convert year_id to string since JSON keys are strings
        year_key = str(year_id)

        train_event_ids = data.get("train_event_ids", {}).get(year_key, [])
        val_event_ids   = data.get("val_event_ids", {}).get(year_key, [])
        test_event_ids  = data.get("test_event_ids", {}).get(year_key, [])

        return train_event_ids, val_event_ids, test_event_ids
