"""IMPORT PACKAGES"""
import zipfile
from pathlib import Path
from typing import Callable
from typing import Tuple

import torch
from PIL import Image
from torch.utils.data import get_worker_info

import numpy as np

import os

import torchvision

"""DATALOADER FOR .ZIP FILES"""


class ZipDataset(torch.utils.data.Dataset):
    """
    Custom PyTorch Dataset class for loading images from a zip file.

    Args:
        transform (Callable): A callable object (e.g., a torchvision transform) to apply to the loaded images.
        zip_path (Path): The path to the zip file containing the images.
        depth_path (Path): The path to the depth labels
        image_suffix (str): The file suffix (e.g., ".jpg") that valid image files should have.

    Attributes:
        transform (Callable): The provided image transformation function.
        zip_path (Path): The path to the zip file.
        images (list): A list of valid image file names within the zip file.
        image_folder_members (dict): A dictionary mapping image file names to corresponding ZipInfo objects.

    Methods:
        __len__(self): Returns the length of the dataset.
        __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]: Returns an image and a dummy label for a given index.

    Static Methods:
        _valid_member(m: zipfile.ZipInfo, image_suffix: str) -> bool: Checks if a member is a valid image file.
    """
    def __init__(
            self,
            transform: Callable,
            depth_transform: Callable,
            zip_path: Path,
            depth_path: Path,
            image_suffix: str,
            train_student: bool,
    ):

        # Assign variables
        self.transform = transform
        self.depth_transform = depth_transform
        self.zip_path = zip_path
        self.depth_path = depth_path
        self.images = []
        self.depths = []
        self.image_suffix = image_suffix
        self.train_student = train_student
        # Load the zip file
        image_zip = zipfile.ZipFile(self.zip_path)
        depth_zip = zipfile.ZipFile(self.depth_path)

        # Get the members of the zip file
        self.image_folder_members = {
            str(Path(m.filename)): m
            for m in sorted(image_zip.infolist(), key=lambda x: x.filename)
        }

        self.depth_folder_members = {
            str(Path(m.filename)): m
            for m in sorted(depth_zip.infolist(), key=lambda x: x.filename)
        }

        # Get the image names from the zip file, check whether they are valid
        for image_name, m in self.image_folder_members.items():
            if not self._valid_member(
                    m, image_suffix
            ):
                continue
            self.images.append(image_name)
        for depth_name, m in self.depth_folder_members.items():
            if depth_name.endswith('.npy'):
                self.depths.append(depth_name)

    @staticmethod
    def _valid_member(
            m: zipfile.ZipInfo,
            image_suffixes: list,
    ):
        """Returns True if the member is valid based on the list of suffixes"""
        return (
                any(m.filename.endswith(suffix) for suffix in image_suffixes)
                and not m.is_dir()
        )
    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.images)

    def get_sample(self, index: int) -> Tuple[torch.Tensor, torch.Tensor,str]:
        """Returns the items for a dataloader object"""
        do_flip = np.random.rand()<0.5 and self.train_student
        # Open the zip file
        with zipfile.ZipFile(self.zip_path) as image_zip:
            fn = self.image_folder_members[self.images[index]].filename
            # Open the image data from the zip file based on index
            with image_zip.open(
                    fn
            ) as image_file:
                image = Image.open(image_file).convert("RGB")
            full_file_path = os.path.join(self.zip_path,fn)

        

        # Return depth labels        
        with zipfile.ZipFile(self.depth_path) as depth_zip:
            fn = self.depth_folder_members[self.depths[index]].filename
            # Open the image data from the zip file based on index
            with depth_zip.open(
                    fn
            ) as depth_file:
                label = torch.from_numpy(np.load(depth_file)).unsqueeze(0)
            full_file_path_depth = os.path.join(self.depth_path,fn)
        
        # Apply torchvision transforms if defined
        if self.transform:
            image = self.transform(image)
        if self.depth_transform:
            label = self.depth_transform(label)
        if do_flip:
            image = torchvision.transforms.functional.hflip(image)
            label = torchvision.transforms.functional.hflip(label)
        return image, label, full_file_path,full_file_path_depth
    
    def __getitem__(self,index:int):
        if self.train_student:
            return [self.get_sample(index),self.get_sample(index)]
        else:
            return self.get_sample(index)



"""FUNCTION FOR CONCATENATING .ZIP DATASETS"""

def concat_zip_datasets(
        parent_folder: str,
        depth_path: str,
        transform: Callable,
        depth_transform: Callable,
        image_suffix: list = ['.png', 'jpg'],
        datasets: list = None,
        train_student: bool = False
):
    """
        Concatenates multiple ZipDatasets into a single ConcatDataset.

        Args:
            parent_folder (str): The path to the parent folder containing multiple zip files to be combined.
            transform (Callable): A callable object (e.g., a torchvision transform) to apply to the loaded images.
            image_suffix (str, optional): The file suffix (e.g., ".jpg") that valid image files should have.
                Defaults to '.png'.

        Returns:
            torch.utils.data.ConcatDataset: A ConcatDataset containing all the ZipDatasets.

        Note:
            To use this function, provide the path to the parent folder containing the zip files you want to combine.
            You can also specify a custom image_suffix and transformation function.
        """

    # Create list of zip folders
    zip_folders = list(Path(parent_folder).iterdir())
    zip_folders_depth = list(Path(depth_path).iterdir())
    included_folders = []
    included_folders_depth = []
    if datasets is not None:
        for dataset in datasets:
            for zip_folder in zip_folders:
                if dataset in zip_folder.name:
                    included_folders.append(zip_folder)
            for zip_folder in zip_folders_depth:
                if dataset in zip_folder.name:
                    included_folders_depth.append(zip_folder)
    else:
        for zip_folder in zip_folders: # Only append zipfiles to the included folder list (not any unzipped folders or orther files)
            if zipfile.is_zipfile(zip_folder):
                included_folders.append(zip_folder)
        for zip_folder in zip_folders_depth:
            if zipfile.is_zipfile(zip_folder):
                included_folders_depth.append(zip_folder)


    # print(included_folders,included_folders_depth)
    # Construct datasets for each zip folder
    dataset = [
        ZipDataset(
            transform=transform,
            depth_transform = depth_transform,
            zip_path=included_folders[i],
            depth_path = included_folders_depth[i],
            image_suffix=image_suffix,
            train_student=train_student)
        for i in range(len(included_folders))
    ]

    # Concatenate the datasets
    dataset = torch.utils.data.ConcatDataset(dataset)

    return dataset