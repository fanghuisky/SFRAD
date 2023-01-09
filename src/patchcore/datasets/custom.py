import os
import pdb
from enum import Enum

import PIL
import torch
from torchvision import transforms

class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class CustomDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MVTec.
    """

    def __init__(
        self,
        source,
        classname,
        resize=(200,360),
        imagesize=(3, 200, 360),
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        _CLASSNAMES=[],
        **kwargs,
    ):
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
        self.train_val_split = train_val_split
        self.imagesize = imagesize
        self.transform_mean = [0.485, 0.456, 0.406]
        self.transform_std = [0.229, 0.224, 0.225]
        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()
        if(resize==imagesize[1:]:
            self.transform_img = [
                transforms.Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.transform_mean, std=self.transform_std),
            ]
            self.transform_mask = [
                transforms.Resize(resize),
                transforms.ToTensor(),
            ]
        else:
            self.transform_img = [
                transforms.Resize(resize),
                transform.CenterCrop(imagesize[1:]),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.transform_mean, std=self.transform_std),
            ]
            self.transform_mask = [
                transforms.Resize(resize),
                transforms.CenterCrop(imagesize[1:]),
                transforms.ToTensor(),
            ]
        
        self.transform_img = transforms.Compose(self.transform_img)
        self.transform_mask = transforms.Compose(self.transform_mask)


    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
        else:
            if('mvtec' in mask_path):
                mask = torch.zeros([1, *image.size()[1:]])
            else:
                mask = torch.zeros([3, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, self.split.value)
            maskpath = os.path.join(self.source, classname, "ground_truth")
            anomaly_types = os.listdir(classpath)

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                anomaly_files = sorted(os.listdir(anomaly_path))
                imgpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_path, x) for x in anomaly_files
                ]

                if self.train_val_split < 1.0:
                    n_images = len(imgpaths_per_class[classname][anomaly])
                    train_val_split_idx = int(n_images * self.train_val_split)
                    if self.split == DatasetSplit.TRAIN:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][:train_val_split_idx]
                    elif self.split == DatasetSplit.VAL:
                        imgpaths_per_class[classname][anomaly] = imgpaths_per_class[
                            classname
                        ][anomaly][train_val_split_idx:]

                if self.split == DatasetSplit.TEST and anomaly != "good":
                    anomaly_mask_path = os.path.join(maskpath, anomaly)
                    anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                    maskpaths_per_class[classname][anomaly] = [
                        os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
                    ]
                else:
                    maskpaths_per_class[classname]["good"] = None

        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)
        return imgpaths_per_class, data_to_iterate
