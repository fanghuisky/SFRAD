#coding:utf-8
import contextlib
import logging
import os
import sys
import pdb
import click
import numpy as np
import torch
import yaml

import argparse
import patchcore.backbones
import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils

from util.utils import updataconfig, metric_result

LOGGER = logging.getLogger(__name__)

_DATASETS = {"custom": ["patchcore.datasets.custom", "CustomDataset"],}

def get_sampler(args, device):
    if args.sampler_name == "identity":
        return patchcore.sampler.IdentitySampler()
    elif args.sampler_name == "greedy_coreset":
        return patchcore.sampler.GreedyCoresetSampler(args.percentage, device)
    elif args.sampler_name == "approx_greedy_coreset":
        return patchcore.sampler.ApproximateGreedyCoresetSampler(args.percentage, device)
    elif args.sampler_name == "basis_feature_sampling":
        return patchcore.sampler.Basis_Feature_Sampling(args.percentage, device)

def get_dataloaders(args):
    dataset_info = _DATASETS["custom"]   ### !
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])
    dataloaders = []
    for subdataset in args._CLASSNAMES:
        train_dataset = dataset_library.__dict__[dataset_info[1]](
            args.data_path,
            classname=subdataset,
            resize=args.resize,
            train_val_split=args.train_val_split,
            imagesize=args.imagesize,
            split=dataset_library.DatasetSplit.TRAIN,
            seed=args.seed,
            augment=args.augment,
            _CLASSNAMES=args._CLASSNAMES,
        )

        test_dataset = dataset_library.__dict__[dataset_info[1]](
            args.data_path,
            classname=subdataset,
            resize=args.resize,
            imagesize=args.imagesize,
            split=dataset_library.DatasetSplit.TEST,
            seed=args.seed,
            _CLASSNAMES=args._CLASSNAMES,
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        train_dataloader.name = args.dataset
        if subdataset is not None:
            train_dataloader.name += "_" + subdataset

        if args.train_val_split < 1:
            val_dataset = dataset_library.__dict__[dataset_info[1]](
                args.data_path,
                classname=subdataset,
                resize=args.resize,
                train_val_split=args.train_val_split,
                imagesize=args.imagesize,
                split=dataset_library.DatasetSplit.VAL,
                seed=args.seed,
                _CLASSNAMES=args._CLASSNAMES,
            )

            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
        else:
            val_dataloader = None
        dataloader_dict = {
            "training": train_dataloader,
            "validation": val_dataloader,
            "testing": test_dataloader,
        }

        dataloaders.append(dataloader_dict)
    return dataloaders


def get_patchcore(args, input_shape, sampler, device):
    backbone_names = list(args.backbone_names)

    loaded_patchcores = []
    for backbone_name in backbone_names:
        backbone_seed = None
        if ".seed-" in backbone_name:
            backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(
                backbone_name.split("-")[-1]
            )
        backbone = patchcore.backbones.load(backbone_name)
        backbone.name, backbone.seed = backbone_name, backbone_seed

        nn_method = patchcore.common.FaissNN(args.faiss_on_gpu, args.faiss_num_workers)

        patchcore_instance = patchcore.patchcore.PatchCore(device)
        patchcore_instance.load(
            backbone=backbone,
            layers_to_extract_from=args.layers_to_extract_from,
            device=device,
            input_shape=input_shape,
            pretrain_embed_dimension=args.pretrain_embed_dimension,
            target_embed_dimension=args.target_embed_dimension,
            patchsize=args.patchsize,
            featuresampler=sampler,
            anomaly_scorer_num_nn=args.anomaly_scorer_num_nn,
            nn_method=nn_method,
            is_low_shot=args.low_shot,
            low_shot_select=args.low_shot_select_data,
            distance = args.distance,
            test_method=args.test_method,
        )
        loaded_patchcores.append(patchcore_instance)
    return loaded_patchcores


def get_args_parser():
    parser = argparse.ArgumentParser('patch_core_ours', add_help=False)
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument('--results_path', default='./results/', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--log_group', default='group', type=str)
    parser.add_argument('--log_project', default='project', type=str)
    parser.add_argument('--save_segmentation_images', default=False)
    parser.add_argument('--save_patchcore_model', default=False)
    parser.add_argument('--patchsize', default=3, type=int)
    parser.add_argument('--patchscore', default='max', type=str)
    parser.add_argument('--patchoverlap', default=0.0, type=float)
    parser.add_argument('--faiss_num_workers', default=8, type=int)
    parser.add_argument('--faiss_on_gpu', default=False, type=bool)
    parser.add_argument('--augment', default=True, type=bool)
    parser.add_argument('--aggregation', default='mean', choices=['mean','mlp'])
    parser.add_argument('--preprocessing', default='mean', choices=["mean", "conv"])
    parser.add_argument('--target_embed_dimension', default=1024, type=int)
    parser.add_argument('--pretrain_embed_dimension', default=1024, type=int)
    parser.add_argument('--anomaly_scorer_num_nn', default=5, type=int)
    parser.add_argument('--backbone_names', default=['wideresnet50'])
    parser.add_argument('--layers_to_extract_from', default=['layer2','layer3'])
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--percentage', default=0.1, type=float)
    parser.add_argument('--data_path', default='/home/zhy/anomaly/datasets/mvtec', type=str)
    parser.add_argument('--train_val_split', default=1, type=int)
    parser.add_argument('--sampler_name', default='basis_feature_sampling', type=str)
    parser.add_argument('--test_method', default='ASOMP', type=str, choices=["ASOMP", "Similarity_Distance"])
    parser.add_argument('--dataset', default='mvtec', type=str)
    parser.add_argument('--low_shot', default=False, type=bool, help='if low shot, make sure batch_size=1')
    parser.add_argument('--dataset_len', default=15, type=int)
    parser.add_argument('--label_times', default=0, type=int, help='for labeling the line of sample id in the low_shot file')
    parser.add_argument('--low_shot_select_data', default=[], help='sample id used in the low shot task')
    parser.add_argument('--distance', default=True, type=bool, help='true-gaussian, false-l2')
    args = parser.parse_args()
    return args

def main():
    args = get_args_parser()
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    args = updataconfig(args, config)

    run_save_path = patchcore.utils.create_storage_folder(
        args.results_path, args.log_project, args.log_group, mode="iterate"
    )

    list_of_dataloaders = get_dataloaders(args)

    device = patchcore.utils.set_torch_device([0])
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )

    result_collect = []

    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["training"].name,
                dataloader_count + 1,
                len(list_of_dataloaders),
            )
        )

        patchcore.utils.fix_seeds(args.seed, device)
        dataset_name = dataloaders["training"].name

        # low shot
        if (args.low_shot == True):
            sub_label = args._CLASSNAMES.index(dataset_name.split('_')[-1])
            select_row = args.label_times * args.dataset_len + sub_label
            save_label_ = open("./low_shot_file/{}_select_data.txt".format(args.dataset), "r")
            lines = save_label_.readlines()
            args.low_shot_select_data = np.array(lines[select_row].split()).astype(int)
            save_label_.close()

        with device_context:
            torch.cuda.empty_cache()
            imagesize = dataloaders["training"].dataset.imagesize
            sampler = get_sampler(args, device)
            PatchCore_list = get_patchcore(args, imagesize, sampler, device)
            if len(PatchCore_list) > 1:
                LOGGER.info(
                    "Utilizing PatchCore Ensemble (N={}).".format(len(PatchCore_list))
                )
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                if PatchCore.backbone.seed is not None:
                    patchcore.utils.fix_seeds(PatchCore.backbone.seed, device)
                LOGGER.info(
                    "Training models ({}/{})".format(i + 1, len(PatchCore_list))
                )
                torch.cuda.empty_cache()
                PatchCore.fit(dataloaders["training"], dataset_name)    # training

            torch.cuda.empty_cache()
            aggregator = {"scores": [], "segmentations": []}
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                LOGGER.info(
                    "Embedding test data with models ({}/{})".format(
                        i + 1, len(PatchCore_list)
                    )
                )
                scores, segmentations, labels_gt, masks_gt = PatchCore.predict(
                    dataloaders["testing"]        # testing
                )
                aggregator["scores"].append(scores)
                aggregator["segmentations"].append(segmentations)

            scores = np.array(aggregator["scores"])
            min_scores = scores.min(axis=-1).reshape(-1, 1)
            max_scores = scores.max(axis=-1).reshape(-1, 1)
            scores = (scores - min_scores) / (max_scores - min_scores)
            scores = np.mean(scores, axis=0)

            segmentations = np.array(aggregator["segmentations"])
            min_scores = (segmentations.reshape(len(segmentations), -1).min(axis=-1).reshape(-1, 1, 1, 1))
            max_scores = (segmentations.reshape(len(segmentations), -1).max(axis=-1).reshape(-1, 1, 1, 1))
            segmentations = (segmentations - min_scores) / (max_scores - min_scores)
            segmentations = np.mean(segmentations, axis=0)

            results = metric_result(dataloaders, scores, masks_gt, segmentations)
            threshold = results[4]
            result_collect.append(
                {
                    "dataset_name": dataset_name,
                    "instance_auroc": results[0],
                    "full_pixel_auroc": results[1],
                    "pro_auc": results[3],
                    "anomaly_pixel_auroc": results[2],
                }
            )
            for key, item in result_collect[-1].items():
                if key != "dataset_name":
                    LOGGER.info("{0}: {1:3.3f}".format(key, item))

            # (Optional) Plot example images.
            if(args.save_segmentation_images):
                image_paths = [
                    x[2] for x in dataloaders["testing"].dataset.data_to_iterate
                ]
                mask_paths = [
                    x[3] for x in dataloaders["testing"].dataset.data_to_iterate
                ]
                def image_transform(image):
                    in_std = np.array(
                        dataloaders["testing"].dataset.transform_std
                    ).reshape(-1, 1, 1)
                    in_mean = np.array(
                        dataloaders["testing"].dataset.transform_mean
                    ).reshape(-1, 1, 1)
                    image = dataloaders["testing"].dataset.transform_img(image)
                    return np.clip(
                        (image.numpy() * in_std + in_mean) * 255, 0, 255
                    ).astype(np.uint8)

                def mask_transform(mask):
                    return dataloaders["testing"].dataset.transform_mask(mask).numpy()

                image_save_path = os.path.join(run_save_path, "segmentation_images", dataset_name)

                os.makedirs(image_save_path, exist_ok=True)
                patchcore.utils.plot_segmentation_images(
                    image_save_path,
                    image_paths,
                    segmentations,
                    scores,
                    threshold,
                    mask_paths,
                    image_transform=image_transform,
                    mask_transform=mask_transform,
                )

            # (Optional) Store PatchCore model for later re-use.
            # SAVE all patchcores only if mean_threshold is passed?
            if(args.save_patchcore_model):
                patchcore_save_path = os.path.join(
                    run_save_path, "models", dataset_name
                )
                os.makedirs(patchcore_save_path, exist_ok=True)
                for i, PatchCore in enumerate(PatchCore_list):
                    prepend = (
                        "Ensemble-{}-{}_".format(i + 1, len(PatchCore_list))
                        if len(PatchCore_list) > 1
                        else ""
                    )
                    PatchCore.save_to_path(patchcore_save_path, prepend)

        LOGGER.info("\n\n-----\n")


    # Store all results and mean scores to a csv-file.
    result_metric_names = list(result_collect[-1].keys())[1:]
    result_dataset_names = [results["dataset_name"] for results in result_collect]
    result_scores = [list(results.values())[1:] for results in result_collect]
    patchcore.utils.compute_and_store_final_results(
        run_save_path,
        result_scores,
        column_names=result_metric_names,
        row_names=result_dataset_names,
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()
