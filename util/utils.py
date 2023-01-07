import pandas as pd
from numpy import ndarray as NDArray
from skimage import measure
from sklearn.metrics import auc, roc_auc_score, roc_curve
from tqdm import tqdm
from statistics import mean
from sklearn.metrics import precision_recall_curve
import numpy as np
import patchcore.metrics
import logging
LOGGER = logging.getLogger(__name__)

def updataconfig(args, config):
    # update some params
    for key, value in config.items():
        setattr(args, key, value)
    # save_path
    if(args.low_shot):
        args.log_project = "{}_{}_low_shot".format(args.log_project, args.dataset)
    else:
        args.log_project = "{}_{}_full_shot".format(args.log_project, args.dataset)
    return args


def metric_result(dataloaders, scores, masks_gt, segmentations):

    LOGGER.info("Computing evaluation metrics.")
    anomaly_labels = [x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate]
    auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(scores, anomaly_labels)["auroc"]

    gt_mask = np.array(masks_gt)[:, 0, :, :]
    gt_mask[gt_mask > 0.5] = 1
    gt_mask[gt_mask <= 0.5] = 0

    pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(segmentations, gt_mask)
    full_pixel_auroc = pixel_scores["auroc"]

    sel_idxs = []
    for i in range(len(gt_mask)):
        if np.sum(gt_mask[i]) > 0:
            sel_idxs.append(i)
    pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
        [segmentations[i] for i in sel_idxs],
        [gt_mask[i] for i in sel_idxs],)
    anomaly_pixel_auroc = pixel_scores["auroc"]


    flatten_gt_mask = gt_mask.ravel()
    flatten_score_map = segmentations.ravel()
    precision, recall, thresholds = precision_recall_curve(flatten_gt_mask, flatten_score_map)
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]

    if(dataloaders["training"].name=='mstc_01'):
        # due to too many image, the code will be kill, so we don't compute this value
        pro_auc=0.5
    else:
        pro_auc = compute_pro_score1(segmentations, gt_mask)

    return [auroc, full_pixel_auroc, anomaly_pixel_auroc, pro_auc, threshold]


def compute_pro_score1(scores, masks, max_step=200, expect_fpr=0.3):
    # per region overlap and per image iou
    max_th = scores.max()
    min_th = scores.min()
    delta = (max_th - min_th) / max_step

    ious_mean = []
    ious_std = []
    pros_mean = []
    pros_std = []
    threds = []
    fprs = []
    binary_score_maps = np.zeros_like(scores, dtype=np.bool)
    for step in range(max_step):
        thred = max_th - step * delta
        # segmentation
        binary_score_maps[scores <= thred] = 0
        binary_score_maps[scores > thred] = 1

        pro = []  # per region overlap
        iou = []  # per image iou
        # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
        # iou: for each image, compute the ratio, i.e. intersection/union between the gt and predicted binary map
        for i in range(len(binary_score_maps)):  # for i th image
            # pro (per region level)
            label_map = measure.label(masks[i], connectivity=2)
            props = measure.regionprops(label_map)
            for prop in props:
                x_min, y_min, x_max, y_max = prop.bbox  # find the bounding box of an anomaly region
                cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                # cropped_mask = masks[i][x_min:x_max, y_min:y_max]   # bug!
                cropped_mask = prop.filled_image  # corrected!
                intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
                pro.append(intersection / prop.area)
            # iou (per image level)
            intersection = np.logical_and(binary_score_maps[i], masks[i]).astype(np.float32).sum()
            union = np.logical_or(binary_score_maps[i], masks[i]).astype(np.float32).sum()
            if masks[i].any() > 0:  # when the gt have no anomaly pixels, skip it
                iou.append(intersection / union)
        # against steps and average metrics on the testing data
        ious_mean.append(np.array(iou).mean())
        #             print("per image mean iou:", np.array(iou).mean())
        ious_std.append(np.array(iou).std())
        pros_mean.append(np.array(pro).mean())
        pros_std.append(np.array(pro).std())
        # fpr for pro-auc
        # masks_neg = ~masks
        masks_neg = 1-masks
        fpr = np.logical_and(masks_neg, binary_score_maps).sum() / masks_neg.sum()
        fprs.append(fpr)
        threds.append(thred)

    # as array
    threds = np.array(threds)
    pros_mean = np.array(pros_mean)
    pros_std = np.array(pros_std)
    fprs = np.array(fprs)

    ious_mean = np.array(ious_mean)
    ious_std = np.array(ious_std)

    # save results
    data = np.vstack([threds, fprs, pros_mean, pros_std, ious_mean, ious_std])
    df_metrics = pd.DataFrame(data=data.T, columns=['thred', 'fpr',
                                                    'pros_mean', 'pros_std',
                                                    'ious_mean', 'ious_std'])
    # save results
    # df_metrics.to_csv(os.path.join('./', 'thred_fpr_pro_iou.csv'), sep=',', index=False)

    # best per image iou
    best_miou = ious_mean.max()
    print(f"Best IOU: {best_miou:.4f}")

    # default 30% fpr vs pro, pro_auc
    idx = fprs <= expect_fpr  # find the indexs of fprs that is less than expect_fpr (default 0.3)
    fprs_selected = fprs[idx]
    fprs_selected = rescale(fprs_selected)  # rescale fpr [0,0.3] -> [0, 1]
    pros_mean_selected = pros_mean[idx]
    pro_auc_score = auc(fprs_selected, pros_mean_selected)
    # print("pro auc ({}% FPR):".format(int(expect_fpr * 100)), pro_auc_score)
    return pro_auc_score

def rescale(x):
    return (x - x.min()) / (x.max() - x.min())