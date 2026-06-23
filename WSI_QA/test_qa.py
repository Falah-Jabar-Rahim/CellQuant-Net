
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import cv2
import torch
import sys
import shutil
import random
import time
import logging
import argparse
import numpy as np
import warnings
import pandas as pd
from PIL import Image
from config import get_config
Image.MAX_IMAGE_PIXELS = None
from network.DHUnet import DHUnet
from torchvision import transforms
import torch.backends.cudnn as cudnn
from utils import (
    test_single_patch,
    find_Tissue_regions,
    create_folder,
    create_patches,
    data_generator,
    fill_holes_wsi_seg,)

test_transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.ToTensor()
])

warnings.filterwarnings(
    "ignore",
    message="torch.meshgrid: in an upcoming release.*"
)

# ============================================================
# Arguments
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
parser = argparse.ArgumentParser(description="WSI-QA inference only")
# Model
parser.add_argument("--pretrained_ckpt", type=str, default=os.path.join(PROJECT_ROOT, "WSI_QA", "pretrained_ckpt", "WSI-QA.pth"))
parser.add_argument("--cfg", type=str, default=os.path.join(PROJECT_ROOT, "WSI_QA", "configs", "DHUnet_224.yaml"))
parser.add_argument('--network', type=str, default='DHUnet')
parser.add_argument('--num_classes', type=int, default=4)
# Input / output
parser.add_argument("--wsi_folder", type=str, default=os.path.join(PROJECT_ROOT, "input"))
parser.add_argument('--output_dir', type=str, default=os.path.join(PROJECT_ROOT, 'output/QA'))
# Inference
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--cpu_workers', type=int, default=32)
parser.add_argument('--wsilevel', type=int, default=0)
parser.add_argument('--thumbnail_size', type=int, default=1000)
# Quality thresholds
parser.add_argument('--back_thr', type=int, default=30)
parser.add_argument('--blur_fold_thr', type=int, default=20)
# Output options
parser.add_argument('--save_seg', type=int, default=0)
# Reproducibility
parser.add_argument('--seed', type=int, default=301)

args = parser.parse_args()
config = get_config(args)


def set_seed(seed):
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def inference(args, model, test_loader, cpu_workers):
    all_tiles = []
    all_stats = []
    all_names = []

    model.eval()

    with torch.no_grad():
        for data, names in test_loader:
            data = data.cuda(non_blocking=True)

            batch_output_seg, batch_tile_sta = test_single_patch(
                args,
                data,
                model,
                cpu_workers,
                network=args.network
            )

            all_tiles.append(list(batch_output_seg))
            all_stats.append(batch_tile_sta)
            all_names.append(names)

    return all_tiles, all_stats, all_names


if __name__ == "__main__":

    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = DHUnet(config, num_classes=args.num_classes)
    net = net.to(device)

    checkpoint = torch.load(args.pretrained_ckpt, map_location=device)
    msg = net.load_state_dict(checkpoint)
    print("Loaded checkpoint:", args.pretrained_ckpt, msg)

    data = args.wsi_folder
    wsi_files = [
        f for f in os.listdir(data)
        if f.endswith(".svs") or f.endswith(".mrxs") or f.endswith(".tif") or f.endswith(".ndpi")
    ]

    if len(wsi_files) == 0:
        raise RuntimeError(f"No WSI files found in: {data}")

    if args.img_size != 270:
        print("Warning: tile size larger than 270 is not recommended.")

    start_time = time.time()
    all_wsi_info = []

    for wsi_id, wsi_file in enumerate(wsi_files):

        print(f"\nProcessing WSI {wsi_id + 1}/{len(wsi_files)}: {wsi_file}")

        stats = [["tile", "%background", "%tissue", "%fold", "%blur", "classification"]]

        wsi_name = os.path.splitext(wsi_file)[0]
        result_dir = os.path.join(args.output_dir, f"{wsi_name}")

        qualified_dir = os.path.join(result_dir, "Qualified")
        unqualified_dir = os.path.join(result_dir, "Unqualified")
        tile_folder = os.path.join(result_dir, "All_tiles")

        create_folder(result_dir)
        create_folder(qualified_dir)
        create_folder(unqualified_dir)
        create_folder(tile_folder)

        wsi_path = os.path.join(data, wsi_file)

        thumbnail, thumbnail_mask, thumbnail_roi, xmin_indx, ymin_indx, xmax_indx, ymax_indx, sf_w, sf_h, wsi_info = find_Tissue_regions(args,
            wsi_path,
            args.thumbnail_size,
            args.img_size
        )


        create_patches(
            wsi_path,
            wsi_file,
            tile_folder,
            args.cpu_workers,
            args.img_size,
            xmin_indx,
            ymin_indx,
            xmax_indx,
            ymax_indx
        )

        print("Tile generation is done.")

        data_loader, total_patches = data_generator(
            tile_folder,
            test_transform=test_transform,
            batch_size=args.batch_size,
            worker=args.cpu_workers
        )

        output_seg, tile_stats, tile_names = inference(
            args,
            net,
            data_loader,
            args.cpu_workers
        )

        print("Tile segmentation is done.")

        thumbnail_h, thumbnail_w, _ = thumbnail.shape
        wsi_seg = np.zeros((thumbnail_h, thumbnail_w, 3), dtype=np.uint8)

        for batch_id, _ in enumerate(output_seg):
            batch_tile_names = tile_names[batch_id]
            batch_tile_imgs = output_seg[batch_id]
            batch_tile_stats = tile_stats[batch_id]

            for idx in range(len(batch_tile_names)):
                tile_img = batch_tile_imgs[idx]
                tile_name = batch_tile_names[idx]
                st = batch_tile_stats[idx]

                x_min_wsi = int(tile_name.split(".")[0].split("_")[-2])
                y_min_wsi = int(tile_name.split(".")[0].split("_")[-1])

                tile_img_resized = cv2.resize(
                    tile_img,
                    (int(args.img_size / sf_w), int(args.img_size / sf_h)),
                    interpolation=cv2.INTER_NEAREST
                )

                source_path = os.path.join(tile_folder, tile_name)

                if st[4] == "qualified":
                    destination_path = os.path.join(qualified_dir, tile_name)
                else:
                    destination_path = os.path.join(unqualified_dir, tile_name)

                shutil.move(source_path, destination_path)

                if args.save_seg:
                    tile_img_arr = Image.fromarray(tile_img)
                    tile_img_arr.save(destination_path.split(".")[0] + "_seg.png")

                x_min_seg = int(x_min_wsi / sf_w)
                y_min_seg = int(y_min_wsi / sf_h)
                x_max_seg = x_min_seg + tile_img_resized.shape[1]
                y_max_seg = y_min_seg + tile_img_resized.shape[0]

                wsi_seg[y_min_seg:y_max_seg, x_min_seg:x_max_seg, :] = tile_img_resized

                stats.append([tile_name, st[0], st[1], st[2], st[3], st[4]])

        print("WSI segmentation mask is done.")

        wsi_seg_fill = fill_holes_wsi_seg(wsi_seg)
        thumbnail_mask_3d = np.repeat(thumbnail_mask[:, :, np.newaxis], 3, axis=2)

        masked_rgb_image = wsi_seg_fill * thumbnail_mask_3d
        masked_rgb_image = masked_rgb_image.astype(np.uint8)

        Image.fromarray(masked_rgb_image).save(
            os.path.join(result_dir, f"{wsi_name}_seg.png")
        )

        Image.fromarray(thumbnail).save(
            os.path.join(result_dir, f"{wsi_name}_thumbnail.png")
        )

        Image.fromarray(thumbnail_roi).save(
            os.path.join(result_dir, f"{wsi_name}_thumbnail_roi.png")
        )

        df = pd.DataFrame(stats[1:], columns=stats[0])
        df.to_excel(
            os.path.join(result_dir, f"{wsi_name}_stats.xlsx"),
            index=False
        )

        # Add WSI name
        wsi_info["wsi_name"] = wsi_name

        # Count tiles
        wsi_info["total_tiles"] = len(df)

        # Count qualified/unqualified tiles
        wsi_info["qualified_tiles"] = (df["classification"] == "qualified").sum()
        wsi_info["unqualified_tiles"] = (df["classification"] != "qualified").sum()

        # Percentages
        wsi_info["qualified_percent"] = (
                100 * wsi_info["qualified_tiles"] / max(wsi_info["total_tiles"], 1)
        )

        all_wsi_info.append(wsi_info.copy())


        shutil.rmtree(tile_folder)
        print("Results are saved in:", result_dir)

    end_time = time.time()
    summary_df = pd.DataFrame(all_wsi_info)

    summary_df.to_excel(
        os.path.join(args.output_dir, "WSI_Summary.xlsx"),
        index=False
    )

    print(f"\nInference time: {(end_time - start_time) / 60:.2f} minutes")
    print("Completed!")