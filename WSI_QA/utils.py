
import time
import os
import cv2
import shutil
import torch
import numpy as np
from PIL import Image
import pyvips as vips
from openslide import open_slide
import torch.multiprocessing as mp
from collections import Counter
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes, binary_dilation, label, find_objects



def fill_holes_wsi_seg(mask):
    # Create a copy of the mask to fill the holes
    filled_mask = mask.copy()
    # Get the height and width of the mask
    height, width, _ = mask.shape
    # Define the 8-neighbor directions (up, down, left, right, and 4 diagonals)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),  # Cardinal directions
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]  # Diagonals
    # Iterate over each pixel in the mask
    for y in range(height):
        for x in range(width):
            # Check if the current pixel is a hole (value [0, 0, 0])
            if np.array_equal(mask[y, x], [0, 0, 0]):
                neighbor_values = []
                # Check all 8 neighbors
                for dy, dx in directions:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        neighbor_value = tuple(mask[ny, nx])
                        if neighbor_value != (0, 0, 0):  # Avoid including holes in the neighbors
                            neighbor_values.append(neighbor_value)

                # If there are valid neighbors, fill the current pixel
                if neighbor_values:
                    most_common_value = Counter(neighbor_values).most_common(1)[0][0]
                    filled_mask[y, x] = most_common_value

    return filled_mask

def find_Tissue_regions(args, wsi_path, thumbnail_size, tile_size, plot=False):
    wsi = open_slide(wsi_path)
    wsi_width, wsi_height = wsi.dimensions
    thumbnail = wsi.get_thumbnail((thumbnail_size, thumbnail_size))
    thumbnail_h = thumbnail.height
    thumbnail_w = thumbnail.width
    sf_h = wsi_height / thumbnail_h
    sf_w = wsi_width / thumbnail_w
    thumbnail_np = np.array(thumbnail)
    gray_image = cv2.cvtColor(thumbnail_np, cv2.COLOR_RGB2GRAY)
    _, otsu_threshold = cv2.threshold(gray_image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = otsu_threshold == 0
    cleaned_mask = remove_small_objects(mask, min_size=25)
    structure_8connectivity = np.ones((4, 4), dtype=bool)
    binary_mask = binary_dilation(cleaned_mask, structure=structure_8connectivity)
    # Find the indices of the non-zero elements
    non_zero_indices = np.argwhere(binary_mask)
    # Find the minimum and maximum x and y coordinates
    ymin, xmin = non_zero_indices.min(axis=0)
    ymax, xmax = non_zero_indices.max(axis=0)
    image_with_bbox = thumbnail_np.copy()
    cv2.rectangle(image_with_bbox, (xmin, ymin), (xmax, ymax), (0, 255, 0), 25)  # Blue box with thickness 2

    if plot:
        plt.imshow(mask)
        plt.show()
        plt.imshow(cleaned_mask)
        plt.show()
        plt.imshow(binary_mask)
        plt.show()
        plt.imshow(image_with_bbox)
        plt.show()

    xmin = int(xmin * sf_w)
    xmax = int(xmax * sf_w)
    ymin = int(ymin * sf_h)
    ymax = int(ymax * sf_h)

    if 0 <= xmin <= tile_size:
        xmin_indx = 0
    else:
        xmin_indx = (xmin // tile_size)
    if 0 <= ymin <= tile_size:
        ymin_indx = 0
    else:
        ymin_indx = (ymin // tile_size)
    if xmax + tile_size < wsi_width:
        xmax_indx = (xmax // tile_size)
    else:
        xmax_indx = (xmax // tile_size)
    if ymax + tile_size < wsi_height:
        ymax_indx = (ymax // tile_size)
    else:
        ymax_indx = (ymax // tile_size)


    properties = dict(wsi.properties)
    wsi_info = {
        "wsi_name": os.path.basename(wsi_path),
        "wsi_width": wsi_width,
        "wsi_height": wsi_height,

        "batch_size": args.batch_size,
        "cpu_workers": args.cpu_workers,

        # Magnification / scanner info

        "magnification": properties.get("aperio.AppMag", properties.get("openslide.objective-power", "Unknown")),

        "scanner_type": properties.get("openslide.vendor", "Unknown"),

        "mpp_x": properties.get("openslide.mpp-x", "Unknown"),

        "mpp_y": properties.get("openslide.mpp-y", "Unknown"),
        "thumbnail_width": thumbnail_w,
        "thumbnail_height": thumbnail_h,
        "scale_factor_w": sf_w,
        "scale_factor_h": sf_h,
        "tissue_xmin": xmin,
        "tissue_ymin": ymin,
        "tissue_xmax": xmax,
        "tissue_ymax": ymax,
        "xmin_tile_index": xmin_indx,
        "ymin_tile_index": ymin_indx,
        "xmax_tile_index": xmax_indx,
        "ymax_tile_index": ymax_indx,
        "tile_size": tile_size,
    }

    return thumbnail_np, binary_mask, image_with_bbox, xmin_indx, ymin_indx, xmax_indx, ymax_indx, sf_w, sf_h, wsi_info

def create_folder(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path):
        # If it exists, remove the folder and its contents
        shutil.rmtree(folder_path)
        #print(f'Deleted existing folder: {folder_path}')
    # Create a new folder
    os.makedirs(folder_path)
    #print(f'Created new folder: {folder_path}')

def crop(region, patch_size, x, y):
    return region.read_region((patch_size * x, patch_size * y), 0, (patch_size, patch_size))

def extract_and_save_patch(y_cord, file_path, file_name, patch_folder, patch_size, xmin_indx, xmax_indx):
    slide = open_slide(file_path)
    f_name = file_name.split(".")[0]
    for x_cord in range(xmin_indx, xmax_indx):
        patch = crop(slide, patch_size, x_cord, y_cord)
        x_start, y_start = x_cord * patch_size, y_cord * patch_size
        base_name = f"{f_name}_{x_start}_{y_start}.png"
        patch_rgb = patch.convert('RGB')
        patch_rgb.save(os.path.join(patch_folder, base_name))

def post_proces(prediction, obj_size, args, back_thr, blur_fold_thr):
    prediction = cv2.resize(prediction, (args.img_size, args.img_size), interpolation=cv2.INTER_NEAREST)

    class_colors = {
        0: (0, 0, 0),  # Class 0: Black for background
        1: (0, 255, 0),  # Class 1: green for tissue
        2: (255, 65, 90),  # Class 2: yellow for fold
        3: (255, 165, 0),  # Class 3: orange for blur
    }

    ### fill small holes in the background with tissue
    structure_8connectivity = np.ones((3, 3), dtype=bool)
    binary_mask = prediction == 1
    binary_mask = binary_dilation(binary_mask, structure=structure_8connectivity)
    # Fill holes in the binary mask
    filled_binary_mask = binary_fill_holes(binary_mask)
    # Create a new mask to store the result
    filled_mask = prediction.copy()
    # Set all regions that were holes to the nearest non-zero label
    filled_mask[filled_binary_mask & (prediction == 0)] = 1  # fill with tissue
    prediction = filled_mask

    ### remove small regions that have blur of fold
    # Create a binary mask for the fold and blur
    class_mask = prediction > 1
    # Label connected components in the class mask
    labeled_array, num_features = label(class_mask)
    # Find slices of labeled objects
    object_slices = find_objects(labeled_array)
    # Create a copy of the original mask to modify
    modified_mask = prediction.copy()
    # Iterate over each detected object
    for i, slice_tuple in enumerate(object_slices):
        # Calculate the size of the object
        object_size = np.sum(labeled_array[slice_tuple] == (i + 1))
        # Replace object with replacement class if its size is less than min_size
        if object_size < obj_size:
            modified_mask[labeled_array == (i + 1)] = 1

    prediction = modified_mask

    ### Assign colors to each pixel based on the class map
    height, width = prediction.shape
    output_image = Image.new("RGB", (width, height))
    for y in range(height):
        for x in range(width):
            class_label = prediction[y, x]
            color = class_colors[class_label]
            output_image.putpixel((x, y), color)

    output_image = np.array(output_image)

    ### comute artifcat statistics
    total_pixels = prediction.size
    num_classes = args.num_classes
    tile_stats = []
    for class_value in range(0, num_classes):
        class_pixel_count = np.sum(prediction == class_value)
        percentage = (class_pixel_count / total_pixels) * 100
        tile_stats.append(round(percentage, 2))

    ### tile classifiacton
    if tile_stats[0] >= back_thr:  # check for white background
        classification = "unqualified"
    elif tile_stats[2] >= blur_fold_thr or tile_stats[3] >= blur_fold_thr:  # check for fold or blur
        classification = "unqualified"
    else:  # artifact free
        classification = "qualified"

    tile_stats.append(classification)
    tile_stats = np.array(tile_stats)

    return output_image, tile_stats

def data_generator(patch_folder, test_transform, batch_size=32, worker=1):
    print(f"\nLoading patches...........")
    # test_images = datasets.ImageFolder(root=patch_folder, transform= test_transform)
    test_images = custom_data_loader(patch_folder, test_transform)
    test_loader = DataLoader(dataset=test_images, batch_size=batch_size, shuffle=False, num_workers=worker,
                             pin_memory=True)
    total_patches = len(test_images)
    print(f"total number of patches are {total_patches}")
    return test_loader, total_patches

class custom_data_loader(Dataset):
    def __init__(self, img_path, transform=None):
        self.img_dir = img_path
        self.transform = transform
        self.data_path = []
        file_list = os.listdir(self.img_dir)
        for img in file_list:
            self.data_path.append(os.path.join(self.img_dir, img))

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        image = Image.open(self.data_path[idx]).convert('RGB')
        img_name = os.path.basename(self.data_path[idx])
        if self.transform is not None:
            image = self.transform(image)
        return image, img_name

def create_patches(wsi_path, wsi_name, patch_folder, workers, patch_size, xmin_indx, ymin_indx, xmax_indx, ymax_indx):
    img_400x = vips.Image.new_from_file(wsi_path, level=0, autocrop=True).flatten()
    w, h = img_400x.width, img_400x.height
    n_down = int(h / patch_size)
    params = [(y, wsi_path, wsi_name, patch_folder, patch_size, xmin_indx, xmax_indx)
              for y in range(ymin_indx, ymax_indx)]
    with mp.Pool(processes=workers) as p:
        result = p.starmap(extract_and_save_patch, params)

def test_single_patch(args, image, net, num_processes, network="DHUnet", obj_size=500):
    back_thr = args.back_thr
    blur_fold_thr = args.blur_fold_thr

    image = image.cuda()
    net.eval()
    with torch.no_grad():
        if network == "DHUnet":
            net_out = net(image, image)[0]
            out = torch.argmax(torch.softmax(net_out, dim=1), dim=1).squeeze(0)
        else:
            net_out = net(image)
            out = torch.argmax(torch.softmax(net_out, dim=1), dim=1).squeeze(0)
        predictions = out.cpu().detach().numpy()

    ### post processing
    params = [(y, obj_size, args, back_thr, blur_fold_thr) for y in predictions]
    with mp.Pool(processes=num_processes) as p:
        result = p.starmap(post_proces, params)

    batch_tile_stat = []
    batch_tile = []
    for idx, (output_image, tile_stats) in enumerate(result):
        batch_tile.append(output_image)
        batch_tile_stat.append(tile_stats)

    return batch_tile, batch_tile_stat



