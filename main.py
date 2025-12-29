import os
import glob
import argparse
import nibabel as nb
import torch
from monai import transforms, data
import warnings

import numpy as np

# Keep your custom import
from TumorGenerated import TumorGenerated 

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Data Generation Only (Auto-Discovery)')

# Paths
parser.add_argument('--train_dir', default=None, type=str, required=True, 
                    help='Root directory containing "imagesTr" and "labelsTr" folders')
parser.add_argument('--output_dir', default='./generated_data', type=str, 
                    help='Where to save the new data')

# Processing configs
parser.add_argument('--workers', default=1, type=int)
parser.add_argument('--tumor_prob', default=1.0, type=float, 
                    help='Probability to add a tumor. Set to 1.0 to force generation.')

def get_generation_transform(args):
    """
    Defines transforms for data generation.
    """
    gen_transform = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.AddChanneld(keys=["image", "label"]),
        transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
        transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        
        # --- THE CORE GENERATION STEP ---
        TumorGenerated(keys=["image", "label"], prob=args.tumor_prob), 
        
        # Optional: Normalize intensity to 0-1 range. 
        # Comment this out if you want to keep original intensity range.
        # transforms.ScaleIntensityRanged(
        #     keys=["image"], a_min=-21, a_max=189,
        #     b_min=0.0, b_max=1.0, clip=True,
        # ),
        
        transforms.ToTensord(keys=["image", "label"]),
    ]
    )
    return gen_transform

def save_nifti(data_tensor, meta_dict, output_path):
    """Helper to save tensor as NIfTI using original affine"""
    # 1. Convert Data Tensor to Numpy
    if torch.is_tensor(data_tensor):
        data_array = data_tensor.cpu().numpy()
    else:
        data_array = data_tensor
        
    # Remove channel dim (C, H, W, D) -> (H, W, D) if C=1
    if data_array.ndim == 4 and data_array.shape[0] == 1:
        data_array = data_array[0]
        
    # 2. Extract and Fix Affine
    # meta_dict["affine"] is usually a Tensor of shape (Batch, 4, 4)
    affine = meta_dict.get("affine", np.eye(4))
    
    # If it's a Tensor, convert to numpy
    if torch.is_tensor(affine):
        affine = affine.detach().cpu().numpy()
        
    # If it has a batch dimension (e.g., shape (1, 4, 4)), take the first item
    if affine.ndim == 3:
        affine = affine[0]
    
    # 3. Create and Save NIfTI
    # Now affine is guaranteed to be (4, 4)
    nifti_img = nb.Nifti1Image(data_array, affine)
    nb.save(nifti_img, output_path)
    print(f"Saved: {output_path}")

def get_datalist_from_folder(root_dir):
    """
    Scans the folder for images and labels automatically.
    Assumes standard structure:
       root_dir/
          imagesTr/
             img1.nii.gz
             ...
          labelsTr/
             img1.nii.gz
             ...
    """
    image_dir = os.path.join(root_dir, "imagesTr")
    label_dir = os.path.join(root_dir, "labelsTr")

    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        raise FileNotFoundError(f"Could not find 'imagesTr' or 'labelsTr' inside {root_dir}")

    # Find all NIfTI files in imagesTr
    # We search for .nii.gz and .nii
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.nii*")))
    
    datalist = []
    for img_path in image_files:
        file_name = os.path.basename(img_path)
        
        # Assume label has the same filename
        lbl_path = os.path.join(label_dir, file_name)
        
        # Only add to list if BOTH image and label exist
        if os.path.exists(lbl_path):
            datalist.append({"image": img_path, "label": lbl_path})
        else:
            print(f"Warning: Label not found for {file_name}, skipping.")

    if not datalist:
        raise ValueError("No matching image/label pairs found!")
        
    return datalist

def main():
    args = parser.parse_args()
    
    # Create output directories
    img_save_dir = os.path.join(args.output_dir, "imagesTr")
    lbl_save_dir = os.path.join(args.output_dir, "labelsTr")
    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(lbl_save_dir, exist_ok=True)

    print(f"Scanning directory: {args.train_dir}")

    # --- AUTO-DISCOVERY START ---
    datalist = get_datalist_from_folder(args.train_dir)
    print(f"Found {len(datalist)} valid image/label pairs.")
    # --- AUTO-DISCOVERY END ---

    # Get Transforms
    train_transform = get_generation_transform(args)

    # Use standard Dataset
    ds = data.Dataset(data=datalist, transform=train_transform)
    loader = data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.workers)

    print("Starting Data Generation...")
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            inputs = batch["image"]
            labels = batch["label"]
            img_meta = batch["image_meta_dict"]
            
            # Get filename
            try:
                original_path = img_meta["filename_or_obj"][0]
                filename = os.path.basename(original_path)
            except:
                filename = f"gen_data_{i:04d}.nii.gz"
                
            out_img_path = os.path.join(img_save_dir, filename)
            out_lbl_path = os.path.join(lbl_save_dir, filename)
            
            save_nifti(inputs[0], img_meta, out_img_path)
            save_nifti(labels[0], img_meta, out_lbl_path)

    print("Data generation finished.")

if __name__ == '__main__':
    main()