import os
import numpy as np
import argparse
import nibabel as nb
import torch
from monai import transforms, data
from monai.data import load_decathlon_datalist
import warnings

# Keep your custom import
from TumorGenerated import TumorGenerated 

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Data Generation Only')

# Paths
parser.add_argument('--train_dir', default=None, type=str, required=True, help='Path to input data')
parser.add_argument('--json_dir', default=None, type=str, required=True, help='Path to datalist json')
parser.add_argument('--output_dir', default='./generated_data', type=str, help='Where to save the new data')

# Processing configs
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--tumor_prob', default=1.0, type=float, help='Probability to add a tumor. Set to 1.0 to force generation.')

def get_generation_transform(args):
    """
    Defines transforms for data generation. 
    Note: Random Cropping and Geometric flips are removed so we save the FULL volume.
    """
    gen_transform = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image", "label"]),
        transforms.AddChanneld(keys=["image", "label"]),
        transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
        # Ensure spacing is consistent (optional, depending on your needs)
        transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        
        # --- THE CORE GENERATION STEP ---
        # We use the probability from args (suggest setting to 1.0 for generation)
        TumorGenerated(keys=["image", "label"], prob=args.tumor_prob), 
        
        # Scale intensity (Optional: decide if you want to save scaled or raw data)
        # If you want raw data with tumors, comment this out.
        transforms.ScaleIntensityRanged(
            keys=["image"], a_min=-21, a_max=189,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        
        transforms.ToTensord(keys=["image", "label"]),
    ]
    )
    return gen_transform

def save_nifti(data_tensor, meta_dict, output_path, is_label=False):
    """Helper to save tensor as NIfTI using original affine"""
    # Convert tensor back to numpy
    if torch.is_tensor(data_tensor):
        data_array = data_tensor.cpu().numpy()
    else:
        data_array = data_tensor
        
    # Remove channel dim (C, H, W, D) -> (H, W, D)
    if data_array.shape[0] == 1:
        data_array = data_array[0]
        
    # Get affine from metadata to keep spatial positioning correct
    affine = meta_dict.get("affine", np.eye(4))
    
    # Create NIfTI image
    nifti_img = nb.Nifti1Image(data_array, affine)
    
    # Save
    nb.save(nifti_img, output_path)
    print(f"Saved: {output_path}")

def main():
    args = parser.parse_args()
    
    # Create output directories
    img_save_dir = os.path.join(args.output_dir, "imagesTr")
    lbl_save_dir = os.path.join(args.output_dir, "labelsTr")
    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(lbl_save_dir, exist_ok=True)

    print("Argument values:")
    for k, v in vars(args).items():
        print(k, '=>', v)
    print('-----------------')

    # Load Datalist
    # base_dir=args.train_dir tells monai where the relative paths in json start
    datalist = load_decathlon_datalist(args.json_dir, True, "training", base_dir=args.train_dir)
    
    print(f"Found {len(datalist)} files to process.")

    # Get Transforms
    train_transform = get_generation_transform(args)

    # Use standard Dataset (Cache is not necessary for one-pass generation)
    ds = data.Dataset(data=datalist, transform=train_transform)
    
    # DataLoader
    loader = data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.workers)

    print("Starting Data Generation...")
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            inputs = batch["image"]
            labels = batch["label"]
            
            # Metadata contains the original filename and affine
            # Note: Because of batch_size=1, we access index 0
            img_meta = batch["image_meta_dict"]
            
            # Construct Output Filename
            # Try to get original filename, otherwise use index
            try:
                original_path = img_meta["filename_or_obj"][0]
                filename = os.path.basename(original_path)
            except:
                filename = f"gen_data_{i:04d}.nii.gz"
                
            # Define output paths
            out_img_path = os.path.join(img_save_dir, filename)
            out_lbl_path = os.path.join(lbl_save_dir, filename)
            
            # Save Image
            save_nifti(inputs[0], img_meta, out_img_path, is_label=False)
            
            # Save Label
            save_nifti(labels[0], img_meta, out_lbl_path, is_label=True)

    print("Data generation finished.")

if __name__ == '__main__':
    main()