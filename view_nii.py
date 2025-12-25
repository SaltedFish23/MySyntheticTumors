import nibabel as nib
import matplotlib.pyplot as plt

PATH = "~/MyCode/DataSet/"

DATASET = "03_CHAOS/mri/t1_img/"

# Load the volume (Make sure to use the file WITHOUT the ._ prefix)
file_path = PATH + DATASET + '1_image_T1DUAL_InPhase.nii.gz'
img = nib.load(file_path)
data = img.get_fdata()

# NIfTI files are 3D (Width, Height, Number of Slices)
print(f"Data shape: {data.shape}")

# Display the middle slice
middle_slice_idx = data.shape[2] // 2
plt.imshow(data[:, :, middle_slice_idx], cmap='gray')
plt.title(f"Slice {middle_slice_idx}")
plt.show()