'''
This script loads the summed integrated gradients for a neural network model and plots brain regions corresponding to the features that make up the top x% proportion of the total contribution from all features.
'''



from pathlib import Path
import argparse
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as cmap



def input_parse():
    '''
    For parsing terminal arguments.
    '''
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", "-m", type=str, required=True, help="Name of directory containing model to be evaluated.")
    parser.add_argument("--top_prop", "-p", type=float, default=0.1, help="Top proportion of summed integrated gradients covered by plotted features.")
    args = parser.parse_args()

    return args



def get_indices(binary_array):
    '''
    Identify three indices corresponding to slices with the highest sum, 75th percentile sum, and median sum for each dimension in a binary array.
    '''

    def get_dimension_indices(sum_slices):
        
        # find indices of slices with non-zero sums
        non_zero_indices = np.nonzero(sum_slices)[0]
        non_zero_sums = sum_slices[non_zero_indices]

        # sort the indices based on the sums
        sorted_indices = non_zero_indices[np.argsort(non_zero_sums)[::-1]]

        # highest sum index
        highest_sum_index = sorted_indices[0]

        # 75th percentile index
        percentile_75_index = sorted_indices[int(0.75 * (len(sorted_indices) - 1))]

        # 55th percentile index
        median_index = sorted_indices[len(sorted_indices) // 2]

        # sort the indices in ascending order
        index1, index2, index3 = sorted([highest_sum_index, percentile_75_index, median_index])

        return index1, index2, index3

    # calculate sum of slices for each dimension
    sagittal_sums = np.sum(binary_array, axis=(1, 2))  # sum along coronal and axial planes
    coronal_sums = np.sum(binary_array, axis=(0, 2))   # sum along sagittal and axial planes
    axial_sums = np.sum(binary_array, axis=(0, 1))     # sum along sagittal and coronal planes

    # get indices for each dimension
    sagittal_indices = get_dimension_indices(sagittal_sums)
    coronal_indices = get_dimension_indices(coronal_sums)
    axial_indices = get_dimension_indices(axial_sums)

    return sagittal_indices, coronal_indices, axial_indices



if __name__ == '__main__':

    args = input_parse()

    # paths
    root = Path(__file__).parents[1]
    model_dir = root / 'models' / args.model_dir
    ig_path = model_dir / 'ig_attributions.txt'
    atlas_path = root / 'data' / 'ADHD-200' / 'ADHD200_HO_TCs_filtfix' / 'templates' / 'ho_mask_pad.nii.gz'
    template_path = root / 'data' / 'ADHD-200' / 'preproc_templates' / 'templates' / 'nihpd_asym_04.5-18.5_t1w.nii'

    # load the summed integrated gradients
    with open(ig_path, 'r') as f:
        features_ig = np.array([float(line.strip()) for line in f.readlines()])
    ig_sum = np.sum(features_ig)

    # load images
    atlas = nib.load(atlas_path).get_fdata()
    template = nib.load(template_path).get_fdata()

    # get region identifiers (should correspond to the order of integrated gradients), and remove 0 as it codes for background
    regions = np.unique(atlas)[1:]

    # combine regions and integrated gradients in pandas dataframe
    features_ig_df = pd.DataFrame({'region': regions, 'ig': features_ig}, dtype=float)

    # sort by integrated gradients
    features_ig_df = features_ig_df.sort_values(by='ig', ascending=False)

    # calculate cumulative sum of integrated gradients
    features_ig_df['cumsum'] = features_ig_df['ig'].cumsum()

    # calculate proportion of total integrated gradients
    features_ig_df['cumsum_prop'] = features_ig_df['cumsum'] / ig_sum

    # get top features
    top_features = features_ig_df['region'][features_ig_df['cumsum_prop'] <= args.top_prop].values.tolist()
    print(f'[INFO]: Top {args.top_prop * 100}% features: {top_features}')

    # expand the atlas to the size of the 1mm template - this involves repeating each value of the atlas 4 times in each dimension
    atlas = np.repeat(atlas, 4, axis=0)
    atlas = np.repeat(atlas, 4, axis=1)
    atlas = np.repeat(atlas, 4, axis=2)

    # binarize atlas so top regions are 1 and the rest is 0
    atlas_binary = np.isin(atlas, top_features).astype(int)

    # delete last slice of each dimension in template
    template = np.delete(template, -1, axis=0)
    template = np.delete(template, -1, axis=1)
    template = np.delete(template, -1, axis=2)

    # make cmap where first color is transparent and the second is magenta
    mask_cmap = cmap.from_list('', [(0, 0, 0, 0), (1, 0, 1, 1)], 2)

    # plot evenly spaced slices of the template in saggital, coronal, and axial views
    saggital_slices, coronal_slices, axial_slices = get_indices(atlas_binary)

    fig, axs = plt.subplots(3, 3, figsize=(15, 15))

    for i, slice in enumerate(saggital_slices):
        template_slice = np.flipud(np.rot90(template[slice, :, :]))
        axs[0, i].imshow(template_slice, cmap='gray', origin='lower')
        atlas_slice = np.flipud(np.rot90(atlas_binary[slice, :, :]))
        axs[0, i].imshow(atlas_slice, cmap=mask_cmap, alpha=0.5, origin='lower')
        axs[0, i].text(0.02, 0.94, f'x={slice+1}mm', fontsize=15, fontweight='bold', color='white', transform=axs[0, i].transAxes)


    for i, slice in enumerate(coronal_slices):
        template_slice = np.flipud(np.rot90(template[:, slice, :]))
        axs[1, i].imshow(template_slice, cmap='gray', origin='lower')
        atlas_slice = np.flipud(np.rot90(atlas_binary[:, slice, :]))
        axs[1, i].imshow(atlas_slice, cmap=mask_cmap, alpha=0.5, origin='lower')
        axs[1, i].text(0.02, 0.94, f'y={slice+1}mm', fontsize=15, fontweight='bold', color='white', transform=axs[1, i].transAxes)
        if i == 0:
            axs[1, i].text(0.94, 0.02, 'L', fontsize=17, fontweight='bold', color='white', transform=axs[1, i].transAxes)
            axs[1, i].text(0.02, 0.02, 'R', fontsize=17, fontweight='bold', color='white', transform=axs[1, i].transAxes)

    for i, slice in enumerate(axial_slices):
        template_slice = np.flipud(np.rot90(template[:, :, slice]))
        axs[2, i].imshow(template_slice, cmap='gray', origin='lower')
        atlas_slice = np.flipud(np.rot90(atlas_binary[:, :, slice]))
        axs[2, i].imshow(atlas_slice, cmap=mask_cmap, alpha=0.5, origin='lower')
        axs[2, i].text(0.02, 0.94, f'z={slice+1}mm', fontsize=15, fontweight='bold', color='white', transform=axs[2, i].transAxes)
        if i == 0:
            axs[2, i].text(0.94, 0.02, 'L', fontsize=17, fontweight='bold', color='white', transform=axs[2, i].transAxes)
            axs[2, i].text(0.02, 0.02, 'R', fontsize=17, fontweight='bold', color='white', transform=axs[2, i].transAxes)

    plt.tight_layout()

    plot_filename = f'top_{args.top_prop}_features.png'
    plt.savefig(model_dir / plot_filename)

    print(f'[DONE]: {plot_filename} saved to {model_dir}')