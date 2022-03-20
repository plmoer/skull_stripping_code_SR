import numpy as np
from skimage import morphology
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage import label, binary_dilation
# import matplotlib.pyplot as plt


def post_processing(img):  # result: nRow*nCol*nSlice
    nPixel = 30
    nValue = 2

    temp_bin = img > 0  # remove small object
    for i in range(img.shape[0]):  # for each slice
        temp_bin[i, :, :] = morphology.remove_small_objects(
            temp_bin[i, :, :], nPixel, connectivity=8)
    img = img * temp_bin

    tran_result_bin = (img > 0).astype(int)
    for i in range(tran_result_bin.shape[0]):
        temp = tran_result_bin[i, :, :]  # get binary slice image
        temp_fill = binary_fill_holes(temp)  # filled with hole
        holes = np.logical_xor(temp, temp_fill).astype(int)  # find the hole
        slice_img = img[i, :, :]  # get its image
        if np.sum(np.isin(np.unique(holes), 1).astype(int)) == 1:  # if hole exists
            dilated_holes = binary_dilation(
                holes, iterations=1)  # dilated the hole
            surrounds = np.logical_xor(holes, dilated_holes).astype(int)
            # find index of surrounding pixels
            pixels = slice_img[np.where(surrounds == 1)]
            (values, counts) = np.unique(pixels, return_counts=True)
            ind = np.argmax(counts)
            nValue = values[ind]  # get the most occurance pixels
            # print('...filled value: ', nValue)
            hole_value = holes * nValue
            slice_img = hole_value + slice_img
        img[i, :, :] = slice_img

    return img
