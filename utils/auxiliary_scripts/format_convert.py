# -*- coding: utf-8 -*-
'''
skimage -> scikit-image 是一个常用的 python 图像处理库
提供了丰富的图像读写、滤波、变换、分割等功能

SimpleITK -> Simple Insight Toolkit 
这个是一个医学图像处理库 可以读取 保存 处理 DICOM NIfTI MHD NRRD 等医学图像
'''
import os
join = os.path.join
import random
import numpy as np
from skimage import io
import SimpleITK as sitk

def dcm2nii(dcm_path, nii_path):
    """
    Convert dicom files to nii files
    DICOM 通常是每张切片一个 .dcm 文件
    读取 DICOM 文件夹 并转换为 NIfTI 文件
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dcm_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    sitk.WriteImage(image, nii_path)

def mhd2nii(mhd_path, nii_path):
    """
    mhd -> MetaHeader Data
    Convert mhd files to nii files
    """
    image = sitk.ReadImage(mhd_path)
    sitk.WriteImage(image, nii_path)

def nii2nii(nii_path, nii_gz_path):
    """
    Convert nii files to nii.gz files, which can reduce the file size
    """
    image = sitk.ReadImage(nii_path)
    sitk.WriteImage(image, nii_gz_path)

def nrrd2nii(nrrd_path, nii_path):
    """
    Convert nrrd files to nii files
    nrrd -> Nearly Raw Raster Data
    更通用 支持多种数据类型和编码方式
    NIfTi -> 专门为神经影像学设计 是MRI/fMRI数据分析的标准格式
    """
    image = sitk.ReadImage(nrrd_path)
    sitk.WriteImage(image, nii_path)

def jpg2png(jpg_path, png_path):
    """
    Convert jpg files to png files
    """
    image = io.imread(jpg_path)
    io.imsave(png_path, image)

def patchfy(img, mask, outpath, basename):
    """
    Patchfy the image and mask into 1024x1024 patches
    puts image and mask into 1024x1024 patches
    就是将医学图像的维度(H, W, C) 切割成 1024 x 1024 的小块
    """
    image_patch_dir = join(outpath, "images")
    mask_patch_dir = join(outpath, "labels")
    os.makedirs(image_patch_dir, exist_ok=True)
    os.makedirs(mask_patch_dir, exist_ok=True)
    assert img.shape[:2] == mask.shape # image size -> (heights, widths, channels)
    patch_height = 1024
    patch_width = 1024

    img_height, img_width = img.shape[:2]
    mask_height, mask_width = mask.shape

    if img_height % patch_height != 0:
        img = np.pad(img, ((0, patch_height - img_height % patch_height), (0, 0), (0, 0)), mode="constant")
    if img_width % patch_width != 0:
        img = np.pad(img, ((0, 0), (0, patch_width - img_width % patch_width), (0, 0)), mode="constant")
    if mask_height % patch_height != 0:
        mask = np.pad(mask, ((0, patch_height - mask_height % patch_height), (0, 0)), mode="constant")
    if mask_width % patch_width != 0:
        mask = np.pad(mask, ((0, 0), (0, patch_width - mask_width % patch_width)), mode="constant")

    assert img.shape[:2] == mask.shape
    assert img.shape[0] % patch_height == 0
    assert img.shape[1] % patch_width == 0
    assert mask.shape[0] % patch_height == 0
    assert mask.shape[1] % patch_width == 0

    '''
    将图像和 mask 切割成 1024 x 1024 的小块
    '''
    height_steps = (img_height // patch_height) if img_height % patch_height == 0 else (img_height // patch_height + 1)
    width_steps = (img_width // patch_width) if img_width % patch_width == 0 else (img_width // patch_width + 1)

    for i in range(height_steps):
        for j in range(width_steps):
            img_patch = img[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width, :]
            mask_patch = mask[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width]
            assert img_patch.shape[:2] == mask_patch.shape
            assert img_patch.shape[0] == patch_height
            assert img_patch.shape[1] == patch_width
            print(f"img_patch.shape: {img_patch.shape}, mask_patch.shape: {mask_patch.shape}")
            img_patch_path = join(image_patch_dir, f"{basename}_{i}_{j}.png")
            mask_patch_path = join(mask_patch_dir, f"{basename}_{i}_{j}.png")
            io.imsave(img_patch_path, img_patch)    
            io.imsave(mask_patch_path, mask_patch)

def rle_decode(mask_rle, img_shape):
    """
    functions to convert encoding to mask and mask to encoding
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 for mask, 0 for background

    ⭐️关于最后的转置部分 
    因为 RLE 编码是按列优先顺序进行编码的
    而 numpy/Python 默认是按行优先顺序存储的
    """
    seq = mask_rle.split()
    starts = np.array(list(map(int, seq[0::2]))) # start positions
    lengths = np.array(list(map(int, seq[1::2]))) # lengths of the runs
    assert len(starts) == len(lengths)
    ends = starts + lengths # end positions
    img = np.zeros((np.product(img_shape),), dtype=np.uint8)
    for begin, end in zip(starts, ends):
        img[begin:end] = 255
    img.shape = img_shape # reshape to original shape
    return img.T # Transpose to make image row-wise