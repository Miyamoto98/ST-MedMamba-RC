# -*- coding: utf-8 -*-
'''
将 nii 转换为用于训练的 numpy 格式的预处理脚本
专门针对 CT 和 MR 图像
同时处理图像和 mask
但是更侧重于对 mask 的处理和清理
'''
import numpy as np
import SimpleITK as sitk
import os
join = os.path.join
from skimage import transform
from tqdm import tqdm
import cc3d

# 可以根据需要修改这些参数
modality = "CT" # 数据类型
anatomy = "Abd" # 解剖部位
img_name_suffix = "_0000.nii.gz" # 图像文件名后缀
gt_name_suffix = ".nii.gz" # 标签文件名后缀
prefix = modality + "_" + anatomy + "_" # 前缀 用于保存文件时命名

nii_path = r" "  # path to the nii images
gt_path = r" "  # path to the ground truth
npy_path = r" " + prefix[:-1] # path to save npy files
os.makedirs(join(npy_path, "gts"), exist_ok=True) # create gts folder
os.makedirs(join(npy_path, "imgs"), exist_ok=True) # create imgs folder

image_size = 1024
voxel_num_thre2d = 100 # 二维连通域最小体素数阈值 过滤小噪声对象
voxel_num_thre3d = 1000 # 三维连通域最小体素数阈值 同理用于 3D 小块过滤

names = sorted(os.listdir(gt_path))
print(f"ori \# files {len(names)=}") # original number of files

# 健全性检查 过滤掉那些没有对应图像的标签文件
names = [
    name
    for name in names
    if os.path.exists(join(nii_path, name.split(gt_name_suffix)[0] + img_name_suffix))
]
print(f"after sanity check \# files {len(names)=}")


remove_label_ids = None # 需要移除的任何结构 我们的任务暂时不需要
tumor_id = None # 当只有一个肿瘤时 不需要实例分割

# only for CT images
WINDOW_LEVEL = 40 
WINDOW_WIDTH = 400


for name in tqdm(names[:1]):
    '''
    处理图像和对应的掩码
    '''
    image_name = name.split(gt_name_suffix)[0] + img_name_suffix
    gt_name = name
    gt_sitk = sitk.ReadImage(join(gt_path, gt_name))
    gt_data_ori = np.uint8(sitk.GetArrayFromImage(gt_sitk))
    for remove_label_id in remove_label_ids:
        gt_data_ori[gt_data_ori == remove_label_id] = 0

    # 只有当有多个肿瘤时才需要实例分割
    if tumor_id is not None:
        tumor_bw = np.uint8(gt_data_ori == tumor_id)
        gt_data_ori[tumor_bw > 0] = 0
        tumor_inst, tumor_n = cc3d.connected_components(
            tumor_bw, connectivity=26, return_N=True
        )
        gt_data_ori[tumor_inst > 0] = (
            tumor_inst[tumor_inst > 0] + np.max(gt_data_ori) + 1
        )

    # 将体素值小于 1000 的连通块排除
    gt_data_ori = cc3d.dust(
        gt_data_ori, threshold=voxel_num_thre3d, connectivity=26, in_place=True
    )
    # 将体素值小于 100 的二维连通块排除
    # 因为对于如此小的连通块而言 最重要的任务是检测而不是分割
    for slice_i in range(gt_data_ori.shape[0]):
        gt_i = gt_data_ori[slice_i, :, :]
        gt_data_ori[slice_i, :, :] = cc3d.dust(
            gt_i, threshold=voxel_num_thre2d, connectivity=8, in_place=True
        )
    # 找到 非零的层
    # 用于后续的数据裁剪
    z_index, _, _ = np.where(gt_data_ori > 0)
    z_index = np.unique(z_index) # 去重并排序

    if len(z_index) > 0:
        gt_roi = gt_data_ori[z_index, :, :]
        img_sitk = sitk.ReadImage(join(nii_path, image_name))
        image_data = sitk.GetArrayFromImage(img_sitk)
        # nii preprocess start
        if modality == "CT":
            '''
            对CT图像进行窗宽窗位调整和归一化处理 突出目标结构并将其像素值缩放到 0 -> 255 范围
            '''
            lower_bound = WINDOW_LEVEL - WINDOW_WIDTH / 2
            upper_bound = WINDOW_LEVEL + WINDOW_WIDTH / 2
            image_data_pre = np.clip(image_data, lower_bound, upper_bound)
            # 归一化到 [0 -> 255]
            image_data_pre = (
                (image_data_pre - np.min(image_data_pre))
                / (np.max(image_data_pre) - np.min(image_data_pre))
                * 255.0
            )
        else:
            '''
            对于非 CT 图像 例如 MR|我们使用更简单的归一化方法
            '''
            lower_bound, upper_bound = np.percentile(
                image_data[image_data > 0], 0.5
            ), np.percentile(image_data[image_data > 0], 99.5)
            image_data_pre = np.clip(image_data, lower_bound, upper_bound)
            image_data_pre = (
                (image_data_pre - np.min(image_data_pre))
                / (np.max(image_data_pre) - np.min(image_data_pre))
                * 255.0
            )
            image_data_pre[image_data == 0] = 0

        # 保存处理后完整的 3D 数据 多个切片
        image_data_pre = np.uint8(image_data_pre)
        img_roi = image_data_pre[z_index, :, :]
        # 将 3D 医学图像转换为 2D 切片用于深度学习训练
        np.savez_compressed(join(npy_path, prefix + gt_name.split(gt_name_suffix)[0]+'.npz'), imgs=img_roi, gts=gt_roi, spacing=img_sitk.GetSpacing()) # 将处理后的图像和标签数据保存为压缩的 numpy 文件格式(.npz)

        # save the cropped nii files for visualization
        img_roi_sitk = sitk.GetImageFromArray(img_roi)
        gt_roi_sitk = sitk.GetImageFromArray(gt_roi)
        sitk.WriteImage(
            img_roi_sitk,
            join(npy_path, prefix + gt_name.split(gt_name_suffix)[0] + "_img.nii.gz"),
        )
        sitk.WriteImage(
            gt_roi_sitk,
            join(npy_path, prefix + gt_name.split(gt_name_suffix)[0] + "_gt.nii.gz"),
        )
        
        # save the each image as npy file
        for i in range(img_roi.shape[0]):
            img_i = img_roi[i, :, :]
            img_3c = np.repeat(img_i[:, :, None], 3, axis=-1)
            resize_img_skimg = transform.resize(
                img_3c,
                (image_size, image_size),
                order=3,
                preserve_range=True,
                mode="constant",
                anti_aliasing=True,
            )
            resize_img_skimg_01 = (resize_img_skimg - resize_img_skimg.min()) / np.clip(
                resize_img_skimg.max() - resize_img_skimg.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1], (H, W, 3)
            gt_i = gt_roi[i, :, :]
            resize_gt_skimg = transform.resize(
                gt_i,
                (image_size, image_size),
                order=0,
                preserve_range=True,
                mode="constant",
                anti_aliasing=False,
            )
            resize_gt_skimg = np.uint8(resize_gt_skimg)
            assert resize_img_skimg_01.shape[:2] == resize_gt_skimg.shape
            np.save(
                join(
                    npy_path,
                    "imgs",
                    prefix
                    + gt_name.split(gt_name_suffix)[0]
                    + "-"
                    + str(i).zfill(3)
                    + ".npy",
                ),
                resize_img_skimg_01,
            )
            np.save(
                join(
                    npy_path,
                    "gts",
                    prefix
                    + gt_name.split(gt_name_suffix)[0]
                    + "-"
                    + str(i).zfill(3)
                    + ".npy",
                ),
                resize_gt_skimg,
            )
