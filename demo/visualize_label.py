#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化脚本：展示标签图像的生成过程
此脚本演示了如何通过bbs.draw()方法将抓取矩形转换为标签图像


python visualize_label.py --dataset-path "E:\Dataset\low-light-cornell-new\" --dataset cornell --index 1
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
from utils.data import get_dataset

def visualize_labels(dataset_name, dataset_path, index=0):
    """
    可视化指定数据集的标签图像
    
    Args:
        dataset_name (str): 数据集名称 ('cornell' 或 'jacquard')
        dataset_path (str): 数据集路径
        index (int): 数据集索引
    """
    # 加载数据集
    Dataset = get_dataset(dataset_name)
    dataset = Dataset(
        dataset_path,
        include_depth=True,
        include_rgb=True,
        output_size=300
    )
    
    # 获取指定索引的数据
    x, y, _, _, _ = dataset[index]
    
    # 解包标签
    pos, cos, sin, width = y
    
    # 将torch张量转换为numpy数组用于可视化
    pos = pos.numpy().squeeze()
    cos = cos.numpy().squeeze()
    sin = sin.numpy().squeeze()
    width = width.numpy().squeeze()
    
    # 同时获取原始图像用于对比
    # 注意：这里我们需要处理两种情况
    # 1. x包含深度和RGB信息，形状为(4, 300, 300) - 第1个通道是深度，后3个是RGB
    # 2. 分别获取RGB图像
    if dataset.include_rgb and dataset.include_depth:
        # 从x中提取RGB部分，x的形状是(4, 300, 300)
        rgb_img = x[1:].numpy().transpose(1, 2, 0)  # 转换为(300, 300, 3)
    elif dataset.include_rgb:
        # 只有RGB图像
        rgb_img = dataset.get_rgb(index, normalise=False)
        if len(rgb_img.shape) == 3 and rgb_img.shape[0] == 3:
            # 如果是(3, H, W)格式，转换为(H, W, C)
            rgb_img = rgb_img.transpose(1, 2, 0)
    else:
        rgb_img = None
    
    depth_img = dataset.get_depth(index)
    
    # 创建可视化图表
    fig = plt.figure(figsize=(15, 10))
    
    # 显示原始RGB图像
    ax1 = plt.subplot(2, 3, 1)
    if rgb_img is not None:
        # 归一化到0-1范围用于显示
        rgb_display = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
        ax1.imshow(rgb_display)
    else:
        ax1.imshow(np.zeros((300, 300, 3)))  # 如果没有RGB图像，显示黑色
    ax1.set_title('RGB Image')
    ax1.axis('off')
    
    # 显示深度图像
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(depth_img, cmap='gray')
    ax2.set_title('Depth Image')
    ax2.axis('off')
    
    # 显示抓取质量图 (pos)
    ax3 = plt.subplot(2, 3, 3)
    im3 = ax3.imshow(pos, cmap='jet', vmin=0, vmax=1)
    ax3.set_title('Grasp Quality (pos)')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3)
    
    # 显示角度cos值图
    ax4 = plt.subplot(2, 3, 4)
    im4 = ax4.imshow(cos, cmap='RdYlGn', vmin=-1, vmax=1)
    ax4.set_title('Angle Cos (cos)')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4)
    
    # 显示角度sin值图
    ax5 = plt.subplot(2, 3, 5)
    im5 = ax5.imshow(sin, cmap='hsv', vmin=-1, vmax=1)
    ax5.set_title('Angle Sin (sin)')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5)
    
    # 显示抓取宽度图
    ax6 = plt.subplot(2, 3, 6)
    im6 = ax6.imshow(width, cmap='jet', vmin=0, vmax=1)
    ax6.set_title('Grasp Width')
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6)
    
    plt.tight_layout()
    plt.suptitle(f'Label Visualization for {dataset_name} Dataset (Index: {index})', fontsize=16)
    plt.subplots_adjust(top=0.93)
    
    # 保存图像
    plt.savefig(f'label_visualization_{dataset_name}_{index}.png', dpi=300, bbox_inches='tight')
    print(f"标签图像已保存为 'label_visualization_{dataset_name}_{index}.png'")
    
    # 显示图像
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='可视化抓取检测标签')
    parser.add_argument('--dataset', type=str, default='cornell', 
                        choices=['cornell', 'jacquard'],
                        help='数据集名称')
    parser.add_argument('--dataset-path', type=str, required=True,
                        help='数据集路径')
    parser.add_argument('--index', type=int, default=0,
                        help='数据集索引')
    
    args = parser.parse_args()
    
    visualize_labels(args.dataset, args.dataset_path, args.index)

if __name__ == '__main__':
    main()