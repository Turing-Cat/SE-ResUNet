#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
抓取框转换为四张标签图的详细演示
演示如何将抓取矩形转换为网络训练所需的四通道标签图像
参考visualize_label.py，直接读取数据集中的真实图像和标签进行可视化
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon
from utils.data import get_dataset

def demonstrate_real_dataset_conversion(dataset_name, dataset_path, index=0):
    """
    使用真实数据集演示抓取框转换为四张标签图的完整过程
    
    Args:
        dataset_name (str): 数据集名称 ('cornell' 或 'jacquard')
        dataset_path (str): 数据集路径
        index (int): 数据集索引
    """
    print("=" * 80)
    print(f"真实数据集 {dataset_name} 中抓取框转换为四张标签图的详细过程")
    print("=" * 80)
    
    # 1. 加载数据集
    print("\n1. 加载数据集")
    print("-" * 40)
    
    Dataset = get_dataset(dataset_name)
    dataset = Dataset(
        dataset_path,
        include_depth=True,
        include_rgb=True,
        output_size=300
    )
    
    print(f"数据集类型: {dataset_name}")
    print(f"数据集大小: {len(dataset)}")
    print(f"输出尺寸: 300x300")
    print(f"当前索引: {index}")
    
    # 2. 获取原始数据和标签
    print("\n2. 获取原始数据和标签")
    print("-" * 40)
    
    # 获取网络输入和标签
    x, y, idx, rot, zoom = dataset[index]
    
    # 解包标签 (已经是最终的四通道标签)
    pos, cos, sin, width = y
    
    # 转换为numpy数组
    pos = pos.numpy().squeeze()
    cos = cos.numpy().squeeze()
    sin = sin.numpy().squeeze()
    width = width.numpy().squeeze()
    
    print(f"输入张量形状: {x.shape}")
    print(f"标签形状: pos={pos.shape}, cos={cos.shape}, sin={sin.shape}, width={width.shape}")
    print(f"旋转角度: {rot:.3f} 弧度 ({np.degrees(rot):.1f}度)")
    print(f"缩放因子: {zoom:.3f}")
    
    # 3. 获取原始图像
    print("\n3. 获取原始图像")
    print("-" * 40)
    
    # 获取RGB和深度图像
    if dataset.include_rgb and dataset.include_depth:
        # 从输入张量中提取
        depth_img = x[0].numpy()  # 第一个通道是深度
        rgb_img = x[1:].numpy().transpose(1, 2, 0)  # 后三个通道是RGB
    else:
        depth_img = dataset.get_depth(index)
        if dataset.include_rgb:
            rgb_img = dataset.get_rgb(index, normalise=False)
            if len(rgb_img.shape) == 3 and rgb_img.shape[0] == 3:
                rgb_img = rgb_img.transpose(1, 2, 0)
        else:
            rgb_img = None
    
    print(f"深度图像范围: {depth_img.min():.3f} 到 {depth_img.max():.3f}")
    if rgb_img is not None:
        print(f"RGB图像形状: {rgb_img.shape}")
        print(f"RGB图像范围: {rgb_img.min():.3f} 到 {rgb_img.max():.3f}")
    
    # 4. 获取原始抓取矩形
    print("\n4. 分析抓取矩形转换")
    print("-" * 40)
    
    # 获取原始抓取矩形（未经变换）
    original_grasps = dataset.get_gtbb(index, rot=0, zoom=1.0)
    # 获取变换后的抓取矩形
    transformed_grasps = dataset.get_gtbb(index, rot, zoom)
    
    print(f"原始抓取矩形数量: {len(original_grasps.grs)}")
    print(f"变换后抓取矩形数量: {len(transformed_grasps.grs)}")
    
    # 显示前几个抓取矩形的属性
    print("\n原始抓取矩形属性:")
    for i, gr in enumerate(original_grasps.grs[:3]):
        print(f"  矩形 {i+1}: 中心{gr.center.astype(int)}, 角度{gr.angle:.3f}弧度({np.degrees(gr.angle):.1f}度)")
    
    print("\n变换后抓取矩形属性:")
    for i, gr in enumerate(transformed_grasps.grs[:3]):
        print(f"  矩形 {i+1}: 中心{gr.center.astype(int)}, 角度{gr.angle:.3f}弧度({np.degrees(gr.angle):.1f}度)")
    
    # 5. 演示标签生成过程
    print("\n5. 标签生成过程演示")
    print("-" * 40)
    
    # 手动生成标签以演示过程
    pos_manual, ang_manual, width_manual = transformed_grasps.draw((300, 300))
    
    # 转换角度为cos和sin
    cos_manual = np.cos(2 * ang_manual)
    sin_manual = np.sin(2 * ang_manual)
    
    # 宽度归一化
    width_manual_norm = np.clip(width_manual, 0.0, 150) / 150
    
    print("GraspRectangles.draw()方法的工作流程:")
    print("1. 遍历所有抓取矩形")
    print("2. 对每个矩形计算压缩区域（长度缩小为1/3）")
    print("3. 在压缩区域填充标签值：")
    print("   - pos_img: 填充1.0（抓取质量）")
    print("   - ang_img: 填充角度值")
    print("   - width_img: 填充长度值")
    print("4. 后处理：角度→cos(2θ), sin(2θ)；宽度→归一化")
    
    print(f"\n手动生成的标签与数据集标签的差异:")
    print(f"  pos最大差异: {np.abs(pos - pos_manual).max():.6f}")
    print(f"  cos最大差异: {np.abs(cos - cos_manual).max():.6f}")
    print(f"  sin最大差异: {np.abs(sin - sin_manual).max():.6f}")
    print(f"  width最大差异: {np.abs(width - width_manual_norm).max():.6f}")
    
    # 6. 可视化结果
    print("\n6. 可视化结果")
    print("-" * 40)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # 第一行：原始图像和抓取矩形
    # RGB图像
    ax = axes[0, 0]
    if rgb_img is not None:
        rgb_display = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
        ax.imshow(rgb_display)
    else:
        ax.imshow(np.zeros((300, 300, 3)))
    ax.set_title('RGB Image')
    ax.axis('off')
    
    # 深度图像
    ax = axes[0, 1]
    ax.imshow(depth_img, cmap='gray')
    ax.set_title('Depth Image')
    ax.axis('off')
    
    # 抓取矩形叠加在RGB图像上
    ax = axes[0, 2]
    if rgb_img is not None:
        ax.imshow(rgb_display)
    else:
        ax.imshow(np.zeros((300, 300, 3)))
    
    # 绘制变换后的抓取矩形
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    for i, gr in enumerate(transformed_grasps.grs[:min(8, len(transformed_grasps.grs))]):
        # 绘制完整矩形
        rect_points = np.vstack([gr.points, gr.points[0]])
        ax.plot(rect_points[:, 1], rect_points[:, 0], '-', 
                color=colors[i % len(colors)], linewidth=2, alpha=0.8)
        
        # 绘制压缩区域（用于标签生成）
        compact_gr = gr.as_grasp
        compact_gr.length = gr.length / 3  # 压缩长度为原长度的1/3
        compact_rect = compact_gr.as_gr
        compact_points = np.vstack([compact_rect.points, compact_rect.points[0]])
        ax.plot(compact_points[:, 1], compact_points[:, 0], '--', 
                color=colors[i % len(colors)], linewidth=1, alpha=0.6)
    
    ax.set_title('Grasp Rectangles\n(solid: full, dashed: compact)')
    ax.axis('off')
    
    # 抓取矩形框架图
    ax = axes[0, 3]
    ax.imshow(np.zeros((300, 300)), cmap='gray')
    for i, gr in enumerate(transformed_grasps.grs[:min(6, len(transformed_grasps.grs))]):
        rect_points = np.vstack([gr.points, gr.points[0]])
        ax.plot(rect_points[:, 1], rect_points[:, 0], '-', 
                color=colors[i % len(colors)], linewidth=2, 
                label=f'Grasp {i+1}')
        
        # 标注中心点
        center = gr.center
        ax.plot(center[1], center[0], 'o', color=colors[i % len(colors)], markersize=4)
    
    ax.set_title('Grasp Rectangles Only')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.set_xlim(0, 300)
    ax.set_ylim(300, 0)
    
    # 第二行：四张标签图
    # 抓取质量图
    ax = axes[1, 0]
    im1 = ax.imshow(pos, cmap='jet', vmin=0, vmax=1)
    ax.set_title('Grasp Quality (pos)')
    ax.axis('off')
    plt.colorbar(im1, ax=ax, shrink=0.8)
    
    # 角度cos图
    ax = axes[1, 1]
    im2 = ax.imshow(cos, cmap='RdYlGn', vmin=-1, vmax=1)
    ax.set_title('Angle Cosine (cos 2θ)')
    ax.axis('off')
    plt.colorbar(im2, ax=ax, shrink=0.8)
    
    # 角度sin图
    ax = axes[1, 2]
    im3 = ax.imshow(sin, cmap='hsv', vmin=-1, vmax=1)
    ax.set_title('Angle Sine (sin 2θ)')
    ax.axis('off')
    plt.colorbar(im3, ax=ax, shrink=0.8)
    
    # 宽度图
    ax = axes[1, 3]
    im4 = ax.imshow(width, cmap='jet', vmin=0, vmax=1)
    ax.set_title('Grasp Width (normalized)')
    ax.axis('off')
    plt.colorbar(im4, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    plt.suptitle(f'Grasp Rectangle to 4-Channel Labels: {dataset_name} Dataset (Index: {index})', fontsize=16)
    plt.subplots_adjust(top=0.93)
    
    # 保存图像
    save_name = f'real_dataset_conversion_{dataset_name}_{index}.png'
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"可视化图像已保存为 '{save_name}'")
    
    plt.show()
    
    # 7. 总结转换过程
    print("\n7. 抓取框到标签图转换过程总结")
    print("-" * 40)
    print("完整的转换流程:")
    print("┌─ 1. 输入：抓取矩形四个角点坐标")
    print("├─ 2. 数据增强：随机旋转和缩放")
    print("├─ 3. 矩形变换：对抓取矩形应用相同变换")
    print("├─ 4. 压缩区域：长度缩小为1/3，宽度不变")
    print("├─ 5. 标签生成：")
    print("│   ├─ pos_img: 在压缩区域填充1.0")
    print("│   ├─ ang_img: 在压缩区域填充角度值")
    print("│   └─ width_img: 在压缩区域填充长度值")
    print("├─ 6. 后处理：")
    print("│   ├─ 角度转换: θ → cos(2θ), sin(2θ)")
    print("│   └─ 宽度归一化: [0, max_width] → [0, 1]")
    print("└─ 7. 输出：四通道标签张量")
    
    print("\n关键设计点:")
    print("• 压缩区域：避免标签区域过大，提高训练精度")
    print("• 2倍角度：cos(2θ), sin(2θ)处理角度周期性")
    print("• 数据增强：提高网络的旋转和尺度不变性")
    print("• 多矩形覆盖：后面的矩形会覆盖前面的标签值")

def explain_draw_method():
    """
    详细解释GraspRectangles.draw()方法的核心逻辑
    """
    print("\n" + "=" * 80)
    print("GraspRectangles.draw()方法核心代码解析")
    print("=" * 80)
    
    code_explanation = '''
def draw(self, shape, position=True, angle=True, width=True):
    """将抓取矩形转换为像素级标签图像"""
    
    # 步骤1: 初始化输出图像
    if position:
        pos_out = np.zeros(shape)    # 抓取质量图
    if angle:
        ang_out = np.zeros(shape)    # 角度图
    if width:
        width_out = np.zeros(shape)  # 宽度图
        
    # 步骤2: 遍历所有抓取矩形
    for gr in self.grs:
        # 步骤3: 获取压缩区域的像素坐标
        # compact_polygon_coords() 返回压缩区域内的所有像素坐标
        rr, cc = gr.compact_polygon_coords(shape)
        
        # 步骤4: 在相应位置填充标签值
        if position:
            pos_out[rr, cc] = 1.0        # 质量标签：1.0表示可抓取
        if angle:
            ang_out[rr, cc] = gr.angle   # 角度标签：抓取方向
        if width:
            width_out[rr, cc] = gr.length # 宽度标签：抓取器张开度
            
    # 步骤5: 返回三张基础标签图
    return pos_out, ang_out, width_out

# compact_polygon_coords()的关键实现：
def compact_polygon_coords(self, shape=None):
    """返回压缩区域（中心1/3区域）的像素坐标"""
    return Grasp(self.center, self.angle, self.length / 3, self.width).as_gr.polygon_coords(shape)
    '''
    
    print("方法实现解析:")
    print(code_explanation)
    
    print("\n核心要点:")
    print("1. 🔹 使用压缩区域而非完整矩形")
    print("   - 压缩区域 = 原长度的1/3 + 原宽度")
    print("   - 目的：避免标签区域过大，提高定位精度")
    
    print("\n2. 🔹 像素级标签填充")
    print("   - 使用polygon()函数获取区域内所有像素坐标")
    print("   - 在这些像素位置填充对应的标签值")
    
    print("\n3. 🔹 多矩形处理策略")
    print("   - 多个抓取矩形可以重叠")
    print("   - 后处理的矩形会覆盖前面的标签值")
    print("   - 这样可以处理一个物体有多个抓取点的情况")
    
    print("\n4. 🔹 后续处理步骤")
    print("   - ang_out → cos(2*ang_out), sin(2*ang_out)")
    print("   - width_out → clip + 归一化到[0,1]")
    print("   - 最终得到4通道标签：pos, cos, sin, width")

def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser(description='抓取框转换为四张标签图的详细演示')
    parser.add_argument('--dataset', type=str, default='cornell', 
                        choices=['cornell', 'jacquard'],
                        help='数据集名称')
    parser.add_argument('--dataset-path', type=str, required=True,
                        help='数据集路径')
    parser.add_argument('--index', type=int, default=0,
                        help='数据集索引')
    
    args = parser.parse_args()
    
    # 运行真实数据集演示
    demonstrate_real_dataset_conversion(args.dataset, args.dataset_path, args.index)
    
    # 解释核心方法
    explain_draw_method()

if __name__ == '__main__':
    main()
