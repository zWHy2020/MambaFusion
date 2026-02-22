# collect_test_files.py
import json
import os
import shutil
from tqdm import tqdm

# --- 配置 ---
# processed_data 文件夹的路径
data_root = './data/processed_data' 
# test.json 文件的路径
json_path = os.path.join(data_root, 'test.json')
# 新建一个文件夹，专门存放所有测试集图片
output_dir = './inference_test_set'

# --- 主逻辑 ---
def main():
    print(f"正在读取索引文件: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        test_samples = json.load(f)

    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)
    print(f"将把所有测试图片文件复制到: {output_dir}")

    copied_count = 0
    # 遍历 test.json 中的每一个样本
    for sample in tqdm(test_samples, desc="正在复制文件"):
        # 获取原始RGB文件路径
        rgb_relative_path = sample['rgb_path']
        rgb_source_path = os.path.join(data_root, rgb_relative_path)

        # ### --- 修改点 1：不再需要处理HSI路径 --- ###
        # # 获取原始HSI文件路径 (如果有的话)
        # hsi_relative_path = sample['hsi_path']
        # hsi_source_path = os.path.join(data_root, hsi_relative_path)
        
        # 定义目标路径 (只定义RGB的)
        rgb_dest_path = os.path.join(output_dir, os.path.basename(rgb_source_path))

        # 复制RGB文件
        if os.path.exists(rgb_source_path):
            shutil.copy(rgb_source_path, rgb_dest_path)
            copied_count += 1
        
        # ### --- 修改点 2：删除复制HSI文件的代码 --- ###
        # # 复制HSI文件
        # if os.path.exists(hsi_source_path):
        #     shutil.copy(hsi_source_path, hsi_dest_path)

    print(f"\n复制完成！共复制了 {copied_count} 个RGB图片文件。")
    print(f"现在您可以使用下面的指令对 '{output_dir}' 文件夹进行批量推理。")

if __name__ == '__main__':
    main()