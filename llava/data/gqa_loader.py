"""
统一的GQA数据集加载器
支持本地parquet文件和HuggingFace远程数据
"""

import os
import json
from typing import List, Dict, Optional, Tuple, Union
import pandas as pd
from PIL import Image
import torch
from datasets import Dataset, load_dataset


class GQALoader:
    """统一的GQA数据集加载器"""
    
    def __init__(self, gqa_root: Optional[str] = None, dataset_name: str = "lmms-lab/GQA"):
        """
        初始化GQA数据集加载器
        
        Args:
            gqa_root: 本地GQA数据集根目录，如果为None则使用远程数据
            dataset_name: HuggingFace数据集名称
        """
        self.gqa_root = gqa_root
        self.dataset_name = dataset_name
        self.use_local = gqa_root is not None
        
        if self.use_local:
            self.available_splits = self._get_local_splits()
        else:
            self.available_splits = self._get_remote_splits()
    
    def _get_local_splits(self) -> List[str]:
        """获取本地可用的数据分割"""
        splits = []
        if os.path.exists(self.gqa_root):
            for item in os.listdir(self.gqa_root):
                if item.endswith('_instructions') and not item.startswith('.'):
                    split_name = item.replace('_instructions', '')
                    splits.append(split_name)
        return splits
    
    def _get_remote_splits(self) -> List[str]:
        """获取远程可用的数据分割"""
        try:
            # 尝试获取远程数据集信息
            dataset_info = load_dataset(self.dataset_name, split="train", streaming=True)
            return ["train", "validation", "test"]  # 默认分割
        except:
            return ["train", "validation", "test"]
    
    def load_dataset(self, split: str = "train_balanced", num_samples: Optional[int] = None) -> Dataset:
        """
        加载GQA数据集
        
        Args:
            split: 数据集分割
            num_samples: 加载的样本数量
            
        Returns:
            HuggingFace Dataset对象
        """
        if self.use_local:
            return self._load_local_dataset(split, num_samples)
        else:
            return self._load_remote_dataset(split, num_samples)
    
    def _load_local_dataset(self, split: str, num_samples: Optional[int]) -> Dataset:
        """加载本地parquet数据集"""
        print(f"加载本地GQA数据集: {split}")
        
        # 构建数据路径
        instructions_dir = os.path.join(self.gqa_root, f"{split}_instructions")
        images_dir = os.path.join(self.gqa_root, f"{split}_images")
        
        if not os.path.exists(instructions_dir):
            raise FileNotFoundError(f"指令目录不存在: {instructions_dir}")
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"图像目录不存在: {images_dir}")
        
        # 查找指令parquet文件
        instruction_files = [f for f in os.listdir(instructions_dir) if f.endswith('.parquet')]
        if not instruction_files:
            raise FileNotFoundError(f"在 {instructions_dir} 中未找到parquet文件")
        
        # 查找图像parquet文件
        image_files = [f for f in os.listdir(images_dir) if f.endswith('.parquet')]
        if not image_files:
            raise FileNotFoundError(f"在 {images_dir} 中未找到parquet文件")
        
        # 加载指令数据
        instruction_path = os.path.join(instructions_dir, instruction_files[0])
        print(f"加载指令parquet文件: {instruction_path}")
        
        instruction_df = pd.read_parquet(instruction_path)
        print(f"指令数据形状: {instruction_df.shape}")
        print(f"指令数据列: {list(instruction_df.columns)}")
        
        # 加载所有图像数据
        print(f"找到 {len(image_files)} 个图像parquet文件")
        image_dict = {}
        
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(images_dir, image_file)
            print(f"加载图像parquet文件 {i+1}/{len(image_files)}: {image_path}")
            
            image_df = pd.read_parquet(image_path)
            print(f"  文件 {i+1} 数据形状: {image_df.shape}")
            
            # 添加到图像映射
            for idx, row in image_df.iterrows():
                image_id = row['id']
                image_data = row['image']
                image_dict[image_id] = image_data
        
        print(f"总图像映射大小: {len(image_dict)}")
        
        # 合并数据
        data_list = []
        for idx, row in instruction_df.iterrows():
            if num_samples and len(data_list) >= num_samples:
                break
                
            # 获取图像ID
            image_id = row.get('imageId', '')
            if not image_id:
                print(f"样本 {idx} 没有imageId，跳过")
                continue
            
            # 从图像映射中获取图像数据
            if image_id not in image_dict:
                print(f"图像ID {image_id} 在图像数据中不存在，跳过")
                continue
            
            sample = {
                'id': row.get('id', ''),
                'imageId': image_id,
                'question': row.get('question', ''),
                'answer': row.get('answer', ''),
                'fullAnswer': row.get('fullAnswer', ''),
                'types': row.get('types', {}),
                'groups': row.get('groups', {}),
                'semantic': row.get('semantic', {}),
                'structural': row.get('structural', {}),
                'image_data': image_dict[image_id]  # 存储图像数据而不是路径
            }
            data_list.append(sample)
        
        print(f"成功加载 {len(data_list)} 个样本")
        
        # 转换为HuggingFace Dataset
        dataset = Dataset.from_list(data_list)
        return dataset
    
    def _load_remote_dataset(self, split: str, num_samples: Optional[int]) -> Dataset:
        """加载远程HuggingFace数据集"""
        print(f"加载远程GQA数据集: {self.dataset_name}, split: {split}")
        
        try:
            # 加载数据集
            dataset = load_dataset(
                self.dataset_name, 
                split=split,
                streaming=False
            )
            
            # 如果指定了样本数量，则截取
            if num_samples is not None and num_samples < len(dataset):
                dataset = dataset.select(range(num_samples))
            
            print(f"成功加载 {len(dataset)} 个样本")
            return dataset
            
        except Exception as e:
            print(f"加载远程数据集失败: {e}")
            raise
    
    def process_sample(self, sample: Dict) -> Dict:
        """
        处理单个样本，加载图像
        
        Args:
            sample: 原始样本数据
            
        Returns:
            处理后的样本数据
        """
        processed = sample.copy()
        
        # 加载图像
        if 'image' in sample and sample['image'] is not None:
            # 如果已经是PIL Image对象
            processed['image'] = sample['image']
        elif 'image_data' in sample and sample['image_data'] is not None:
            # 从parquet中的图像数据加载
            try:
                image_data = sample['image_data']
                if isinstance(image_data, dict) and 'bytes' in image_data:
                    # 从字节数据创建PIL图像
                    from io import BytesIO
                    image_bytes = image_data['bytes']
                    image = Image.open(BytesIO(image_bytes)).convert('RGB')
                    processed['image'] = image
                else:
                    print(f"图像数据格式不正确: {type(image_data)}")
                    processed['image'] = None
            except Exception as e:
                print(f"从图像数据加载图像失败: {e}")
                processed['image'] = None
        elif 'image_path' in sample and sample['image_path']:
            # 从路径加载图像（兼容性）
            image_path = sample['image_path']
            if os.path.exists(image_path):
                try:
                    image = Image.open(image_path).convert('RGB')
                    processed['image'] = image
                except Exception as e:
                    print(f"加载图像失败 {image_path}: {e}")
                    processed['image'] = None
            else:
                print(f"图像文件不存在: {image_path}")
                processed['image'] = None
        else:
            processed['image'] = None
        
        return processed
    
    def get_dataset_info(self, dataset: Dataset) -> Dict:
        """获取数据集信息"""
        info = {
            'total_samples': len(dataset),
            'features': list(dataset.features.keys()),
            'available_splits': self.available_splits,
            'use_local': self.use_local
        }
        return info


def test_gqa_loader():
    """测试GQA加载器"""
    print("测试GQA加载器...")
    
    # 测试本地加载器
    print("\n=== 测试本地加载器 ===")
    try:
        local_loader = GQALoader(gqa_root="/home/Dataset/Dataset/GQA")
        print(f"本地可用分割: {local_loader.available_splits}")
        
        # 加载小量数据测试
        dataset = local_loader.load_dataset(split="train_balanced", num_samples=3)
        
        # 获取数据集信息
        info = local_loader.get_dataset_info(dataset)
        print(f"本地数据集信息: {info}")
        
        # 处理第一个样本
        if len(dataset) > 0:
            sample = dataset[0]
            processed_sample = local_loader.process_sample(sample)
            print(f"处理后的样本键: {list(processed_sample.keys())}")
            print(f"图像是否加载成功: {processed_sample.get('image') is not None}")
        
        print("本地GQA加载器测试完成!")
        
    except Exception as e:
        print(f"本地加载器测试失败: {e}")
    
    # 测试远程加载器
    print("\n=== 测试远程加载器 ===")
    try:
        remote_loader = GQALoader()
        print(f"远程可用分割: {remote_loader.available_splits}")
        
        # 加载小量数据测试
        dataset = remote_loader.load_dataset(split="train", num_samples=3)
        
        # 获取数据集信息
        info = remote_loader.get_dataset_info(dataset)
        print(f"远程数据集信息: {info}")
        
        print("远程GQA加载器测试完成!")
        
    except Exception as e:
        print(f"远程加载器测试失败: {e}")


if __name__ == "__main__":
    test_gqa_loader()
