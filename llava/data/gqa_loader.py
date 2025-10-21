"""
智能 GQA 数据加载器（完整版本）
支持本地 parquet 与 HuggingFace 远程数据集。
自动识别 train/val/testdev/challenge/submission 等结构。
"""

import os
import pandas as pd
from io import BytesIO
from typing import List, Dict, Optional
from PIL import Image
from datasets import Dataset, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor, as_completed



class GQALoader:
    """统一的 GQA 数据加载器"""

    def __init__(self, data_source: str, num_workers: int = 4, batch_size: int = 8, use_cache: bool = True):
        self.data_source = data_source.strip()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.use_cache = use_cache

        self.is_local = os.path.exists(self.data_source)
        if self.is_local:
            self.gqa_root = self.data_source
            print(f"使用本地 GQA 数据集路径: {self.gqa_root}")
        else:
            self.dataset_name = self.data_source
            print(f"使用远程 GQA 数据集: {self.dataset_name}")

    # ------------------------------------------------------------------
    def _find_best_match_dir(self, split_base: str, kind: str) -> Optional[str]:
        """
        智能匹配文件夹：
        支持 challenge / submission / testdev / train / val / test
        优先顺序：balanced > all
        """
        assert kind in ["instructions", "images"]
        candidates = [
            f"{split_base}_balanced_{kind}",
            f"{split_base}_all_{kind}",
            f"{split_base}_{kind}",
        ]
        for cand in candidates:
            path = os.path.join(self.gqa_root, cand)
            if os.path.exists(path):
                return path
        return None

    # ------------------------------------------------------------------
    def _parallel_read_parquets(self, file_list, key_name="id") -> Dict[str, dict]:
        """并行读取多个 parquet 文件"""
        image_dict = {}
        if not file_list:
            return image_dict

        with ThreadPoolExecutor(max_workers=min(self.num_workers * 2, len(file_list))) as ex:
            futures = {ex.submit(pd.read_parquet, f): f for f in file_list}
            for fut in as_completed(futures):
                df = fut.result()
                for _, row in df.iterrows():
                    image_dict[row[key_name]] = row["image"]

        print(f"并行加载完成，共 {len(image_dict)} 张图像")
        return image_dict

    # ------------------------------------------------------------------
    def load_dataset(self, split: str = "train_balanced", num_samples: Optional[int] = None) -> Dataset:
        """加载本地或远程数据"""
        if self.is_local:
            return self._load_local_dataset(split, num_samples)
        else:
            return self._load_remote_dataset(split, num_samples)

    # ------------------------------------------------------------------
    def _load_local_dataset(self, split: str, num_samples: Optional[int], use_cache: bool = True) -> Dataset:
        """加载本地 GQA parquet 数据（Arrow 缓存版）"""
        print(f"加载本地 GQA split: {split}")

        split_base = split.replace("_balanced", "").replace("_all", "")
        instructions_dir = self._find_best_match_dir(split_base, "instructions")
        images_dir = self._find_best_match_dir(split_base, "images")

        if not instructions_dir or not images_dir:
            raise FileNotFoundError(
                f"未找到匹配目录，请检查数据路径。\n"
                f"  根路径: {self.gqa_root}\n"
                f"  split: {split}\n"
                f"  可选: train / val / testdev / challenge / submission / test"
            )

        cache_dir = os.path.join(self.gqa_root, ".gqa_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{split_base}_arrow")

        # 若存在 Arrow 缓存
        if use_cache and os.path.exists(cache_path):
            print(f"从 Arrow 缓存加载: {cache_path}")
            dataset = load_from_disk(cache_path)
            if num_samples:
                dataset = dataset.select(range(min(num_samples, len(dataset))))
            print(f"已加载 {len(dataset)} 条样本（来自缓存）")
            return dataset

        # 首次加载 parquet
        print(f"首次加载 split={split}，读取 parquet 文件...")
        inst_files = [os.path.join(instructions_dir, f) for f in os.listdir(instructions_dir) if f.endswith(".parquet")]
        img_files = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith(".parquet")]

        if not inst_files or not img_files:
            raise FileNotFoundError("在指定目录中未找到 parquet 文件")

        instruction_df = pd.read_parquet(inst_files[0])
        print(f"指令数据: {len(instruction_df)} 条")

        image_dict = self._parallel_read_parquets(img_files, "id")
        print(f"图像数据: {len(image_dict)} 张")

        samples = []
        for _, row in instruction_df.iterrows():
            if num_samples and len(samples) >= num_samples:
                break
            img_id = row.get("imageId")
            if not img_id or img_id not in image_dict:
                continue
            samples.append({
                "id": row.get("id", ""),
                "imageId": img_id,
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "image_data": image_dict[img_id],
            })

        dataset = Dataset.from_list(samples)

        # 缓存为 Arrow 格式
        dataset.save_to_disk(cache_path)
        print(f"已缓存 {len(samples)} 条样本到 {cache_path}（Arrow 格式）")

        print(f"成功加载 {len(samples)} 条样本")
        return dataset


    # ------------------------------------------------------------------
    def _load_remote_dataset(self, split: str, num_samples: Optional[int]) -> Dataset:
        """加载远程 HuggingFace 数据集"""
        print(f"加载远程 GQA 数据集: {self.dataset_name}, split={split}")
        ds = load_dataset(self.dataset_name, split=split)
        if num_samples:
            ds = ds.select(range(min(num_samples, len(ds))))
        print(f"远程数据加载完成，共 {len(ds)} 条")
        return ds

    # ------------------------------------------------------------------
    def process_sample(self, sample: Dict) -> Dict:
        """加载单个样本图像"""
        processed = sample.copy()
        image = None
        try:
            if "image" in sample and isinstance(sample["image"], Image.Image):
                image = sample["image"]
            elif "image_data" in sample:
                data = sample["image_data"]
                if isinstance(data, dict) and "bytes" in data:
                    image = Image.open(BytesIO(data["bytes"])).convert("RGB")
            elif "image_path" in sample and os.path.exists(sample["image_path"]):
                image = Image.open(sample["image_path"]).convert("RGB")
        except Exception as e:
            print(f"图像加载失败: {e}")
        processed["image"] = image
        return processed

     # ------------------------------------------------------------------
    def _decode_image(self, data):
        """安全地从 bytes 解码图像"""
        try:
            if isinstance(data, dict) and "bytes" in data:
                return Image.open(BytesIO(data["bytes"])).convert("RGB")
        except Exception:
            return None
        return None
    
    def collate_fn(self, batch):
        """并行 decode 图像"""
        def decode_safe(b):
            try:
                if "image_data" in b and b["image_data"] is not None:
                    b["image"] = self._decode_image(b["image_data"])
                else:
                    b["image"] = None
            except Exception:
                b["image"] = None
            return b

        with ThreadPoolExecutor(max_workers=min(self.num_workers, len(batch))) as ex:
            batch = list(ex.map(decode_safe, batch))
        return batch

    # ------------------------------------------------------------------
    def as_dataloader(self, split: str, num_samples: Optional[int] = None):
        dataset = self._load_local_dataset(split, num_samples) if self.is_local else self._load_remote_dataset(split, num_samples)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
        print(f"DataLoader 就绪: batch_size={self.batch_size}, num_workers={self.num_workers}")
        return dataloader


def test_split(split_name: str, num_samples: int = None):
    print("\n" + "=" * 80)
    print(f"🚀 测试加载 split = '{split_name}'")
    print("=" * 80)

    loader = GQALoader("/home/Dataset/Dataset/GQA")

    try:
        import time
        begin_time = time.time()
        dataset = loader.load_dataset(split=split_name, num_samples=num_samples)
        print(f"✅ 成功加载 {len(dataset)} 个样本")
        end_time = time.time()
        print(f"加载时间: {end_time - begin_time} 秒")

        if len(dataset) > 0:
            sample = dataset[0]
            print(f"📋 样本键: {list(sample.keys())}")
            print(f"🧠 问题: {sample.get('question')}")
            print(f"🎯 答案: {sample.get('answer')}")
            print(f"🖼️ 图像ID: {sample.get('imageId')}")
        else:
            print("⚠️ 数据集为空，请检查 parquet 文件内容。")

    except Exception as e:
        print(f"❌ 加载 split='{split_name}' 失败: {e}")


if __name__ == "__main__":
    # 依次测试各种分割类型
    splits_to_test = [
        "train", "val", 
        "testdev",
        "challenge", "submission",
        "test", "train_all", "val_balanced"
    ]



    for s in splits_to_test:
        test_split(s)