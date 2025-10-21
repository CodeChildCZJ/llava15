# config.py
import argparse

def get_args():
    parser = argparse.ArgumentParser("LLaVA-1.5 GQA评估")
    # 模型加载
    parser.add_argument("--model_path", type=str, default="/home/czj/llava15_test/llava-v1.5-7b")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")

    # 超参数
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=64, help="model生成最大tokens数")


    # GPU并行
    parser.add_argument("--multi_gpu", action="store_true", help="使用所有可用GPU进行并行推理")

    # mask选项
    parser.add_argument("--mask_visual_token", action="store_true")
    parser.add_argument("--mask_ratio", type=float, default=0.1)
    parser.add_argument("--mask_strategy", type=str, default="random",
                        choices=["random", "attention", "position_center", "position_edge"])
    parser.add_argument("--mask_token_value", type=float, default=0.0)

    # 数据集
    parser.add_argument("--dataset_path", type=str, default="/home/Dataset/Dataset/GQA",
            help="GQA 数据源路径或名称，例如 '/home/Dataset/Dataset/GQA' 或 'lmms-lab/GQA'")
    parser.add_argument("--split", type=str, default="testdev", choices=["train", "val", "testdev", "challenge", "submission", "test", "train_all", "val_balanced"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--conv_mode", type=str, default="llava_v1")

    # 输出
    parser.add_argument("--output_dir", type=str, default="./eval_GQA_results/testdev_VQA_100samples")
    parser.add_argument("--save_json", action="store_true")
    parser.add_argument("--save_pred_gt", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()
