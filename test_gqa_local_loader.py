import os, sys, json, torch, random
import numpy as np
from pathlib import Path
from PIL import Image
from eval.pred_gt_match import compute_match

# 固定随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(42)

sys.path.append("/home/czj/llava15_test/LLaVA")

from llava.model.builder import load_pretrained_model
from llava.data.gqa_loader import GQALoader
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN



model_path = "/home/czj/llava15_test/llava-v1.5-7b"
clip_path = "/home/czj/llava15_test/clip-vit-large-patch14-336"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    device_map="cuda:1",
    torch_dtype=torch.float16,
    use_masked_model=True  # 使用masked模型
)

print("✅ 模型加载完成")

import pickle

CACHE_FILE = "gqa_cache.pkl"
if Path(CACHE_FILE).exists():
    print("✅ 从缓存加载数据样本...")
    with open(CACHE_FILE, "rb") as f:
        dataset_cache = pickle.load(f)
else:
    print("🚀 第一次加载数据集...")
    gqa_loader = GQALoader(gqa_root="/home/Dataset/Dataset/GQA")
    dataset = gqa_loader.load_dataset(split="train_balanced", num_samples=10)
    dataset_cache = [gqa_loader.process_sample(dataset[i]) for i in range(min(10, len(dataset)))]

    
    # 存缓存
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(dataset_cache, f)
    print("✅ 已缓存前10个样本到 gqa_cache.pkl")

print(f"缓存样本数: {len(dataset_cache)}")
sample = dataset_cache[0]
sample.keys()



sample = dataset_cache[0]
image = sample["image"].convert("RGB")

image_tensor = image_processor(image, return_tensors="pt")["pixel_values"].to(model.device, dtype=torch.float16)

print(f"[DEBUG] 原图尺寸: {image.size}, tensor形状: {image_tensor.shape}")



from llava.conversation import conv_templates
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.mm_utils import tokenizer_image_token

question = sample["question"]

conv = conv_templates["llava_v1"].copy()
conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)

print(f"[DEBUG] input_ids 形状: {input_ids.shape}")
print(f"[DEBUG] <image> token 出现次数: {(input_ids == IMAGE_TOKEN_INDEX).sum().item()}")


model.eval()
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        images=image_tensor,
        output_hidden_states=True,
        return_dict=True,
    )

print(f"[DEBUG] Last hidden shape: {outputs.hidden_states[-1].shape}")


with torch.no_grad():
    output_tokens = model.generate(
        inputs=input_ids,
        images=image_tensor,
        max_new_tokens=64,
        do_sample=False,
        temperature=0,
        top_p=None,
        num_beams=1
    )

answer = tokenizer.batch_decode(output_tokens[0], skip_special_tokens=True)[0]
print("🧩 模型输出:", answer)


