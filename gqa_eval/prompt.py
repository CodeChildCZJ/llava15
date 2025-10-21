# gqa_eval/prompt.py
import torch
from concurrent.futures import ThreadPoolExecutor
from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates


def build_prompt(question: str, conv_mode: str = "llava_v1") -> str:
    """构造单样本 prompt"""
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def build_multimodal_batch_inputs(
    batch,
    tokenizer,
    image_processor,
    device,
    dtype=torch.float16,
    conv_mode="llava_v1",
    max_workers: int = 8,
):
    """
    批量构建多模态输入 (文本 + 图像)
    - 文本部分批量 tokenizer padding
    - 图像部分多线程 image_processor 预处理
    """

    # 1 文本批量生成 prompt
    prompts = []
    for s in batch:
        q = s["question"]
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + q)
        conv.append_message(conv.roles[1], None)
        prompts.append(conv.get_prompt())

    # 2 Tokenizer 批量编码（自动 padding）
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt"
    ).to(device)

    # 3 图像并行预处理
    def _process_img(s):
        image = s.get("image", None)
        if image is None:
            return None
        if image.mode != "RGB":
            image = image.convert("RGB")
        try:
            return image_processor(image, return_tensors="pt")["pixel_values"]
        except Exception as e:
            print(f"[warn] 图像处理失败: {e}")
            return None

    with ThreadPoolExecutor(max_workers=min(max_workers, len(batch))) as ex:
        img_tensors = list(ex.map(_process_img, batch))

    img_tensors = [x for x in img_tensors if x is not None]
    image_tensors = torch.cat(img_tensors, dim=0).to(device, dtype=dtype, non_blocking=True)

    return encoded["input_ids"], image_tensors
