# gqa_eval/prompt.py
import torch
from concurrent.futures import ThreadPoolExecutor
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token


def build_prompt(question: str, conv_mode: str = "llava_v1") -> str:
    """构造单样本 prompt"""
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def build_multimodal_batch_inputs(
    batch, tokenizer, image_processor, device, dtype=torch.float16, conv_mode="llava_v1", max_workers=8
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


    tok = tokenizer_image_token(
        prompts,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
        padding_side=tokenizer.padding_side # 或 "left"
    )
    input_ids = tok["input_ids"]
    attention_mask = tok["attention_mask"]


    # 3 图像并行预处理
    def _process_img(s):
        image = s.get("image", None)
        if image is None:
            return None
        if image.mode != "RGB":
            image = image.convert("RGB")
        try:
            return image_processor(image, return_tensors="pt")["pixel_values"].to(device, dtype)
        except Exception as e:
            print(f"[warn] 图像处理失败: {e}")
            return None

    with ThreadPoolExecutor(max_workers=min(max_workers, len(batch))) as ex:
        image_tensors = list(ex.map(_process_img, batch))   

    image_sizes = [s["image"].size for s in batch]
    # print(input_ids, image_tensors)
    return input_ids, attention_mask, torch.cat(image_tensors, dim=0), image_sizes