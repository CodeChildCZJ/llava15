
import torch
import argparse
import re
import os
from io import BytesIO
import requests
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
def disable_torch_init():
    import torch
    import math
    torch.set_grad_enabled(False)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)


# ------------------ util ------------------

def image_parser(args):
    return args.image_file.split(args.sep)

def load_image(image_file):
    if image_file.startswith("http"):
        resp = requests.get(image_file)
        return Image.open(BytesIO(resp.content)).convert("RGB")
    else:
        return Image.open(image_file).convert("RGB")

def load_images(image_files):
    return [load_image(x) for x in image_files]

# ------------------ hook ------------------

import re
import torch

def register_attention_hooks(model):
    """
    注册 forward hook 到 LLaMA 语言模型的 self_attn 层，
    用于抓取注意力权重（不包括 vision_tower 的部分）。
    """
    attn_cache = {}

    def save_attn_hook(module, inp, out):
        # 某些层只返回 hidden_states，不返回 attn_weights
        if isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
            attn = out[1].detach().cpu()
            attn_cache.setdefault(module.layer_idx, []).append(attn)

    num_hooks = 0

    # ✅ 只 hook LLaMA 的文本部分
    for name, m in model.model.named_modules():
        # 跳过 vision_tower 的层，只保留 LLaMA 自身的 self_attn
        if re.match(r"layers\.\d+\.self_attn$", name):
            layer_idx = int(re.findall(r"layers\.(\d+)", name)[0])
            m.layer_idx = layer_idx
            m.register_forward_hook(save_attn_hook)
            num_hooks += 1
            print(f"Hooked attention layer {layer_idx}: {name}")

    print(f"\n✅ Registered {num_hooks} attention hooks in LLaMA layers.")
    return attn_cache



# ------------------ main eval ------------------

def eval_model_with_attn(args):
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    # build text prompt
    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        qs = re.sub(IMAGE_PLACEHOLDER, image_token_se if model.config.mm_use_im_start_end else DEFAULT_IMAGE_TOKEN, qs)
    else:
        qs = (image_token_se if model.config.mm_use_im_start_end else DEFAULT_IMAGE_TOKEN) + "\n" + qs

    # choose conversation template
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    args.conv_mode = args.conv_mode or conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # image
    image_files = image_parser(args)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    image_tensors = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)

    tokenized = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    input_ids = tokenized["input_ids"].to(model.device)
    attention_mask = tokenized.get("attention_mask", None)

    print(f"input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}")


    # for name, _ in model.named_modules():
    #     print(name)

    # register hooks
    model.config.output_attentions = True
    attn_cache = register_attention_hooks(model)


    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            output_attentions=True,
            return_dict=True
        )
    # output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print("\nModel output:\n", outputs)


    print(f"Collected attention layers: {list(attn_cache.keys())}")
    print(f"model.config.output_attentions: {model.config.output_attentions}")
    for k, v in attn_cache.items():
        print(f"Layer {k}: got {len(v)} attention tensors, shape {v[0].shape}")
    # visualize one example attention map
    # save_dir = "attn_maps"
    # os.makedirs(save_dir, exist_ok=True)
    # if attn_cache:
    #     for layer, mats in attn_cache.items():
    #         # use last step attention of first head
    #         attn = mats[-1][0, 0].float()
    #         sns.heatmap(attn, cmap="viridis")
    #         plt.title(f"Layer {layer} attention (head 0, last step)")
    #         plt.savefig(os.path.join(save_dir, f"layer{layer}.png"), bbox_inches="tight")
    #         plt.close()
    #     print(f"Saved {len(attn_cache)} attention maps to {save_dir}/")



# ------------------ entry ------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/czj/llava15_test/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default="/home/czj/llava15_test/LLaVA/images/llava_logo.png")
    parser.add_argument("--query", type=str, default="What is this image?")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    eval_model_with_attn(args)
