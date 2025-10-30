import argparse, re, os, torch
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
from io import BytesIO
from PIL import Image
import requests
from llava.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_PLACEHOLDER
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path


def disable_torch_init():
    import torch
    import math
    torch.set_grad_enabled(False)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

# ============== 基础工具 ==============

def load_image(image_file):
    if image_file.startswith("http"):
        resp = requests.get(image_file)
        return Image.open(BytesIO(resp.content)).convert("RGB")
    return Image.open(image_file).convert("RGB")

def build_prompt(query, model, conv_mode=None):
    model_name = model.config._name_or_path if hasattr(model.config, "_name_or_path") else ""
    if conv_mode is None:
        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        else:
            conv_mode = "llava_v0"

    qs = query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        qs = re.sub(IMAGE_PLACEHOLDER, image_token_se if model.config.mm_use_im_start_end else DEFAULT_IMAGE_TOKEN, qs)
    else:
        qs = (image_token_se if model.config.mm_use_im_start_end else DEFAULT_IMAGE_TOKEN) + "\n" + qs

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt, conv_mode

def encode_images_for_len(model, images):
    with torch.inference_mode():
        feats = model.encode_images(
            process_images(images, model.get_vision_tower().image_processor, model.config)
            .to(model.device, dtype=torch.float16)
        )
    # 兼容不同返回结构
    if isinstance(feats, list):
        feats = feats[0]
    if feats.dim() == 4:  # [B, N, H, D]
        num_tokens = feats.shape[1] * feats.shape[2]
    elif feats.dim() == 3:  # [B, N, D]
        num_tokens = feats.shape[1]
    elif feats.dim() == 2:  # [N, D]
        num_tokens = feats.shape[0]
    else:
        raise ValueError(f"Unexpected feats shape: {feats.shape}")
    return num_tokens


def compute_spans(prompt, tokenizer, img_feat_len):
    """
    计算 system / image / question 的 token 区间（在“插入图像特征之后”的序列上）。
    做法：先找出 <image> 占位符位置前后的文本 token 数，再把 image 特征长度插进去。
    """
    # 让 tokenizer_image_token 在文本里保留 IMAGE_TOKEN_INDEX 占位，便于找到它的位置
    toks = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors=None, padding_side="right")
    ids = toks[0] if isinstance(toks, list) else toks
    # 找到占位符下标
    try:
        image_pos = ids.index(IMAGE_TOKEN_INDEX)
    except ValueError:
        # 没有图片占位符，就当没有图像（极少见）
        image_pos = len(ids)

    sys_len = image_pos               # image 占位符之前的 token 数
    q_len  = len(ids) - (image_pos + 1)  # image 占位符之后的文本 token 数
    # 插入图像特征后，最终序列分段：
    # [ 0, sys_len-1 ] -> system/text-before-image
    # [ sys_len, sys_len+img_feat_len-1 ] -> image patch tokens（实际是特征）
    # [ sys_len+img_feat_len, sys_len+img_feat_len+q_len-1 ] -> question/text-after-image
    sys_span = (0, sys_len-1) if sys_len>0 else None
    img_span = (sys_len, sys_len+img_feat_len-1) if img_feat_len>0 else None
    q_start  = sys_len + img_feat_len
    q_span   = (q_start, q_start+q_len-1) if q_len>0 else None
    base_len = sys_len + img_feat_len + q_len
    spans = {
        "sys": sys_span,
        "image": img_span,
        "question": q_span,
        "base_len": base_len  # 生成开始的下标
    }
    return spans

def parse_layers_arg(layers_str, num_layers):
    if not layers_str:
        return list(range(num_layers))
    out = []
    for part in layers_str.split(","):
        part = part.strip()
        if "-" in part:
            a,b = part.split("-")
            out.extend(range(int(a), int(b)+1))
        else:
            out.append(int(part))
    out = sorted(set([i for i in out if 0 <= i < num_layers]))
    return out

# ============== Hook ==============

def register_selected_layer_hooks(model, selected_layers):
    """
    仅在 LLaMA 文本层的指定层注册 hook，抓取多头注意力。
    结果结构：attn_trace[step][layer] = [Tensor_of_shape(B,H,S,S)]
    """
    attn_trace = {}
    step_ref = {"step": -1}  # 可变引用，循环里更新

    def save_attn_hook(module, inp, out):
        if not (isinstance(out, tuple) and len(out)>1 and out[1] is not None):
            return
        layer_idx = module.layer_idx
        attn = out[1].detach().cpu()  # [B, num_heads, S, S]
        if step_ref["step"] >= 0:
            attn_trace.setdefault(step_ref["step"], {})[layer_idx] = attn

    num_hooks = 0
    for name, m in model.model.named_modules():
        mname = re.match(r"layers\.(\d+)\.self_attn$", name)
        if mname:
            lid = int(mname.group(1))
            if lid in selected_layers:
                m.layer_idx = lid
                m.register_forward_hook(save_attn_hook)
                num_hooks += 1

    return attn_trace, step_ref, num_hooks

# ============== 主逻辑：手写解码 + 收集注意力 + 分段 ==============

def run(args):
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )
    model.eval().to(model.device)
    model.config.output_attentions = True  # 让 attention 参与计算

    # 构建 prompt
    prompt, conv_mode = build_prompt(args.query, model, args.conv_mode)

    # 图像
    image = load_image(args.image_file)
    images = [image]
    image_sizes = [image.size]
    image_tensors = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)

    # 计算图像特征长度（用于分段）
    img_feat_len = encode_images_for_len(model, images)
    spans = compute_spans(prompt, tokenizer, img_feat_len)

    # 文本 token
    tok = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt", padding_side="right")
    input_ids = tok["input_ids"].to(model.device)
    attention_mask = tok.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    # 只在「指定层」注册 hook
    num_layers = model.config.num_hidden_layers
    selected_layers = parse_layers_arg(args.layers, num_layers)
    attn_trace, step_ref, num_hooks = register_selected_layer_hooks(model, selected_layers)
    print(f"✅ Registered hooks on layers: {selected_layers} (total {num_hooks})")
    print(f"Spans: {spans}  (base_len={spans['base_len']})")

    # 自回归解码
    max_new = args.max_new_tokens
    eos_token_id = getattr(tokenizer, "eos_token_id", 2)
    past_key_values = None

    generated = input_ids.clone()      # 文本 token 序列（不含图像 patch）
    current_input = input_ids          # 第 0 步喂完整 prompt
    base_len_text = input_ids.shape[1] # ✅ 文本解码起点（用于最终 decode）

    with torch.inference_mode():
        for t in range(max_new):
            step_ref["step"] = t  # 给 hook 标记时间步

            outputs = model(
                input_ids=current_input,
                images=image_tensors if t == 0 else None,   # 只在第 0 步喂图像
                image_sizes=image_sizes if t == 0 else None,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )

            # 限定到 tokenizer 词表长度，避免越界 token
            logits = outputs.logits[:, -1, :tokenizer.vocab_size]
            next_token = torch.argmax(logits, dim=-1)   # [B]

            # 追加到生成序列（文本 token 侧）
            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)

            # 终止条件：至少生成 1 个 token 后才允许停
            if eos_token_id is not None and t > 0 and (next_token == eos_token_id).all():
                break

            # 增量解码：下一步只喂“上一步生成”的 token，并带上 KV cache
            current_input = next_token.unsqueeze(-1)    # 形状 [B, 1]
            past_key_values = outputs.past_key_values

    print("\n=== Attn ")
    for step, layer_dict in attn_trace.items():
        print(f"\nStep {step}:")
        for layer, attn in layer_dict.items():
            print(f"Layer {layer}:")
            print(f"Shape: {attn.shape}")
            print(f"attn: {attn}")

    # 输出文本与索引范围
    gen_only = generated[:, base_len_text:]    # ✅ 用文本长度作为分界

    final_text = tokenizer.batch_decode(gen_only, skip_special_tokens=True)[0].strip()

    base_len = spans["base_len"]
    gen_positions = (spans["base_len"], spans["base_len"] + gen_only.shape[1])
    print("\n=== Decode Done ===")
    print(f"generated.shape: {generated.shape}")
    print("Prediction:", final_text)
    print("System span:", spans["sys"], "Image span:", spans["image"], "Question span:", spans["question"])
    print("Gen positions:", gen_positions)
    print(f"Collected steps: {len(attn_trace)} (each has layers {sorted(attn_trace.get(0, {}).keys())})")

    # 你可以在此把注意力落盘
    # torch.save({"attn_trace": attn_trace, "spans": spans, "gen_positions": gen_positions}, "attn_trace.pt")


    return final_text, attn_trace, spans, gen_positions, selected_layers

# ============== CLI ==============
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/czj/llava15_test/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, default="/home/czj/llava15_test/LLaVA/images/llava_logo.png")
    parser.add_argument("--query", type=str, default="What is this image?")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--layers", type=str, default="0,15,31")  # 例如 "0,6,15,31" 或 "0-7"
    args = parser.parse_args()
    run(args)
