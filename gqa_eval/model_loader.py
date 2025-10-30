# model_loader.py
import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

def str2dtype(s: str):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[s]

def load_llava_model(args):
    """加载普通或带mask的LLaVA模型"""
    print("="*60)
    print(f"加载模型: {args.model_path}")
    print(f"设备: {args.device} | dtype={args.dtype} | multi_gpu={args.multi_gpu}")
    print(f"mask_visual_token={args.mask_visual_token}, strategy={args.mask_strategy}")
    print("="*60)

    # 根据mask选项选择模型版本
    use_masked_model = bool(args.mask_visual_token)
    print(f"模型类型: {'Llava_Masked' if use_masked_model else 'LLaVA'}")

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=get_model_name_from_path(args.model_path),
        device_map="cuda" if not args.multi_gpu else None,
        torch_dtype=str2dtype(args.dtype),
        load_4bit=args.load_in_4bit,
        load_8bit=args.load_in_8bit,
        use_masked_model=use_masked_model,
        mask_visual_token=args.mask_visual_token,
        mask_ratio=args.mask_ratio,
        mask_strategy=args.mask_strategy,
        mask_token_value=args.mask_token_value,
    )

    # 多GPU并行
    if args.multi_gpu and torch.cuda.device_count() > 1:
        print(f"启用多GPU并行，GPU数量: {torch.cuda.device_count()}")
        model = torch.nn.DataParallel(model)

    model.eval()
    return tokenizer, model, image_processor
