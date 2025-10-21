import sys
from pathlib import Path

# 自动把 gqa_eval 目录加入 Python 路径
sys.path.append(str(Path(__file__).parent / "gqa_eval"))


from gqa_eval.config import get_args
from gqa_eval.seed_utils import set_seed
from gqa_eval.model_loader import load_llava_model
from gqa_eval.evaluator import eval_on_gqa
from llava.data.gqa_loader import GQALoader

def main():
    args = get_args()
    set_seed(args.seed)
    gqa_loader = gqa_loader = GQALoader(args.dataset_path, num_workers=args.num_workers, batch_size=args.batch_size)
    tokenizer, model, image_processor = load_llava_model(args)
    summary = eval_on_gqa(model, tokenizer, image_processor, gqa_loader, args)
    print(summary)

if __name__ == "__main__":
    main()
