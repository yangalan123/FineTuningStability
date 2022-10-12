import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from run_glue import task_to_keys


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download Model Checkpoints and Dataset From Huggingface")
    parser.add_argument("--model_name_or_path", type=str, help="HF model path, e.g., bert-base-uncased")
    parser.add_argument("--task_name", default=None, type=str, help="HF dataset path")
    parser.add_argument("--task_type", default="glue", type=str, help="glue or super_glue")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Where you want to cache the model and dataset")
    args = parser.parse_args()
    if args.task_type is not None and args.task_name is not None:
        assert args.task_type in task_to_keys, "you have to choose glue or super_glue"
        assert args.task_name in task_to_keys[args.task_type], f"Since we run {args.task_type} tasks, you should specify one (you specify: {args.task_name}) of the task from {list(task_to_keys.keys())}"
        print("Downloading Dataset....")
        raw_dataset = load_dataset(args.task_type, args.task_name, cache_dir=args.cache_dir)
    print("Downloading Tokenizer Files....")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    print("Downloading Model Files....")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    except:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    print(model.device)
    print(f"Done! All downloading complete. Now you should be able to run offline training (remember to modify the cache_dir in run.sh to {args.cache_dir}!)")

