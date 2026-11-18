from transformers import AutoProcessor, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from common import load_model, make_run_dir, parse_args, save_final
from data import math_dataset
from rewards import REWARD_FUNCS


def train(config, resume=False):
    run_dir = make_run_dir(config)
    trainer = GRPOTrainer(
        model=load_model(config["model"]),
        reward_funcs=REWARD_FUNCS,
        args=GRPOConfig(output_dir=str(run_dir), **config["train"]),
        train_dataset=math_dataset(config["dataset"], config["levels"]),
        processing_class=AutoTokenizer.from_pretrained(config["model"]),
    )
    trainer.train(resume_from_checkpoint=resume or None)
    save_final(trainer.model, AutoProcessor.from_pretrained(config["model"]), run_dir)


if __name__ == "__main__":
    args = parse_args()
    for config in args.configs:
        train(config, args.resume)