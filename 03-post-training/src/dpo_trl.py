from transformers import AutoProcessor, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from common import load_model, make_run_dir, parse_args, save_final
from data import load_pairs


def train(config, resume=False):
    run_dir = make_run_dir(config)
    trainer = DPOTrainer(
        model=load_model(config["model"]),
        args=DPOConfig(output_dir=str(run_dir), **config["train"]),
        train_dataset=load_pairs(config),
        processing_class=AutoTokenizer.from_pretrained(config["model"]),
    )
    trainer.train(resume_from_checkpoint=resume or None)
    save_final(trainer.model, AutoProcessor.from_pretrained(config["model"]), run_dir)


if __name__ == "__main__":
    args = parse_args()
    for config in args.configs:
        train(config, args.resume)
