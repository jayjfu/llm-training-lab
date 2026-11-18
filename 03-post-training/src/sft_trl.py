from peft import LoraConfig
from transformers import AutoProcessor, AutoTokenizer, GenerationConfig
from trl import SFTConfig, SFTTrainer

from common import load_model, make_run_dir, parse_args, save_final
from data import sft_dataset


def train(config, resume=False):
    run_dir = make_run_dir(config)
    train_args = config["train"]
    tokenizer = AutoTokenizer.from_pretrained(config["instruct_model"])
    dataset = sft_dataset(
        config["dataset"], tokenizer, train_args["max_length"],
        config.get("max_train_samples"), train_args["seed"], train_args.get("dataset_num_proc"),
    )

    lora = config.get("lora")
    peft_config = None
    if lora:
        peft_config = LoraConfig(
            r=lora["r"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora.get("dropout", 0.05),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )

    trainer = SFTTrainer(
        model=load_model(config["model"]),
        processing_class=tokenizer,
        args=SFTConfig(output_dir=str(run_dir), **train_args),
        train_dataset=dataset,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=resume or None)
    model = trainer.model.merge_and_unload() if lora else trainer.model
    model.generation_config = GenerationConfig.from_pretrained(config["instruct_model"])
    save_final(model, AutoProcessor.from_pretrained(config["instruct_model"]), run_dir)


if __name__ == "__main__":
    args = parse_args()
    for config in args.configs:
        train(config, args.resume)