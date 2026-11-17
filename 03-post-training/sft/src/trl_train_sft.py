import argparse
from pathlib import Path

from transformers import AutoProcessor, AutoModelForImageTextToText
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from peft import LoraConfig

ROOT = Path(__file__).resolve().parents[3]

def train_variant(name, lr, batch, accum, packing, peft, base_model, dataset, processor, output_dir, epochs, lora_r=None, lora_alpha=None):
    out_dir = output_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = AutoModelForImageTextToText.from_pretrained(base_model)

    peft_config = None
    if peft:
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )

    training_args = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=accum,
        bf16=True,
        gradient_checkpointing=True,
        packing=packing,
        assistant_only_loss=True,
        max_length=4096,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(str(Path(output_dir) / "final"))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, default="Qwen/Qwen3.5-0.8B-Base")
    p.add_argument("--instruct_model", type=str, default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--dataset", type=str, default="allenai/tulu-3-sft-mixture")
    p.add_argument("--output_dir", type=str, default=str(ROOT / "03-post-training/sft/ckpts"))
    p.add_argument("--epochs", type=int, default=1)
    args = p.parse_args()

    dataset = load_dataset(args.dataset, split="train")

    processor = AutoProcessor.from_pretrained(args.base_model)
    instruct_processor = AutoProcessor.from_pretrained(args.instruct_model)
    processor.chat_template = instruct_processor.chat_template

    # Full fine-tune
    train_variant("sft_lr2e5_bs32_acc1_pack", 2e-5, 32, 1, True, False, args.base_model, dataset, processor,
                  args.output_root, args.epochs)
    train_variant("sft_lr1e5_bs32_acc1_pack", 1e-5, 32, 1, True, False, args.base_model, dataset, processor,
                  args.output_root, args.epochs)
    train_variant("sft_lr5e5_bs32_acc1_pack", 5e-5, 32, 1, True, False, args.base_model, dataset, processor,
                  args.output_root, args.epochs)
    train_variant("sft_lr2e5_bs32_acc2_pack", 2e-5, 32, 2, True, False, args.base_model, dataset, processor,
                  args.output_root, args.epochs)
    train_variant("sft_lr5e5_bs32_acc1_nopack", 5e-5, 32, 1, False, False, args.base_model, dataset, processor,
                  args.output_root, args.epochs)

    # LoRA
    train_variant("sft_lora_r16_lr2e4_pack", 2e-4, 32, 1, True, True, args.base_model, dataset, processor,
                  args.output_root, args.epochs, 16, 32)
    train_variant("sft_lora_r32_lr2e4_pack", 2e-4, 32, 1, True, True, args.base_model, dataset, processor,
                  args.output_root, args.epochs, 32, 64)
    train_variant("sft_lora_r32_lr1e4_pack", 1e-4, 32, 1, True, True, args.base_model, dataset, processor,
                  args.output_root, args.epochs, 32, 64)
    train_variant("sft_lora_r8_lr2e4_nopack", 2e-4, 32, 1, False, True, args.base_model, dataset, processor,
                  args.output_root, args.epochs, 8, 16)

if __name__ == "__main__":
    main()