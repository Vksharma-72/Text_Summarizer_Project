from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from textSummarizer.entity import ModelTrainerConfig

import os
import torch
import mlflow

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config


    
    def train(self):
        os.environ["WANDB_DISABLED"] = "true"

        device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)

        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_ckpt,
            torch_dtype=torch.float16
            ).to(device)

    # Memory optimization
        model_pegasus.gradient_checkpointing_enable()

        seq2seq_data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model_pegasus
            )

        # Load dataset
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir,

            num_train_epochs=1,

            warmup_steps=500,

            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,

            gradient_accumulation_steps=16,

            weight_decay=0.01,

            logging_steps=10,

            evaluation_strategy="steps",
            eval_steps=500,

            save_steps=1000000,
            bf16=True,
            fp16=False,

            dataloader_pin_memory=False,

            report_to="none",

            run_name="pegasus_summarizer_v1"
            )

        trainer = Trainer(
            model=model_pegasus,
            args=trainer_args,
            tokenizer=tokenizer,
            data_collator=seq2seq_data_collator,
            train_dataset=dataset_samsum_pt["test"],
            eval_dataset=dataset_samsum_pt["validation"]
        )

        mlflow.set_tracking_uri("sqlite:///mlflow.db")

        torch.cuda.empty_cache()

        trainer.train()

    # Save model
        model_pegasus.save_pretrained(
        os.path.join(self.config.root_dir, "pegasus-samsum-model")
        )

        tokenizer.save_pretrained(
            os.path.join(self.config.root_dir, "tokenizer")
            )