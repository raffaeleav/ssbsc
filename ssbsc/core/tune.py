import torch
import Levenshtein
import torch.nn as nn

from ssbsc.core import dataset as ds
from ssbsc.helpers import folders as fld
from transformers import BartForConditionalGeneration, GPT2Tokenizer, Trainer, TrainingArguments
from ssbsc.core import NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, LOG_STEPS, WARMUP_STEPS, MAX_SEQ_LEN


class SecTrainer(Trainer):
    def __init__(self, *args, alpha=0.1, delta=1.0, **kwargs):
        super().__init__(*args, **kwargs)

        self.alpha = alpha
        self.delta = delta

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        loss = custom_loss(logits, labels, self.processing_class, alpha=self.alpha, delta=self.delta)
        
        return (loss, outputs) if return_outputs else loss


def custom_loss(logits, labels, tokenizer, alpha, delta):
    loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    ce_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    
    pred_ids = torch.argmax(logits, dim=-1)
    batch_edit_loss = 0.0

    for pred_seq, label_seq in zip(pred_ids, labels):
        label_seq = label_seq[label_seq != tokenizer.pad_token_id]
        pred_seq  = pred_seq[:len(label_seq)]

        pred_str  = tokenizer.decode(pred_seq, skip_special_tokens=True)

        edit_dist = Levenshtein.distance(pred_str, label_seq)

        batch_edit_loss += edit_dist / (len(label_seq) + delta)
    
    batch_edit_loss /= logits.size(0)

    total_loss = ce_loss + alpha * batch_edit_loss

    return total_loss


def init_model():
    tokenizer = GPT2Tokenizer.from_pretrained("alisawuffles/superbpe-tokenizer-128k")

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    assert tokenizer.pad_token_id is not None and tokenizer.pad_token_id >= 0

    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

    model.resize_token_embeddings(len(tokenizer))

    model.config.pad_token_id = tokenizer.pad_token_id

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    return tokenizer, model


def init__train_dataset(tokenizer):
    train_pairs, _ = ds.get_pairs()
    train_dataset = ds.SecDataset(train_pairs, tokenizer, MAX_SEQ_LEN)
    
    return train_dataset


def tune_model():
    ssbsc_temp_dir = fld.get_ssbsc_temp_dir()

    tokenizer, model = init_model()
    train_dataset = init__train_dataset(tokenizer)
    
    # due to my 8GB VRAM i have to use 1 element x training batch, so gradient_accumulation_steps=8 is leveraged
    training_args = TrainingArguments(
        output_dir=ssbsc_temp_dir,

        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=2,
        learning_rate=LEARNING_RATE,
        logging_steps=LOG_STEPS,
        # to better leverage Adam
        warmup_steps=WARMUP_STEPS,  

        gradient_accumulation_steps=6,
        gradient_checkpointing=True,

        save_strategy="epoch",

        fp16=torch.cuda.is_available(),
        bf16=False,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    alpha = 0.6
    delta = 0.1

    trainer = SecTrainer(
        model=model,
        args=training_args,

        train_dataset=train_dataset,
        processing_class=tokenizer,

        alpha=alpha,
        delta=delta
    )

    trainer.train()
    model.save_pretrained(ssbsc_temp_dir)
    tokenizer.save_pretrained(ssbsc_temp_dir)

    print("[Success] Fine-tuning completed")
