import torch
import torch.nn as nn
import Levenshtein

from ssbsc.swl import dataset as ds
from ssbsc.helpers import folders as fld
from ssbsc.swl import NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, LOG_STEPS, WARMUP_STEPS, MAX_SEQ_LEN
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments


class SecTrainer(Trainer):
    def __init__(self, *args, alpha=0.1, delta=1.0, **kwargs):
        super().__init__(*args, **kwargs)

        self.alpha = alpha
        self.delta = delta

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        loss = custom_loss(logits, labels, self.tokenizer, alpha=self.alpha, delta=self.delta)
        
        return (loss, outputs) if return_outputs else loss

# [to-do] check 
def custom_loss(logits, labels, tokenizer, alpha, delta):
    loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    ce_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    
    pred_ids = torch.argmax(logits, dim=-1)
    batch_edit_loss = 0.0

    for pred_seq, label_seq in zip(pred_ids, labels):
        label_seq = label_seq[label_seq != tokenizer.pad_token_id]
        pred_seq  = pred_seq[:len(label_seq)]

        pred_str  = tokenizer.decode(pred_seq, skip_special_tokens=True)
        label_str = tokenizer.decode(label_seq, skip_special_tokens=True)

        edit_dist = Levenshtein.distance(pred_str, label_str)

        batch_edit_loss += edit_dist / (len(label_str) + delta)
    
    batch_edit_loss /= logits.size(0)

    total_loss = ce_loss + alpha * batch_edit_loss

    return total_loss


def init_model():
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

    return tokenizer, model


def init__train_dataset(tokenizer):
    train_pairs, _ = ds.get_pairs()
    train_dataset = ds.SecDataset(train_pairs, tokenizer, MAX_SEQ_LEN)
    
    return train_dataset

def tune_model():
    temp_dir = fld.get_temp_dir()

    tokenizer, model = init_model()
    train_dataset = init__train_dataset(tokenizer)
    
    training_args = TrainingArguments(
        output_dir=temp_dir,

        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        logging_steps=LOG_STEPS,
        # to better leverage Adam
        warmup_steps=WARMUP_STEPS,  

        gradient_accumulation_steps=4,
        gradient_checkpointing=True,

        save_strategy="epoch",

        fp16=torch.cuda.is_available()
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    alpha = 0.6
    delta = 0.1

    trainer = SecTrainer(
        model=model,
        args=training_args,

        train_dataset=train_dataset,
        tokenizer=tokenizer,

        alpha=alpha,
        delta=delta
    )

    trainer.train()
    model.save_pretrained(temp_dir)
    tokenizer.save_pretrained(temp_dir)

    print("[Success] Fine-tuning completed")
