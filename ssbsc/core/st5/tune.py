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
        label_str = tokenizer.decode(label_seq, skip_special_tokens=True)

        edit_dist = Levenshtein.distance(pred_str, label_str)

        batch_edit_loss += edit_dist / (len(label_str) + delta)
    
    batch_edit_loss /= logits.size(0)

    total_loss = ce_loss + alpha * batch_edit_loss

    return total_loss


def init_model():
    tokenizer = None
    model = None

    return tokenizer, model