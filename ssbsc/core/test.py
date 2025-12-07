import json
import torch
import numpy as np

from tqdm import tqdm
from datetime import datetime
from ssbsc.core import dataset as ds
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader
from ssbsc.helpers import folders as fld
from nltk.translate.bleu_score import corpus_bleu
from ssbsc.swl import MAX_SEQ_LEN, BATCH_SIZE, SEGMENTS
from transformers import BartForConditionalGeneration, GPT2Tokenizer


def init_model():
    ssbsc_temp_dir = fld.get_ssbsc_temp_dir()

    tokenizer = GPT2Tokenizer.from_pretrained(ssbsc_temp_dir)
    model = BartForConditionalGeneration.from_pretrained(ssbsc_temp_dir)

    return tokenizer, model


def init_test_dataset(tokenizer):
    _, test_pairs = ds.get_pairs()
    test_dataset  = ds.SecDataset(test_pairs, tokenizer, MAX_SEQ_LEN)

    return test_dataset


def decode(model, tokenizer, test_dataset, batch_size=32):
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    sentences = []
    pred_sentences = []

    for batch in tqdm(loader, desc="Decoding batches"):
        # noisy encoded sentences
        input_ids = batch["input_ids"]         
        attention_mask = batch["attention_mask"]
        # original sentences
        label_ids = batch["labels"]            

        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            label_ids = label_ids.cuda()
            model = model.cuda()

        with torch.no_grad():
            pred_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=4,
                max_length=MAX_SEQ_LEN,
                early_stopping=True
            )

        for l_ids, p_ids in zip(label_ids, pred_ids):
            # original sentence
            s = tokenizer.decode(l_ids, skip_special_tokens=True)

            # predicted sentence
            s2 = tokenizer.decode(p_ids, skip_special_tokens=True)

            sentences.append(s)
            pred_sentences.append(s2)

    return sentences, pred_sentences


def bler(sentences, pred_sentences, segments):
    blocks = 0
    errors = 0

    for s, s2 in zip(sentences, pred_sentences):
        s_list = np.array(list(s))
        s2_list = np.array(list(s2))

        s_blocks = np.array_split(s_list, segments)
        s2_blocks = np.array_split(s2_list, segments)

        blocks += segments

        for s_block, s2_block in zip(s_blocks, s2_blocks):
            if not np.array_equal(s_block, s2_block):
                errors+= 1

    return errors / blocks


def bleu(sentences, pred_sentences):
    s_tokens = [[s.split()] for s in sentences]
    s2_tokens = [s2.split() for s2 in pred_sentences]

    score = corpus_bleu(s_tokens, s2_tokens)

    return score


def rouge_l(sentences, pred_sentences):
    f1_scores = []
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    for s, s2 in zip(sentences, pred_sentences):
        score = scorer.score(s, s2)
        f1_scores.append(score['rougeL'].fmeasure)

    score = sum(f1_scores) / len(f1_scores)

    return score


def test_model():
    results_dir = fld.get_results_dir()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    results_file = fld.get_file_path(results_dir, f"ssbsc_results_{timestamp}.json")
    tokenizer, model = init_model()

    test_dataset = init_test_dataset(tokenizer)
    
    sentences, pred_sentences = decode(model, tokenizer, test_dataset, BATCH_SIZE)

    b = bler(sentences, pred_sentences, SEGMENTS)
    l = bleu(sentences, pred_sentences)
    r = rouge_l(sentences, pred_sentences)

    results = {
        "approach": "ssbsc",
        "bler": b,
        "bleu": l,
        "rouge_l": r
    }

    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"[Success] Results - BLER: {b}, BLEU: {l}, ROUGE-L: {r}")