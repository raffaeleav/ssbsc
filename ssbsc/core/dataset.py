import os
import json
import torch
import galois
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from datasets import load_dataset
from ssbsc.helpers import folders as fld
from sionna.phy.fec.linear import LinearEncoder, OSDecoder
from concurrent.futures import ProcessPoolExecutor, as_completed
from ssbsc.core import MAX_BYTES, SEGMENTS, SNR_DB_LIST


class SecDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, tokenizer, max_len):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        s1, s = self.pairs[idx]
        enc = self.tokenizer(s1, truncation=True, padding='max_length', max_length=self.max_len, return_tensors="pt")
        tgt = self.tokenizer(s, truncation=True, padding='max_length', max_length=self.max_len, return_tensors="pt")
        
        return {
            "input_ids": enc.input_ids.squeeze(0),
            "attention_mask": enc.attention_mask.squeeze(0),
            "labels": tgt.input_ids.squeeze(0)
        }


def awgn(bits, snr_db):
    snr_linear = 10.0 ** (snr_db/10.0)
    noise_std = np.sqrt(1.0 / (2.0 * snr_linear))

    bits = np.array(bits, dtype=np.float32)
    tx = 2.0 * bits - 1.0
    rx = tx + noise_std * np.random.randn(*tx.shape)

    llr = 2 * rx / (noise_std ** 2)

    return llr


def init_lbc_generator(k, n):
    if n != 32 or k != 16:
        raise ValueError("Only extended BCH(32,16) is supported")

    bch = galois.BCH(n - 1, k)

    # for coding an extended BCH Code (128,64) is needed
    G_31 = np.array(bch.G, dtype=int)
    parity_col = np.mod(np.sum(G_31, axis=1), 2).reshape(-1, 1)
    G = np.concatenate([G_31, parity_col], axis=1)

    return G


def init_lbc_encoder(G): 
    encoder = LinearEncoder(G)

    return encoder


def init_lbc_decoder(G): 
    decoder = OSDecoder(G)

    return decoder


def encode(encoder, bits):
    # enconder needs int32
    bits = np.array(bits, dtype=np.int32)
    cw = encoder(bits)

    return cw


def decode(decoder, llr):
    # decoder needs float32
    llr = np.array(llr, dtype=np.float32)

    with tf.device("/CPU:0"):
        db = decoder(llr)

    return db


def process_sentence(G, s, n, k):
    pairs = []

    # encoders and decoders are not thread-safe, so only the gen. matrix is shared
    encoder = init_lbc_encoder(G)
    decoder = init_lbc_decoder(G)

    s = s.ljust(MAX_BYTES, "\x00")

    bytes = s.encode("ascii", errors="replace")[:MAX_BYTES]
    bytes = np.frombuffer(bytes, dtype=np.uint8)

    bits = np.unpackbits(bytes)

    total_len = SEGMENTS * k

    if len(bits) < total_len:
        bits = np.concatenate([bits, np.zeros(total_len - len(bits), dtype=np.uint8)])

    seg_bits = np.split(bits, SEGMENTS)

    for snr in SNR_DB_LIST:
        rec_bits = []

        for seg in seg_bits:
            cw = encode(encoder, seg)
            cw = tf.cast(cw, tf.float32)

            llr = awgn(cw, snr)
            llr = tf.reshape(llr, (1, -1))

            db = decode(decoder, llr)
            db = db.numpy()
            db = db.reshape(-1)
            
            rec_bits.append(db[:k].astype(np.uint8))

        rec_bits = np.concatenate(rec_bits)
        rec_bytes = np.packbits(np.array(rec_bits, dtype=np.uint8))[:MAX_BYTES]

        try:
            s1 = rec_bytes.tobytes().decode("ascii", errors="replace")
        except Exception:
            s1 = ""

        pairs.append((s1, s))

    return pairs


def enc_dec(sentences):
    pairs = []

    k = 16
    n = 32

    G = init_lbc_generator(k, n)

    # since there are more segments i can only leverage 2 workers because of the increased number of tf tensors
    max_workers = 2

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_sentence, G, s, n, k) for s in sentences]

        for f in tqdm(as_completed(futures), total=len(futures), desc="encoding / decoding"):
            pairs.extend(f.result())

    return pairs



def get_sentences(test_pairs):
    sentences = []

    for pair in test_pairs:
        s = pair[1]
        sentences.append(s)

    return sentences


def get_pairs():
    datasets_dir = fld.get_datasets_dir()
    train_pairs_file = fld.get_file_path(datasets_dir, "swl_train_pairs.json")
    test_pairs_file = fld.get_file_path(datasets_dir, "ssbsc_test_pairs.json")
    swl_test_pairs_file = fld.get_file_path(datasets_dir, "swl_test_pairs.json")

    with open(train_pairs_file, "r") as f:
        train_pairs = json.load(f)

    if os.path.isfile(test_pairs_file):
        with open(test_pairs_file, "r") as f:
            test_pairs = json.load(f)
    else:
        with open(swl_test_pairs_file, "r") as f:
            swl_test_pairs = json.load(f)
            
        test_sentences = get_sentences(swl_test_pairs)
        test_pairs = enc_dec(test_sentences)

        with open(test_pairs_file, "w") as f:
            json.dump(test_pairs, f)

    return train_pairs, test_pairs
