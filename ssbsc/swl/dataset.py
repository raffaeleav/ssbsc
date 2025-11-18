import os
import json
import torch
import galois
import numpy as np

from tqdm import tqdm
from datasets import load_dataset
from ssbsc.helpers import folders as fld
from sionna.phy.fec.linear import LinearEncoder, OSDecoder
from concurrent.futures import ProcessPoolExecutor, as_completed
from ssbsc.swl import NUM_TRAIN_SENTENCES, NUM_TEST_SENTENCES, MAX_BYTES, SEGMENTS, SNR_DB_LIST


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


def sentence_to_bytes(s, length):
    # substitute characters that are not ascii with ?
    b = s.encode("ascii", errors="replace")[:length]

    if len(b) < length:
        b = b + b'\x00' * (length - len(b))

    return np.frombuffer(b, dtype=np.uint8)


def awgn(bits, snr_db):
    # snr is converted from db in linear scale
    snr_linear = 10.0 ** (snr_db/10.0)

    # noise standard deviation
    noise_std = np.sqrt(1.0 / (2.0 * snr_linear))

    # a cast is needed for float arithmetic
    bits = np.array(bits, dtype=np.float32)

    # bspk modulation
    tx = 1.0 - 2.0 * bits

    # noise is added to the modulation
    rx = tx + noise_std * np.random.randn(*tx.shape)

    # log-likelihood ratios (confidence on which kind of bit is received)
    llr = 2 * rx / (noise_std ** 2)

    return llr


# encoders and decoders need a generator matrix or a parity-check matrix
def init_lbc_generator(k, n):
    if n != 128 or k != 64:
        raise ValueError("Only extended BCH(128,64) is supported")

    bch = galois.BCH(n - 1, k)

    # for coding an extended BCH Code (128,64) is needed
    G_127 = np.array(bch.G, dtype=int)
    parity_col = np.mod(np.sum(G_127, axis=1), 2).reshape(-1, 1)
    G = np.concatenate([G_127, parity_col], axis=1)

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
    db = decoder(llr)

    return db


def process_sentence(G, s):
    pairs = []

    # encoders and decoders are not thread-safe, so only the gen. matrix is shared
    encoder = init_lbc_encoder(G)
    decoder = init_lbc_decoder(G)

    bytes_ = sentence_to_bytes(s, MAX_BYTES)
    bits = np.unpackbits(bytes_)
    seg_bits = np.array_split(bits, SEGMENTS)

    for snr in SNR_DB_LIST:
        rec_bits = []

        # encode, add noise through awgn channel, decode
        for seg in seg_bits:
            cw  = encode(encoder, seg)
            llr = awgn(cw, snr)
            db  = decode(decoder, llr)
            rec_bits.append(db)

        rec_bits = np.concatenate(rec_bits)
        rec_bytes = np.packbits(np.array(rec_bits, dtype=np.uint8))[:MAX_BYTES]

        try:
            # estimated sentece
            s1 = rec_bytes.tobytes().decode("ascii", errors="replace")

            # checks if each character is s1 is ascii printable
            s1 = "".join(ch if 32 <= ord(ch) <= 126 else " " for ch in s1)
        except Exception:
            s1 = ""

        pairs.append((s1, s))

    return pairs


def enc_dec(sentences):
    pairs = []

    k = 64
    n = 128

    G = init_lbc_generator(k, n)

    max_workers = 8

    # parallel senteces processing
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_sentence, G, s) for s in sentences]

        for f in tqdm(as_completed(futures), total=len(futures), desc="encoding / decoding"):
            pairs.extend(f.result())

    return pairs


def get_pairs():
    datasets_dir = fld.get_datasets_dir()
    train_pairs_file = fld.get_file_path(datasets_dir, "train_pairs.json")
    test_pairs_file = fld.get_file_path(datasets_dir, "test_pairs.json")

    if os.path.isfile(train_pairs_file) and os.path.isfile(test_pairs_file):
        with open(train_pairs_file, "r") as f:
            train_pairs = json.load(f)

        with open(test_pairs_file, "r") as f:
            test_pairs = json.load(f)
    else:
        snli_corpus = load_dataset("snli")

        num_train_sentences = NUM_TRAIN_SENTENCES / 2
        num_test_sentences = NUM_TEST_SENTENCES / 2

        train_sentences = snli_corpus["train"]["premise"][:num_train_sentences] + snli_corpus["train"]["hypothesis"][:num_train_sentences]
        test_sentences = snli_corpus["validation"]["premise"][:num_test_sentences] + snli_corpus["validation"]["hypothesis"][:num_test_sentences]

        train_pairs = enc_dec(train_sentences)
        test_pairs = enc_dec(test_sentences)

        with open(train_pairs_file, "w") as f:
            json.dump(train_pairs, f)

        with open(test_pairs_file, "w") as f:
            json.dump(test_pairs, f)

    return train_pairs, test_pairs
