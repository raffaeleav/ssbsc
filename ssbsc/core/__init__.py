# dataset parameters
MAX_BYTES = 64
SEGMENTS = 32

# awgn channel parameters
SNR_DB_LIST = [-2.0, -1.2, -0.6, -0.2, 0.0, 0.2, 0.4, 0.8, 1.5, 2.0]

# training parameters
NUM_EPOCHS = 8
BATCH_SIZE = 2
LEARNING_RATE = 3e-5
LOG_STEPS = 100
WARMUP_STEPS = 500
MAX_SEQ_LEN = 64
