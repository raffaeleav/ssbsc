# dataset parameters
NUM_TRAIN_SENTENCES = 20000
NUM_TEST_SENTENCES = 500
MAX_BYTES = 64
SEGMENTS = 8

# awgn channel parameters
SNR_DB_LIST = [-2.0, -1.5, -1.1, -0.6,-0.2, 0.2, 0.6, 1.1,1.5, 2.0]

# training parameters (batch size is set to 32 because of my 8GB VRAM)
NUM_EPOCHS = 8
BATCH_SIZE = 32
LEARNING_RATE = 3e-5
LOG_STEPS = 100
WARMUP_STEPS = 500
MAX_SEQ_LEN = 64
