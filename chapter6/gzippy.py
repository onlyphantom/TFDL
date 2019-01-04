import gzip
import numpy as np

train_data_filename = 'data_input/train-images-idx3-ubyte.gz'
IMAGE_SIZE = 28
NUM_CHANNELS = 1

def extract_data_w_offset(filename, num_images, offset):
    print("Extracting using offset", filename)
    with gzip.open(filename) as bytestream:
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS + offset)
        off16 = np.frombuffer(buf, offset=offset, dtype=np.uint8).astype(np.float32)
        return off16

def extract_data(filename, num_images):
    print("Extracting", filename)
    with gzip.open(filename) as bytestream:
        # skip header
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
        next16 = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        return next16


off16 = extract_data_w_offset(train_data_filename, 1, offset=16)
next16 = extract_data(train_data_filename, 1)
print(np.array_equal(off16, next16))
# equivalently: (off16==next16).all()