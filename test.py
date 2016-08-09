import numpy as np
import lstm

# Session variables
floatX = 'float32'

# Data I/O
data = open('yoursourcetexthere.txt', 'r').read()
chars = sorted(list(set(data)))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# Vectorize data
chars_as_vectors = np.zeros((data_size,vocab_size)).astype(dtype=floatX)
for x in range(0,len(chars_as_vectors)):
    char = data[x]
    char_id = char_to_ix[char]
    chars_as_vectors[x,char_id] = 1

# Ix-to-char
def unit_to_char(data):
    char = ""
    for x in range(0,len(data)):
        char_id = np.argmax(data[x])
        char += ix_to_char[char_id]
    return char

# Initialize RNN
self = lstm.RNN((3,5),(vocab_size,32,32,vocab_size),softmax_temperature=0.75)

# Train RNN for 50 epochs and print training results
# after each epoch
####################
for x in range(0,50):
    self.train(chars_as_vectors,128)
    print(unit_to_char(self.predict(chars_as_vectors[:self.network_width],100,False)))