import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.utils import get_file

path_to_file = get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(path_to_file,'rb').read().decode(encoding='utf-8')

vocab_size = 10000
seq_length = 10

vectorizer = TextVectorization(max_tokens=vocab_size,output_mode = 'int')
print(len(text))
text_ds = tf.data.Dataset.from_tensor_slices([text]).batch(1)
vectorizer.adapt(text_ds)
print(vectorizer([text]))
vectorized_text = vectorizer([text])[0]
print("vector text shape",vectorized_text.shape)
print("First 10 vectorized tokens",vectorized_text.numpy()[:10])

def create_sequence(text,seq_length):
  input_seqs=[]
  target_seqs=[]
  for i in range(len(text) - seq_length):
    input_seq = text[i:i+seq_length]
    target_seq = text[i+1:i+1+seq_length]
    input_seqs.append(input_seq)
    target_seqs.append(target_seq)
  return np.array(input_seqs), np.array(target_seqs)

X,Y = create_sequence(vectorized_text,seq_length)
print("Number of sequences generated:", len(X))
print("Sample input sequence:", X[0] if len(X) > 0 else "No sequences generated")

assert X.size > 0, "Input X is empty"
assert Y.size > 0, "Target Y is empty"

X=tf.convert_to_tensor(X)
Y = tf.convert_to_tensor(Y)

print(X.shape)
print(Y.shape)

from tensorflow.keras.layers import Embedding,MultiHeadAttention,Dense,LayerNormalization,Dropout,Layer
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
class TransformerBlock(Layer):
  def __init__(self,embed_dim,num_heads,ff_dim,rate=0.1):
    super(TransformerBlock,self).__init__()
    self.att = MultiHeadAttention(num_heads = num_heads, key_dim = embed_dim)
    self.fnn = Sequential([
        Dense(ff_dim,activation='relu'),
        Dense(embed_dim),
    ])
    self.layernorm1 = LayerNormalization(epsilon=1e-6)
    self.layernorm2 = LayerNormalization(epsilon = 1e-6)
    self.dropout1 = Dropout(rate)
    self.dropout2 = Dropout(rate)

  def call(self,inputs,training=False):
    attn_output = self.att(inputs,inputs)
    attn_output = self.dropout1(attn_output,training=training)
    out1 = self.layernorm1(inputs + attn_output) #殘差結構
    ffn_output = self.fnn(out1)
    ffn_output = self.dropout2(ffn_output,training=training)
    return self.layernorm2(out1+ffn_output)


class TransformerModel(Model):
  def __init__(self,vocab_size,embed_dim,num_heads,ff_dim,num_layers,seq_length):
    super(TransformerModel,self).__init__()
    self.embedding = Embedding(vocab_size, embed_dim)
    self.pos_encoding = self.positional_encoding(seq_length,embed_dim)
    self.transformer_blocks = [TransformerBlock(embed_dim,num_heads,ff_dim) for _ in range(num_layers)]
    self.dense = Dense(vocab_size) #會這樣設計是因為要用全部vocab_size去算機率 算說哪一個字比較適合出現在生成的上面

  def positional_encoding(self,seq_length,embed_dim):
    angle_rads = self.get_angles(np.arange(seq_length)[:,np.newaxis],np.arange(embed_dim)[np.newaxis,:],embed_dim) #會新增一個維度是為其矩陣相乘輸出要為seq*embed 所以第一個矩陣維seq*1 第二個1*embed
    angle_rads[:,0::2] = np.sin(angle_rads[:,0::2]) #奇數用sin
    angle_rads[:,1::2] = np.cos(angle_rads[:,1::2]) #偶數用cos
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding,dtype=tf.float32)

  def get_angles(self,pos,i,embed_dim):
    angle_rates = 1/np.power(10000,(2* (i//2))/np.float32(embed_dim)) #來自論文 其概念為將每個維度用不同的波長(sin cos)因為波長越長頻率越短 所以也可以看成可能比較遠的字或者比較少用到的字
    return pos * angle_rates

  def call(self, inputs, training=False):
    seq_len = tf.shape(inputs)[1]
    x = self.embedding(inputs)
    x += self.pos_encoding[:,:seq_len,:]
    for transformer_block in self.transformer_blocks:
      x = transformer_block(x, training=training)  # Pass training argument correctly
    output = self.dense(x)
    return output

embed_dim = 256
num_heads = 4
ff_dim = 512
num_layers = 4

model = TransformerModel(vocab_size,embed_dim,num_heads,ff_dim,num_layers,seq_length)

_= model(tf.random.uniform((1,seq_length),maxval=vocab_size,dtype = tf.int32))
model.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy')
model.summary()

import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)

# Train the transformer model on the full input and target sequences
history = model.fit(X, Y, epochs=20, batch_size=32, callbacks=[early_stopping])

# Plot training loss to monitor model performance over epochs
plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

def generate_text(model,start_string,num_generate=100,temperature=1.0):
  input_eval = vectorizer([start_string]).numpy()

  if input_eval.shape[1] < seq_length:
    padding=np.zeros((1,seq_length - input.eval.shape[1]))
    input_eval = np.concatenate((padding,input_eval),axis=1)
  elif input_eval.shape[1] > seq_length:
    input_eval = input_eval[:,-seq_length:]

  input_eval = tf.convert_to_tensor(input_eval)
  text_generated = []

  for i in range(num_generate):
    predictions = model(input_eval)
    predictions = predictions[0]

    predictions = predictions / temperature
    predictions = tf.expand_dims(predictions,0)

    predicted_id = tf.random.categorical(predictions,num_samples=1)[0,0].numpy()

    input_eval = np.append(input_eval.numpy(),[[predicted_id]],axis=1)
    input_eval = input_eval[:,-seq_length:]
    input_eval = tf.convert_to_tensor(input_eval)

    text_generated.append(vectorizer.get_vocabulary()[predicted_id])

  return start_string + ' ' + ' '.join(text_generated)

start_string = "To be , or not to be"
generated_text = generate_text(model,start_string,temperature=0.7)
print(generated_text)

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.utils import get_file
# Load the dataset
path_to_file = get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

# Preview the dataset
print(text[:1000])
# Preprocess the dataset
vocab_size = 10000
seq_length = 100

# Adapt TextVectorization to full text
vectorizer = TextVectorization(max_tokens=vocab_size, output_mode='int')
text_ds = tf.data.Dataset.from_tensor_slices([text]).batch(1)
vectorizer.adapt(text_ds)

# Vectorize the text
vectorized_text = vectorizer([text])[0]
print("Vectorized text shape:", vectorized_text.shape)
print("First 10 vectorized tokens:", vectorized_text.numpy()[:10])

def create_sequences(text, seq_length):
    input_seqs = []
    target_seqs = []
    for i in range(len(text) - seq_length):
        input_seq = text[i:i + seq_length]
        target_seq = text[i + 1:i + seq_length + 1]
        input_seqs.append(input_seq)
        target_seqs.append(target_seq)
    return np.array(input_seqs), np.array(target_seqs)

# Generate sequences
X, Y = create_sequences(vectorized_text.numpy(), seq_length)

# Check if sequences are correctly generated
print("Number of sequences generated:", len(X))
print("Sample input sequence:", X[0] if len(X) > 0 else "No sequences generated")

# Check if X and Y are not empty
assert X.size > 0, "Input data X is empty"
assert Y.size > 0, "Target data Y is empty"
X = tf.convert_to_tensor(X)
Y = tf.convert_to_tensor(Y)
print("Shape of X:", X.shape)
print("Shape of Y:", Y.shape)


from tensorflow.keras.layers import Embedding, MultiHeadAttention, Dense, LayerNormalization, Dropout
from tensorflow.keras.models import Model

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TransformerModel(Model):  # Model is now properly imported
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, seq_length):
        super(TransformerModel, self).__init__()
        self.embedding = Embedding(vocab_size, embed_dim)
        self.pos_encoding = self.positional_encoding(seq_length, embed_dim)
        self.transformer_blocks = [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        self.dense = Dense(vocab_size)

    def positional_encoding(self, seq_length, embed_dim):
        angle_rads = self.get_angles(np.arange(seq_length)[:, np.newaxis], np.arange(embed_dim)[np.newaxis, :], embed_dim)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, embed_dim):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embed_dim))
        return pos * angle_rates

    def call(self, inputs, training=False):
        seq_len = tf.shape(inputs)[1]
        x = self.embedding(inputs)
        x += self.pos_encoding[:, :seq_len, :]
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training)  # Pass training argument correctly
        output = self.dense(x)
        return output

# Hyperparameters
embed_dim = 256
num_heads = 4
ff_dim = 512
num_layers = 4

# Build the Transformer model
model = TransformerModel(vocab_size, embed_dim, num_heads, ff_dim, num_layers, seq_length)

# Provide input shape to build the model by passing a dummy input with maxval specified
_ = model(tf.random.uniform((1, seq_length), maxval=vocab_size, dtype=tf.int32))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Summary of the model
model.summary()

# Import necessary libraries for training visualization
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

# Early stopping callback to stop training if the loss doesn't improve
early_stopping = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)

# Train the transformer model on the full input and target sequences
history = model.fit(X, Y, epochs=20, batch_size=32, callbacks=[early_stopping])

# Plot training loss to monitor model performance over epochs
plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

def generate_text(model, start_string, num_generate=100, temperature=1.0):
    # Convert the start string to a vectorized format
    input_eval = vectorizer([start_string]).numpy()

    # Ensure the input length is the same as the model's expected input shape
    if input_eval.shape[1] < seq_length:
        # Pad the input if it's shorter than the expected sequence length
        padding = np.zeros((1, seq_length - input_eval.shape[1]))
        input_eval = np.concatenate((padding, input_eval), axis=1)
    elif input_eval.shape[1] > seq_length:
        # Truncate the input if it's longer than the expected sequence length
        input_eval = input_eval[:, -seq_length:]

    input_eval = tf.convert_to_tensor(input_eval)

    # Initialize an empty list to store generated text
    text_generated = []

    # Start generating text
    for i in range(num_generate):
        # Make predictions using the model
        predictions = model(input_eval)

        # Remove only the batch dimension, keep the logits as 2D (batch_size, vocab_size)
        predictions = predictions[0]  # This should be of shape [vocab_size]

        # Apply temperature to predictions
        predictions = predictions / temperature

        # Ensure predictions has two dimensions
        predictions = tf.expand_dims(predictions, 0)  # Adding a batch dimension to make it [1, vocab_size]

        # Use a categorical distribution to predict the next word
        predicted_id = tf.random.categorical(predictions, num_samples=1)[0, 0].numpy()

        # Update the input tensor to include the predicted word, maintaining the sequence length
        input_eval = np.append(input_eval.numpy(), [[predicted_id]], axis=1)  # Append predicted token
        input_eval = input_eval[:, -seq_length:]  # Keep only the last `seq_length` tokens
        input_eval = tf.convert_to_tensor(input_eval)  # Convert back to tensor

        # Append the predicted word to the generated text
        text_generated.append(vectorizer.get_vocabulary()[predicted_id])

    # Return the generated text starting from the initial seed
    return start_string + ' ' + ' '.join(text_generated)

# Generate text with temperature control
start_string = "To be, or not to be"
generated_text = generate_text(model, start_string, temperature=0.7)  # Lower temperature for more focused predictions
print(generated_text)


