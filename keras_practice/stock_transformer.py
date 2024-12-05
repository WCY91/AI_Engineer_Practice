import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Layer,Dense,LayerNormalization,Dropout
from sklearn.preprocessing import MinMaxScaler
import requests
import pandas as pd

np.random.seed(42)

data_length=2000
trend = np.linspace(100,200,data_length)

noise = np.random.normal(0,2, data_length)

synthetic_data = trend + noise
data =  pd.DataFrame(synthetic_data,columns=['Close'])
data.to_csv('stock_prices.csv',index=False)

data = pd.read_csv('stock_prices.csv')
data = data['Close'].values

scaler = MinMaxScaler(feature_range=(0,1))
data = scaler.fit_transform(data)

def create_dataset(data,time_step=1):
    X,Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i+time_step),0] 
        X.append(a)
        Y.append(data[i+time_step,0])
    return np.array(X) , np.array(Y)

time_step = 100
X,Y = create_dataset(data,time_step)
X = X.reshape(X.shape[0],X.shape[1],1)
print(X.shape)
print(Y.shape)

class MultiHeadSelfAttention(Layer):
    def __init__(self,embed_dim,num_heads=8):
        super(MultiHeadSelfAttention,self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def attention(self,query,key,value):
        score = tf.matmul(query,key,transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1],tf.float32)

        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score,axis=1)
        output = tf.matmul(weights,value)
        return output , weights
    
    def split_heads(self,x,batch_size):
        x = tf.reshape(x,(batch_size,-1,self.num_heads,self.projection_dim))
        return tf.transpose(x,perm = [0,2,1,3])
    
    def call(self,inputs):
        batch_size = tf.shape(inputs)[0]
        query  = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        query = self.split_heads(query,batch_size)
        key = self.split_heads(key,batch_size)
        value = self.split_heads(value,batch_size)

        attention,_ = self.attention(query,key,value)
        attention = tf.transpose(attention,perm = [0,2,1,3])

        concat_attention = tf.reshape(attention,(batch_size,-1,self.embed_dim))
        output = self.combine_heads(concat_attention)

        return output
    

class TransformerBlock(Layer):
    def __init__(self,embed_dim,num_heads,ff_dim,rate=0.1):
        super(TransformerBlock).__init__()
        self.att = MultiHeadSelfAttention(embed_dim,num_heads)
        self.fnn = Sequential([
            Dense(ff_dim,activaion = 'relu'),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon = 1e-6)
        self.layernorm2 = LayerNormalization(epsilon = 1e-6)
        self.dropout1 = Dropout(rate = rate)
        self.dropout2 = Dropout(rate)

    def call(self,inputs,training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output,training=training)
        out1 = self.layernorm1(attn_output + inputs) #殘差結構
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(ffn_output + out1)
    
class TransformerEncoder(Layer):
    def __init__(self,num_layers,embed_dim,num_heads,ff_dim,rate=0.1):
        super(TransformerBlock).__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.enc_layers = [TransformerBlock(embed_dim,num_heads,ff_dim,rate) for _ in range(num_layers)]
        self.dropout = Dropout(rate)

    def call(self,inputs,training=False):
        x = inputs
        for i in range(self.num_layers):
            x = self.enc_layers[i](x,training=training)

        return x

    
embed_dim = 128
num_heads = 8
ff_dim = 512
num_layers = 4

transformer_encoder = TransformerEncoder(num_layers,embed_dim,num_heads,ff_dim)
inputs = tf.random.uniform((1,100,embed_dim))
outputs = transformer_encoder(inputs,training=False)
print(outputs.shape)

input_shape = (X.shape[1],X.shape[2])
inputs = tf.keras.Input(shape = input_shape)
x = tf.keras.layers.Dense(embed_dim)(inputs) #先透過DNN embedding 到高維度
encoder_outputs = transformer_encoder(x)
flatten = tf.keras.layers.Flatten()(encoder_outputs)

outputs = tf.keras.layers.Dense(1)(flatten) #會這樣是因為最後股票是一個數值
model = tf.keras.Model(inputs,outputs)

model.compile(optimizer='adam',loss = 'mse')
model.summary()


model.fit(X,Y,epochs=20,batch_size=32)

predictions = model.predict(X)
predictions = scaler.inverse_transform(predictions)

plt.plot(data,label = 'True_data')
plt.plot(np.arange(time_step,time_step+len(predictions)),predictions,label='Predictions')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()