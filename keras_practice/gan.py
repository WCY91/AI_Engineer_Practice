from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Layer,Flatten,Dense,BatchNormalization,LeakyReLU,Reshape,Input
from tensorflow.keras.dataset import mnist
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

(x_train,_),(_,_) = mnist.load_data()
x_train = x_train.astype('float32')/127.5 - 1 #rescale [-1,1]
x_train = np.expand_dims(x_train,axis=1)

def build_generator():
    model = Sequential()
    model.add(Dense(256,input_dim=100))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(28*28*1,activation='tanh')) #會用這個是因為其收斂快 且比較不會像relu有死亡的問題
    model.add(Reshape(28*28*1)) #將dens 1d 轉3d

    return model

generator = build_generator()

def build_discriminator():
    model = Sequential()
    model.add(Flatten(input_shape = (28,28,1)))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1,activation='sigmoid'))

    return model

discriminator = build_discriminator()
discriminator.compile(optmizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

def build_gan(generator,discriminator):
    discriminator.trainable = False
    gan_input = Input(shape=(100,))
    generated_image = generator(gan_input)
    gan_output = discriminator(generated_image)
    gan = Model(gan_input,gan_output)
    gan.compile(optimizer='adam',loss = 'binary_crossentropy')
    return gan

gan = build_gan(generator,discriminator)
batch_size = 64
epochs=50
sample_interval = 10

real = np.ones((batch_size,1))
fake = np.zeros((batch_size,1))

for epoch in range(epochs):
    idx = np.random.randint(0,x_train.shape[0],batch_size)
    real_images = x_train[idx]
    noise = np.random.normal(0,1,(batch_size,100))
    generated_image = generator.predict(noise)
    d_loss_real = discriminator.train_on_batch(real_images,real)
    d_loss_fake = discriminator.train_on_batch(generated_image,fake)

    d_loss = 0.5*np.add(d_loss_real,d_loss_fake)

    noise = np.random.normal(0,1,(batch_size,100))
    g_loss = gan.train_on_batch(noise,real)

    if epoch % sample_interval ==0:
        print(f"{epoch} [D loss : {d_loss[0]}] [D acc : {100*d_loss[1]}%] [G Loss : {g_loss}]")

def sample_image(generator,epoch,num_images=25):
    noise = np.random.normal(0,1,(num_images,100))
    generated_image = generator.predict(noise)
    generated_image = 0.5*generated_image + 0.5
    fig,axs = plt.subplots(5,5,figsize=(10,10))
    count = 0

    for i in range(5):
        for j in range(5):
            axs[i,j].imshow(generated_image[count,:,:,0],cmap='gray')
            axs[i,j].axis('off')
            count +=1
    plt.show()


sample_image(generator,epochs)


noise = np.random.normal(0,1,(batch_size,100))
generated_image = generator.predict(noise)

real_images = x_train[np.random.randint(0,x_train.shape[0],batch_size)]
d_loss_real = discriminator.evaluate(real_images,np.ones((batch_size,1)),verbose=0)

d_loss_fake  = discriminator.evaluate(generated_image,np.zeros((batch_size,1)),verbose=0)


print(f"Discriminator Accuracy on Real Images: {d_loss_real[1] * 100:.2f}%")
print(f"Discriminator Accuracy on Fake Images: {d_loss_fake[1] * 100:.2f}%")