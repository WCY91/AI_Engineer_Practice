import tensorflow as tf
from tensorflow.keras.dataset import mnist
from tensorflow.keras.layers import Flatten,Dense,BatchNormalization,LayerNormalization,Softmax
from tensorflow.keras.models import Model,Sequential

(x_train,y_train),(x_val,y_val) = mnist.load_data()
x_train,x_val = x_train.astype('float32')/255.0 , x_val.astype('float32')/255.0

teacher_model = Sequential([
    Flatten(input_shape = (28,28)),
    Dense(256,activation='relu'),
    Dense(10,activation='softmax')
])

teacher_model.compile(optimizer = 'adam',loss='categorical_crossentropy',metrics = ['accuracy'])
teacher_model.fit(x_train,y_train,epochs=5,validation_data=(x_val,y_val))

student_model = Sequential([
    Flatten(input_shape=(28,28)),
    Dense(128,activation='relu'),
    Dense(10,activation='softmax')
])

def distillation_loss(y_true,y_pred,teacher_pred,temperature): #y_pred : student_pred and the temperature 可以控制softmax平滑度 如果>1會比較平滑可能會更有創造性 <1會越準確
    y_true_loss = tf.keras.losses.categorical_crossentropy(y_true,y_pred)
    y_teacher_loss = tf.keras.losses.categorical_crossentropy(
        tf.nn.softmax(teacher_pred / temperature),
        tf.nn.softmax(y_pred/temperature)
    )
    return y_true_loss + temperature ** 2* y_teacher_loss

optimizer = tf.keras.optimizers.Adam()

@tf.function # 裝飾器 將 Python 函數轉換為 TensorFlow 的圖模式（Graph Mode）。
def train_step(x,y):
    teacher_pred = teacher_model(x,training=False)
    with tf.GradientTape() as tape:
        student_pred = student_model(x,training=True)
        loss = distillation_loss(y,student_pred,teacher_pred)
    gradients = tape.gradient(loss,student_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients,student_model.trainable_variables))
    return loss

train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_dataset = train_dataset.cache().batch(64).prefetch(tf.data.AUTOTUNE)
for epoch in range(5):
    for x_batch,y_batch in train_dataset:
        loss = train_step(x_batch,y_batch)  

#### 上面的範例是基於 loss function的蒸餾  下面為基於 Logits (前回饋出來的結果去蒸餾的)

teacher_model = Sequential([
    Input(shape=(28,28)),
    Flatten(),
    Dense(128,activation='relu'),
    Dense(10,activation='softmax')
])

teacher_model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
teacher_model.fit(x_train,y_train,epochs =5 , validation_data = (x_val,y_val))

student_model = Sequential([
    Input(shape=(28,28)),
    Flatten(),
    Dense(32,activation='relu'),
    Dense(10,activation='softmax')
])

student_model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

def distillation_loss(teacher_logits, student_logits,temperature=3): #其原理是透過softmax多分類的概念將logits映射到對各類別的置信度(可以看成機率) 所以其值的範圍為0到1 總合為1
    teacher_probs = tf.nn.softmax(teacher_logits/temperature)
    student_probs = tf.nn.softmax(student_logits/temperature)
    #教師跟學生輸出層的 Logits 維度應該一致，才能進行蒸餾計算。 (如範例均為10)
    return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(teacher_probs,student_probs)) #tf.reduce_mean 將批次中所有樣本的交叉熵損失取平均，得到整個批次的損失值。


def train_student(student,teacher,x_train,y_train,batch_size=32,epochs=2,temperature=3):
    num_batchs = len(x_train) // batch_size
    for epoch in range(epochs):
        for batch in range(num_batchs):
            x_batch = x_train[batch * batch_size:(batch+1) * batch_size]
            y_batch = y_train[batch * batch_size:(batch+1) * batch_size]

            teacher_logits = teacher.predict(x_batch)

            with tf.GradientTape() as tape:
                student_logits = student(x_batch)
                loss = distillation_loss(teacher_logits,student_logits,temperature)

            grads = tape.gradient(loss,student.trainable_variables)
            student.optimizer.apply_gradients(zip(grads,student.trainable_variables))
            print(f"Epoch {epoch + 1}, Batch {batch + 1}/{num_batches}, Loss: {loss.numpy()}")

(x_train,y_train),(x_test,y_test) = tf.keras.dataset.mnist.load_data()
x_train,x_test = x_train.astype('float32')/255.0 , x_test.astype('float32') / 255.0

train_student(student_model,teacher_model,x_train,y_train)