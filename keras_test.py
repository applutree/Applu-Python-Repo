import tensorflow as tf
print("step 1")
mnist = tf.keras.datasets.mnist
print("step 2")
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print("step 3")
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
print("step 4")
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print("step 5")
model.fit(x_train, y_train, epochs=8)
print("step 6")
model.evaluate(x_test, y_test)