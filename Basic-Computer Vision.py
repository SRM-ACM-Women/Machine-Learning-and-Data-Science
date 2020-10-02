import tensorflow as tf
print(tf.__version__)

mnist = tf.keras.datasets.mnist  # fashion mnist dataset

(training_images, training_labels) ,  (test_images, test_labels) = mnist.load_data()

training_images = training_images/255.0  # normalising data
test_images = test_images/255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),                      # Creating a neural net of 128 hidden layer and 10 output layer
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),     
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy')

model.fit(training_images, training_labels, epochs=30)    # training dataset

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

print(classifications[34])
print(test_labels[34])
