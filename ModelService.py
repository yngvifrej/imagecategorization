import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import pathlib
class ModelService(object):
    """description of class"""
    def Predict(url, class_names, model_file):
        # define dataset parameters
        batch_size = 32
        img_height = 180
        img_width = 180

        print("Load model")
        model = load_model(model_file)
        print("Loaded!! ")
        print(model)
        sunflower_url = url
        import random
        n = random.randint(0,2200)
        sunflower_path = tf.keras.utils.get_file('image'+str(n), origin=sunflower_url)

        img = keras.preprocessing.image.load_img(
            sunflower_path, target_size=(img_height, img_width)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )

        print("END!")


    def GenerateModel(dataset_url, dataset_name, model_name):
        print("dataset_name")
        print(dataset_name)
        data_dir = tf.keras.utils.get_file(dataset_name, origin=dataset_url, untar=True,archive_format='zip')
        data_dir = pathlib.Path(data_dir)
        print("data_dir")
        print(data_dir)
        image_count = len(list(data_dir.glob('*/*.jpg')))
        print(image_count)


        # define dataset parameters
        batch_size = 32
        img_height = 180
        img_width = 180

        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
          data_dir,
          validation_split=0.2,
          subset="training",
          seed=123,
          image_size=(img_height, img_width),
          batch_size=batch_size)

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
          data_dir,
          validation_split=0.2,
          subset="validation",
          seed=123,
          image_size=(img_height, img_width),
          batch_size=batch_size)

        class_names = train_ds.class_names
        print("Class names")
        print(class_names)

        #visualize data
        #import matplotlib.pyplot as plt

        #plt.figure(figsize=(10, 10))
        #for images, labels in train_ds.take(1):
        #  for i in range(9):
        #    ax = plt.subplot(3, 3, i + 1)
        #    plt.imshow(images[i].numpy().astype("uint8"))
        #    plt.title(class_names[labels[i]])
        #    plt.axis("off")

        #for image_batch, labels_batch in train_ds:
        #  print(image_batch.shape)
        #  print(labels_batch.shape)
        #  break

        print("Configure the dataset for performance")
        AUTOTUNE = tf.data.experimental.AUTOTUNE

        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

        normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        image_batch, labels_batch = next(iter(normalized_ds))
        first_image = image_batch[0]
        # Notice the pixels values are now in `[0,1]`.
        print(np.min(first_image), np.max(first_image)) 


        print("Create the model")
        num_classes = 5

        model = Sequential([
          layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
          layers.Conv2D(16, 3, padding='same', activation='relu'),
          layers.MaxPooling2D(),
          layers.Conv2D(32, 3, padding='same', activation='relu'),
          layers.MaxPooling2D(),
          layers.Conv2D(64, 3, padding='same', activation='relu'),
          layers.MaxPooling2D(),
          layers.Flatten(),
          layers.Dense(128, activation='relu'),
          layers.Dense(num_classes)
        ])

        print("Compile the model")

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        print("Model summary")
        model.summary()

        print("Train the model")
        epochs=10
        history = model.fit(
          train_ds,
          validation_data=val_ds,
          epochs=epochs
        )

        #print("Visualize training results")
        #acc = history.history['accuracy']
        #val_acc = history.history['val_accuracy']

        #loss=history.history['loss']
        #val_loss=history.history['val_loss']

        #epochs_range = range(epochs)

        #plt.figure(figsize=(8, 8))
        #plt.subplot(1, 2, 1)
        #plt.plot(epochs_range, acc, label='Training Accuracy')
        #plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        #plt.legend(loc='lower right')
        #plt.title('Training and Validation Accuracy')

        #plt.subplot(1, 2, 2)
        #plt.plot(epochs_range, loss, label='Training Loss')
        #plt.plot(epochs_range, val_loss, label='Validation Loss')
        #plt.legend(loc='upper right')
        #plt.title('Training and Validation Loss')
        #plt.show()
        print("Saving model")
        model.save(model_name)

        print("Class names")
        print(class_names)
        print("End")

