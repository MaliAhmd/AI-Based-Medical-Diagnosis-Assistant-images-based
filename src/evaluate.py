import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

IMG_SIZE = 224
BATCH_SIZE = 16

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    "../dataset/test",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

model = tf.keras.models.load_model("../models/pneumonia_model.h5")

predictions = model.predict(test_generator)
predictions = (predictions > 0.5)

print(classification_report(test_generator.classes, predictions))