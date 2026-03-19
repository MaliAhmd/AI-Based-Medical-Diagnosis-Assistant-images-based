import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from preprocess import load_data

IMG_SIZE = 224

train_generator, val_generator = load_data()

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)

predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5
)

import os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_save_path = os.path.join(base_dir, "models", "pneumonia_model.h5")
model.save(model_save_path)

print("Model saved successfully to", model_save_path)