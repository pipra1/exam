import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load training images
train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",
    image_size=(128, 128),
    batch_size=32,
    label_mode="int",
    seed=42,
    validation_split=0.2,
    subset="training"
)

# 2. Load validation images
val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",
    image_size=(128, 128),
    batch_size=32,
    label_mode="int",
    seed=42,
    validation_split=0.2,
    subset="validation"
)

# 3. Load independent test images
test_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=(128, 128),
    batch_size=32,
    label_mode="int",
    shuffle=False
)

num_classes = len(train_ds.class_names)
print("Classes:", train_ds.class_names)

# 4. Build CNN
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Rescaling(1./255),

    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="softmax")
])

# 5. Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 6. Early stopping
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# 7. Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[early_stop]
)

# 8. Evaluate
test_loss, test_accuracy = model.evaluate(test_ds)
print("Test Accuracy:", test_accuracy)
