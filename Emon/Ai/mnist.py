import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# 1. LOAD DATASET (MNIST)
# ==========================================

(x, y), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Training data:", x.shape)
print("Testing data:", x_test.shape)

# ==========================================
# 2. PREPROCESS THE DATA
# ==========================================

# Normalize pixel values from 0-255 to 0-1
x = x.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension for CNN: (28, 28) -> (28, 28, 1)
x = x[..., np.newaxis]
x_test = x_test[..., np.newaxis]

# ==========================================
# 3. SPLIT TRAINING DATA
# ==========================================

x_train, x_val, y_train, y_val = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train:", x_train.shape)
print("Validation:", x_val.shape)
print("Test:", x_test.shape)

# ==========================================
# 4. BUILD A CUSTOM NEURAL NETWORK
# ==========================================

def build_custom_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])
    return model

# ==========================================
# 5. BUILD A CNN MODEL
# ==========================================

def build_cnn_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax")
    ])
    return model

# ==========================================
# 6. SELECT MODEL
# ==========================================

model = build_cnn_model()
# To use the custom neural network instead:
# model = build_custom_model()

model.summary()

# ==========================================
# 7. COMPILE MODEL
# ==========================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================================
# 8. EARLY STOPPING
# ==========================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# ==========================================
# 9. TRAIN MODEL
# ==========================================

history = model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=64,
    validation_data=(x_val, y_val),
    callbacks=[early_stop]
)

# ==========================================
# 10. EVALUATE MODEL
# ==========================================

test_loss, test_accuracy = model.evaluate(
    x_test, y_test, verbose=0
)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# ==========================================
# 11. MAKE A PREDICTION
# ==========================================

predictions = model.predict(x_test[:5], verbose=0)
predicted_labels = np.argmax(predictions, axis=1)

print("Predicted:", predicted_labels)
print("Actual:   ", y_test[:5])

# ==========================================
# 12. PLOT TRAINING RESULTS
# ==========================================

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()
