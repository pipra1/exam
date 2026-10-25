import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# 1. Generate custom polynomial data
np.random.seed(42)

x = np.linspace(-5, 5, 1000).reshape(-1, 1)
y = 3*x**2 + 2*x + 5
y += np.random.normal(0, 2, y.shape)  # Add noise

# 2. Split dataset
x_train, x_temp, y_train, y_temp = train_test_split(
    x, y, test_size=0.3, random_state=42
)

x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp, test_size=0.5, random_state=42
)

# 3. Build regression model
model = models.Sequential([
    layers.Input(shape=(1,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(1)
])

# 4. Compile
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# 5. Early stopping
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

# 6. Train
model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=200,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# 7. Evaluate
test_loss, test_mae = model.evaluate(x_test, y_test)
print("Test MSE:", test_loss)
print("Test MAE:", test_mae)

# 8. Predict
print("Prediction:", model.predict(np.array([[2.0]])))
