"""
Name : Tanush Pavan V
Roll.no : CB.EN.U4EEE24147


Fashion Product Classification using a CNN on Fashion-MNIST
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import layers, models

# ----------------------------------------------------------------------
# 1. Load Fashion-MNIST
# ----------------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")

# ----------------------------------------------------------------------
# 2. Display sample images
# ----------------------------------------------------------------------
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.title(class_names[y_train[i]])
    plt.axis("off")
plt.suptitle("Sample Fashion-MNIST Images")
plt.tight_layout()
plt.savefig("sample_images.png")
plt.show()

# ----------------------------------------------------------------------
# 3. Normalize the images
# ----------------------------------------------------------------------
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# ----------------------------------------------------------------------
# 4. Add the channel dimension
# ----------------------------------------------------------------------
x_train = np.expand_dims(x_train, -1)   # (N, 28, 28, 1)
x_test = np.expand_dims(x_test, -1)

# ----------------------------------------------------------------------
# 5. Design a CNN (Model 1 - baseline)
# ----------------------------------------------------------------------
def build_model_1():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

model_1 = build_model_1()
model_1.summary()

# ----------------------------------------------------------------------
# 6. Train and validate the model
# ----------------------------------------------------------------------
history_1 = model_1.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
)

# ----------------------------------------------------------------------
# 7. Modify the CNN (Model 2) - deeper network + BatchNorm + Dropout
# ----------------------------------------------------------------------
def build_model_2():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

model_2 = build_model_2()
model_2.summary()

history_2 = model_2.fit(
    x_train, y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.1,
)

# Choose the better-performing model for the rest of the analysis
val_acc_1 = max(history_1.history["val_accuracy"])
val_acc_2 = max(history_2.history["val_accuracy"])
best_model, best_history, best_name = (
    (model_2, history_2, "Model 2") if val_acc_2 >= val_acc_1 else (model_1, history_1, "Model 1")
)
print(f"Best model: {best_name} (val_acc={max(val_acc_1, val_acc_2):.4f})")

# ----------------------------------------------------------------------
# 8. Plot accuracy and loss for both models
# ----------------------------------------------------------------------
def plot_history(history, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Train Acc")
    axes[0].plot(history.history["val_accuracy"], label="Val Acc")
    axes[0].set_title(f"{title} - Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title(f"{title} - Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}_curves.png")
    plt.show()

plot_history(history_1, "Model 1")
plot_history(history_2, "Model 2")

# ----------------------------------------------------------------------
# 9. Evaluate the test set (using the best model)
# ----------------------------------------------------------------------
test_loss, test_acc = best_model.evaluate(x_test, y_test, verbose=0)
print(f"\n[{best_name}] Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

y_pred_probs = best_model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# ----------------------------------------------------------------------
# 10. Generate a confusion matrix
# ----------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 7))
plt.imshow(cm, cmap="Blues")
plt.title(f"Confusion Matrix ({best_name})")
plt.colorbar()
tick_marks = np.arange(10)
plt.xticks(tick_marks, class_names, rotation=45, ha="right")
plt.yticks(tick_marks, class_names)
plt.xlabel("Predicted label")
plt.ylabel("True label")
for i in range(10):
    for j in range(10):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                  color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=7)
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# ----------------------------------------------------------------------
# 11. Identify the three most confused classes
# ----------------------------------------------------------------------
cm_off_diag = cm.copy()
np.fill_diagonal(cm_off_diag, 0)

# Find the top confusion pairs (true_class -> predicted_class)
pairs = []
for i in range(10):
    for j in range(10):
        if i != j and cm_off_diag[i, j] > 0:
            pairs.append((cm_off_diag[i, j], i, j))
pairs.sort(reverse=True)

print("\nTop confused (true -> predicted) pairs:")
for count, true_idx, pred_idx in pairs[:5]:
    print(f"  {class_names[true_idx]:12s} -> {class_names[pred_idx]:12s} : {count} times")

# "Three most confused classes" = the 3 true classes with the highest
# total number of misclassifications
misclass_per_class = cm_off_diag.sum(axis=1)
top3_confused_idx = np.argsort(misclass_per_class)[::-1][:3]

print("\nThree most confused (misclassified) classes:")
for idx in top3_confused_idx:
    print(f"  {class_names[idx]}: {misclass_per_class[idx]} misclassifications")

# ----------------------------------------------------------------------
# 12. Display incorrect predictions for those three classes
# ----------------------------------------------------------------------
for idx in top3_confused_idx:
    wrong_indices = np.where((y_test == idx) & (y_pred != idx))[0]
    show_n = min(5, len(wrong_indices))

    plt.figure(figsize=(12, 3))
    plt.suptitle(f"Incorrect predictions for true class: {class_names[idx]}")
    for k, sample_idx in enumerate(wrong_indices[:show_n]):
        plt.subplot(1, show_n, k + 1)
        plt.imshow(x_test[sample_idx].squeeze(), cmap="gray")
        plt.title(f"Pred: {class_names[y_pred[sample_idx]]}", fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"incorrect_{class_names[idx].replace('/', '_')}.png")
    plt.show()
