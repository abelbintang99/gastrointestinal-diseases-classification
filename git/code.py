import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

def build_model():
    base_model = InceptionV3(input_shape=IMAGE_SIZE + (3,), weights='imagenet', include_top=False)
    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

train_files, train_labels = get_file_paths_and_labels(train_path)
val_files, val_labels = get_file_paths_and_labels(val_path)
X_all = np.concatenate([train_files, val_files])
y_all = np.concatenate([train_labels, val_labels])



skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=23)
acc_per_fold, loss_per_fold = [], []

splits = list(skf.split(X_all, y_all))  

for fold, (train_idx, val_idx) in enumerate(splits, start=1):
    print(f"\n Training Fold {fold}/{K_FOLDS}")

    train_df = np.array([[X_all[i], y_all[i]] for i in train_idx])
    val_df = np.array([[X_all[i], y_all[i]] for i in val_idx])

    train_aug = ImageDataGenerator(
        rotation_range=10, 
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True)
    val_aug = ImageDataGenerator()

    train_gen = custom_generator(train_df, BATCH_SIZE, train_aug)
    val_gen = custom_generator(val_df, BATCH_SIZE, val_aug, shuffle=False)

    model = build_model()
    model_path = f"best_model_fold_{fold}.keras"
    checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, verbose=1)

    history = model.fit(
        train_gen,
        steps_per_epoch=len(train_df) // BATCH_SIZE,
        validation_data=val_gen,
        validation_steps=len(val_df) // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )

    model.load_weights(model_path)
    val_gen_eval = custom_generator(val_df, BATCH_SIZE, val_aug, shuffle=False)
    val_loss, val_acc = model.evaluate(val_gen_eval, steps=len(val_df) // BATCH_SIZE, verbose=0)
    acc_per_fold.append(val_acc)
    loss_per_fold.append(val_loss)

    print(f" Fold {fold} - Loss: {val_loss:.4f} - Accuracy: {val_acc:.4f}")

print("\n K-Fold Training Selesai")
print(f"Rata-rata Akurasi: {np.mean(acc_per_fold):.4f}")

test_files, test_labels = get_file_paths_and_labels(test_path)
test_df = np.array([[f, l] for f, l in zip(test_files, test_labels)])
test_aug = ImageDataGenerator()
test_gen = custom_generator(test_df, BATCH_SIZE, test_aug, shuffle=False)

steps_test = len(test_df) // BATCH_SIZE
best_fold = np.argmax(acc_per_fold) + 1
print(f"\n Menggunakan model terbaik dari Fold {best_fold}")

model = build_model()
model.load_weights(f"best_model_fold_{best_fold}.keras")
test_loss, test_acc = model.evaluate(test_gen, steps=steps_test)
print(f" Test Accuracy: {test_acc:.4f}")

model = build_model()
model.load_weights(f"best_model_fold_{best_fold}.keras")
for layer in model.layers:
    layer.trainable = True
model.compile(optimizer=Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

train_idx, val_idx = splits[best_fold - 1]
train_df = np.array([[X_all[i], y_all[i]] for i in train_idx])
val_df = np.array([[X_all[i], y_all[i]] for i in val_idx])

train_gen = custom_generator(train_df, BATCH_SIZE, train_aug)
val_gen = custom_generator(val_df, BATCH_SIZE, val_aug, shuffle=False)

checkpoint_fine = ModelCheckpoint(f"fine_tuned_model_fold_{best_fold}.keras", monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
early_stop = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, mode='max', verbose=1)

history_fine = model.fit(
    train_gen,
    steps_per_epoch=len(train_df) // BATCH_SIZE,
    validation_data=val_gen,
    validation_steps=len(val_df) // BATCH_SIZE,
    epochs=10,
    callbacks=[checkpoint_fine, early_stop],
    verbose=1
)

model.load_weights(f"fine_tuned_model_fold_{best_fold}.keras")
test_gen = custom_generator(test_df, BATCH_SIZE, test_aug, shuffle=False)
test_loss, test_acc = model.evaluate(test_gen, steps=steps_test)
print(f"\n Test Accuracy setelah Fine-Tuning: {test_acc:.4f}")


y_true, y_pred = [], []
for batch in test_gen:
    imgs, labels = batch[0], batch[1]
    preds = model.predict(imgs, verbose=0)
    y_true.extend(labels)
    y_pred.extend(np.argmax(preds, axis=1))
    if len(y_true) >= len(test_df): break

print("\n Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

print("\n Confusion Matrix:")
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
axs[0].plot(history.history['accuracy'], label='Train')
axs[0].plot(history.history['val_accuracy'], label='Validation')
axs[0].set_title('Model Accuracy (Before Fine-Tuning)')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Accuracy')
axs[0].legend(loc='lower right')
axs[0].grid(True)

# Loss plot
axs[1].plot(history.history['loss'], label='Train')
axs[1].plot(history.history['val_loss'], label='Validation')
axs[1].set_title('Model Loss (Before Fine-Tuning)')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Loss')
axs[1].legend(loc='upper right')
axs[1].grid(True)

plt.tight_layout()
plt.show()


fig, axs = plt.subplots(1, 2, figsize=(14, 5))


axs[0].plot(history_fine.history['accuracy'], label='Train (Fine-Tune)', marker='o')
axs[0].plot(history_fine.history['val_accuracy'], label='Validation (Fine-Tune)', marker='o')
axs[0].set_title('Accuracy After Fine-Tuning')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Accuracy')
axs[0].legend(loc='lower right')
axs[0].grid(True)


axs[1].plot(history_fine.history['loss'], label='Train (Fine-Tune)', marker='o')
axs[1].plot(history_fine.history['val_loss'], label='Validation (Fine-Tune)', marker='o')
axs[1].set_title('Loss After Fine-Tuning')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Loss')
axs[1].legend(loc='upper right')
axs[1].grid(True)

plt.tight_layout()
plt.show()

