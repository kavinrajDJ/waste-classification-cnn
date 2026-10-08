import os
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
EPOCHS = 10

def build_cnn(num_classes=2):
    model = models.Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.Rescaling(1./255),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    train_dir = os.path.join('dataset', 'train')
    test_dir = os.path.join('dataset', 'test')

    if not os.path.exists(train_dir) or len(os.listdir(train_dir)) == 0:
        print("[!] Train directory is empty. Add class folders with images to dataset/train.")
        return

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names
    print(f"Classes identified: {class_names}")

    model = build_cnn(num_classes=len(class_names))
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    os.makedirs('results', exist_ok=True)
    model.save('results/waste_classifier.keras')

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()

    plt.savefig('results/training_history.png')
    print("[+] Model training complete. Outputs saved in results/")

if __name__ == '__main__':
    main()