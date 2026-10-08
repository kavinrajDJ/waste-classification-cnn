import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

IMG_HEIGHT = 150
IMG_WIDTH = 150
CLASS_NAMES = ['organic', 'recyclable']

def predict_single_image(img_path, model_path='results/waste_classifier.keras'):
    if not os.path.exists(model_path):
        print(f"[!] Model file '{model_path}' not found. Run model.py first.")
        return

    model = tf.keras.models.load_model(model_path)

    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_batch)
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx] * 100

    predicted_label = CLASS_NAMES[predicted_idx] if predicted_idx < len(CLASS_NAMES) else str(predicted_idx)

    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Prediction: {predicted_label} ({confidence:.2f}%)")
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/sample_prediction.png', bbox_inches='tight')
    plt.show()

    print(f"Result: {predicted_label} with {confidence:.2f}% confidence.")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        predict_single_image(sys.argv[1])
    else:
        print("Usage: python src/predict.py <path_to_image>")