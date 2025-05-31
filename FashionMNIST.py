import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

train_images = train_images[..., np.newaxis] / 255.0
test_images = test_images[..., np.newaxis] / 255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
def build_basic_cnn():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model
def build_resnet_like():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x_shortcut = x
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.Add()([x, x_shortcut])
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    return models.Model(inputs, outputs)


def compile_and_train(model, train_images, train_labels, model_name="model"):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=5, validation_split=0.1)
    model.save(f"{model_name}.h5")
def evaluate_and_plot(model, test_images, test_labels):
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(f"Test accuracy: {test_acc:.2f}")

    predictions = model.predict(test_images)
    y_pred = np.argmax(predictions, axis=1)

    cm = confusion_matrix(test_labels, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

basic_cnn = build_basic_cnn()
compile_and_train(basic_cnn, train_images, train_labels, "basic_cnn")

resnet_model = build_resnet_like()
compile_and_train(resnet_model, train_images, train_labels, "resnet_like")

basic_cnn_loaded = load_model("basic_cnn.h5")
print("Evaluating Basic CNN:")
evaluate_and_plot(basic_cnn_loaded, test_images, test_labels)

resnet_loaded = load_model("resnet_like.h5")
print("Evaluating ResNet-like CNN:")
evaluate_and_plot(resnet_loaded, test_images, test_labels)
