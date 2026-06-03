import tensorflow as tf
from tensorflow.keras import datasets, layers, models

def main():
    # Load dataset
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    # Normalize
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Build model
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train model
    print("Training model...")
    model.fit(x_train, y_train, epochs=5)

    # Evaluate model
    print("\nEvaluating model...")
    loss, accuracy = model.evaluate(x_test, y_test)

    print(f"\nTest Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
