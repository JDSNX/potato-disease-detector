import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 50


def get_dataset():
    dataset = tf.keras.utils.image_dataset_from_directory(
        "plant-village/PlantVillage",
        shuffle=True,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
    )

    return dataset


def get_dataset_partitions_tf(
    ds,
    *,
    train_split=0.8,
    val_split=0.1,
    test_split=0.1,
    shuffle=True,
    shuffle_size=10000,
):
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)

    training_sets = ds.take(int(len(ds) * train_split))
    temp_sets = ds.skip(int(len(ds) * train_split))
    validation_sets = temp_sets.take(int(len(ds) * val_split))
    testing_sets = temp_sets.skip(int(len(ds) * test_split))

    return training_sets, validation_sets, testing_sets


def shuffle_datasets():
    train_ds, val_ds, test_ds = get_dataset_partitions_tf(get_dataset())

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds


def create_model():
    resize_and_rescale = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
            tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255),
        ]
    )

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomFlip(
                "horizontal_and_vertical"
            ),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        ]
    )

    model = tf.keras.models.Sequential(
        [
            resize_and_rescale,
            data_augmentation,
            tf.keras.layers.Conv2D(
                16,
                (3, 3),
                activation="relu",
                input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS),
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )

    return model


def compile_model():
    model = create_model()

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer="adam",
        metrics=["accuracy"],
    )

    train_ds, val_ds, test_ds = shuffle_datasets()

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        validation_data=val_ds,
    )

    training_and_validation_plot(model, history, test_ds)
    save_model(model)


def save_model(model):
    from datetime import datetime
    from os import listdir

    date = datetime.today().strftime("%Y%m%d")
    model_version = max([int(i[-1]) for i in listdir("models")]) + 1
    model.save(f"models/{date}-{model_version}")


def training_and_validation_plot(model, history, test_ds):
    scores = model.evaluate(test_ds, verbose=1)

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(range(EPOCHS), acc, label="Training Accuracy")
    plt.plot(range(EPOCHS), val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(range(EPOCHS), loss, label="Training Accuracy")
    plt.plot(range(EPOCHS), val_loss, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")


def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array, verbose=0)

    predicted_class = get_dataset()[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence


if __name__ == "__main__":
    compile_model()
