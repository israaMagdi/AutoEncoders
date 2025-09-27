
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from data import load_transform_data

# Input: 32x32x1
input_img = layers.Input(shape=(32, 32, 1))

# Encoder
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
encoded = layers.MaxPooling2D((2, 2), padding="same")(x)

print("Encoder output shape:", encoded.shape)

# Decoder
x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)
print("Decoder output shape:", decoded.shape)

# Autoencoder model
autoencoder = models.Model(input_img, decoded)

autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder.summary()

x_train, y_train, x_test, y_test = load_transform_data()
history = autoencoder.fit(
    x_train, x_train,   # input = output
    epochs=20,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test)
)

decoded_imgs = autoencoder.predict(x_test[:10])

plt.figure(figsize=(20, 4))
for i in range(10):
    # Original
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(x_test[i].reshape(32, 32), cmap="gray")
    plt.axis("off")

    # Reconstructed
    ax = plt.subplot(2, 10, i + 1 + 10)
    plt.imshow(decoded_imgs[i].reshape(32, 32), cmap="gray")
    plt.axis("off")
plt.show()