import numpy as np
import h5py
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv3D, BatchNormalization, LeakyReLU, MaxPooling3D, UpSampling3D, Dropout, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau

keras.backend.clear_session()

# Load data
with h5py.File('modelnet40_dataset', 'r') as f:
    train_data = f['train_mat'][...]
    val_data = f['val_mat'][...]
    test_data = f['test_mat'][...]

train_data = train_data.reshape([-1, 32, 32, 32, 1])
val_data = val_data.reshape([-1, 32, 32, 32, 1])
test_data = test_data.reshape([-1, 32, 32, 32, 1])


# Model definition
input_img = Input(shape=(32, 32, 32, 1))

# Encoder
x = Conv3D(64, (3, 3, 3), padding='same')(input_img)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.1)(x)
x1 = MaxPooling3D((2, 2, 2), padding='same')(x)

x = Conv3D(128, (3, 3, 3), padding='same')(x1)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.1)(x)
x2 = MaxPooling3D((2, 2, 2), padding='same')(x)

x = Conv3D(256, (3, 3, 3), padding='same')(x2)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.1)(x)
encoded = MaxPooling3D((2, 2, 2), padding='same')(x)

# Decoder
x = Conv3D(256, (3, 3, 3), padding='same')(encoded)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.1)(x)
x = UpSampling3D((2, 2, 2))(x)

x = concatenate([x, x2])
x = Dropout(0.2)(x)
x = Conv3D(128, (3, 3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.1)(x)
x = UpSampling3D((2, 2, 2))(x)

x = concatenate([x, x1])
x = Dropout(0.2)(x)
x = Conv3D(64, (3, 3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.1)(x)
x = UpSampling3D((2, 2, 2))(x)

decoded = Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)

# Model training
autoencoder.fit(train_data, train_data,
                epochs=50,
                batch_size=128,
                validation_data=(val_data, val_data),
                callbacks=[reduce_lr])

autoencoder.save('autoencoder.h5')
print("Training finished...")
