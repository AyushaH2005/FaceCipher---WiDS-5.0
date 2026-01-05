import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 30
DATA_DIR = "homer_bart"   


datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.1,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)


train_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

test_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)


model = Sequential([
    Flatten(input_shape=(64, 64, 3)),

    Dense(1024, activation='relu'),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),

    Dense(1, activation='sigmoid')
])



model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss='binary_crossentropy',
    metrics=['accuracy']
)



model.fit(
    train_data,
    epochs=50,
    steps_per_epoch=len(train_data)
)


test_loss, test_acc = model.evaluate(test_data)
print(f"\n Test Accuracy: {test_acc:.4f}")
