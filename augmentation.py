from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image augmentation
augmentation = ImageDataGenerator(

    rescale=1./255,

    rotation_range=20,
    zoom_range=0.2,
    shear_range=0.2,

    horizontal_flip=True,

    validation_split=0.2
)

# Load augmented training data
train_data = augmentation.flow_from_directory(

    "dataset",

    target_size=(224, 224),

    batch_size=32,

    class_mode='categorical',

    subset='training'
)

# Load validation data
val_data = augmentation.flow_from_directory(

    "dataset",

    target_size=(224, 224),

    batch_size=32,

    class_mode='categorical',

    subset='validation'
)

print("Data augmentation ready!")