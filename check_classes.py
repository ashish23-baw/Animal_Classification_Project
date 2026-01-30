import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Apne dataset ka rasta (path) yahan likhein
# Maan lijiye aapka data 'dataset' folder mein hai
data_path = 'dataset' 

# 2. ImageDataGenerator ka use karke classes check karein
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# 3. Asli indices print karein
print("\n--- Model Class Indices ---")
print(train_generator.class_indices)
print("---------------------------\n")