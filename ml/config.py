# Dataset and training configuration
DATA_DIR = "./dataset"       # Path to the folder containing the images
BATCH_SIZE = 8
NUM_EPOCHS = 25
LEARNING_RATE = 1e-3
TRAIN_RATIO = 0.8            # Ratio for train/validation split

# Image preprocessing configuration
IMAGE_SIZE = 320           # Images will be resized to IMAGE_SIZE x IMAGE_SIZE
# Using ImageNet normalization values
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# Number of classes (acne levels 0,1,2,3)
NUM_CLASSES = 4

MODEL_SAVE_PATH = "./saved_models/light_cnn.pth"