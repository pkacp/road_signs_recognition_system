CATEGORIES = ["20", "30", "50", "60", "70", "Koniec pierwszeństwa", "Nakaz jazdy po lewej stronie znaku",
              "Nakaz jazdy po prawej stronie znaku",
              "Nakaz w lewo za znakiem", "Nakaz w prawo za znakiem", "Pierwszeństwo przejazdu",
              "Przejście dla pieszych", "Rondo", "Stop", "Ustąp pierwszeństwa", "Zakaz ruchu", "Zakaz wjazdu",
              "Zakaz wyprzedzania", "Zakaz zatrzymywania"]
CAN_BE_AUGMENTED_WITH_VERT_FLIP_INDEXES = [10, 14, 15, 16, 18]
IMG_WIDTH = 32
IMG_HEIGHT = 32
DESIRED_TRAINING_CATEGORY_SIZE = 3000
DESIRED_VALIDATION_CATEGORY_SIZE = DESIRED_TRAINING_CATEGORY_SIZE * 0.3
IMAGES_TRAIN_DIR = "/home/piotr/Obrazy2/train_images"
IMAGES_TEST_DIR = "/home/piotr/Obrazy2/test_images"
IMAGES_VAL_DIR = "/home/piotr/Obrazy2/val_images"
PICKLED_DIR = "../pickled_datasets"
PLOTSDIR = '../plots/'
X_TRAIN_PICKLED = f"{PICKLED_DIR}/X_train.pickle"
Y_TRAIN_PICKLED = f"{PICKLED_DIR}/y_train.pickle"
X_VAL_PICKLED = f"{PICKLED_DIR}/X_val.pickle"
Y_VAL_PICKLED = f"{PICKLED_DIR}/y_val.pickle"
X_TEST_PICKLED = f"{PICKLED_DIR}/X_test.pickle"
Y_TEST_PICKLED = f"{PICKLED_DIR}/y_test.pickle"
