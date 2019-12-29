# CATEGORIES = ["20", "30", "40", "50", "70", "Ustąp pierwszeństwa", "Koniec pierwszeństwa", "Pierwszeństwo przejazdu",
#               "Przejście dla pieszych", "Stop", "Rondo", "Zakaz zatrzymywania", "Zakaz wjazdu", "Zakaz ruchu"]  # small dataset

CATEGORIES = ["20", "30", "50", "60", "70", "Nakaz jazdy po lewej stronie znaku", "Nakaz jazdy po prawej stronie znaku",
              "Nakaz w lewo za znakiem", "Nakaz w prawo za znakiem", "Pierwszeństwo przejazdu",
              "Przejście dla pieszych", "Rondo", "Stop", "Ustąp pierwszeństwa", "Zakaz ruchu", "Zakaz wjazdu",
              "Zakaz wyprzedzania", "Zakaz zatrzymywania"]  # bigger dataset with GTSRB
CAN_BE_AUGMENTED_WITH_VERT_FLIP_INDEXES = [9, 13, 14, 15, 17]
IMG_WIDTH = 50
IMG_HEIGHT = 50
DESIRED_CATEGORY_SIZE = 3000
# IMAGES_BASE_DIR = "/home/piotr/Obrazy"  # pc
IMAGES_BASE_DIR = "/home/piotr/Obrazy2"  # pc z GTSRB
# IMAGES_BASE_DIR = "/home/piokac/Dokumenty/!inzynierka/Obrazy"  # laptop
PICKLED_DIR = "../pickled_datasets"
PLOTSDIR = '../plots/'
X_PICKLED = f"{PICKLED_DIR}/X.pickle"
Y_PICKLED = f"{PICKLED_DIR}/y.pickle"
