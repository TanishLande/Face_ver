import os

# POS_PATH = os.path.join("data", "positive")
NEV_PATH = os.path.join("data", "negatives")
# ANC_PATH = os.path.join("data", "anchor")

# os.makedirs(POS_PATH)
# os.makedirs(NEV_PATH)
# os.makedirs(ANC_PATH)


for direct in os.listdir("lfw-deepfunneled"):
    full_dir_path = os.path.join("lfw-deepfunneled", direct)
    for file in os.listdir(full_dir_path):
        EX_PATH = os.path.join("lfw-deepfunneled", direct, file)
        NEW_PATH = os.path.join(NEV_PATH, file)
        os.replace(EX_PATH, NEW_PATH)
    