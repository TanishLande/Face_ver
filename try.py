import glob
import os

POS_PATH = os.path.join("data", "positive")
NEV_PATH = os.path.join("data", "negatives")
ANC_PATH = os.path.join("data", "anchor")

num_images = len(glob.glob(NEV_PATH + "/*.jpg"))
print("Total images in ANC_PATH:", num_images)
