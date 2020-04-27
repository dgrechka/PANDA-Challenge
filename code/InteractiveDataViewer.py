from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

datasetPath = "data\\kaggleOfficial"
citoImagesPath = os.path.join(datasetPath,"train_images")
images = os.listdir(citoImagesPath)
images = [x for x in images if x.endswith(".tiff")]
#print(images)


toOpen = os.path.join(citoImagesPath,images[0])

print("file exists {0}".format(os.path.exists(toOpen)))


print("Attempting to open {0}".format(toOpen))

im = io.imread(toOpen)
print("Read")
#print(im)

plt.figure()
plt.imshow(im) 
plt.show()  # display it

#im.show()