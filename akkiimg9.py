from PIL import Image
import numpy as np

# Opening the test image and saving its object
img = Image.open(r'akki.jpg')

# Creating an array out of pixel values of the image
img_arr = np.array(img, np.uint8)

# Setting the value of every pixel in the 3rd channel to 0
# Change the 2 to 1 if wanting to drop the green channel 
# Change the 2 to 0 if wanting to drop the red channel
img_arr[::, ::, 2] = 0

# Creating an image from the modified array
img = Image.fromarray(img_arr)

# Displaying the image
img.show()

