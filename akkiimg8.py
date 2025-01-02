from PIL import Image

# Creating a image object, of the sample image
img = Image.open(r'akki.jpg')

# A 12-value tuple which is a transform matrix for dropping 
# green channel (in this case)
matrix = ( 0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0)

# Transforming the image to RGB using the aforementioned matrix 
img = img.convert("RGB", matrix)

# Displaying the image 
img.show()

