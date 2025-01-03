from PIL import Image, ImageFilter

img = Image.open(r"akki.jpg")

# Converting the image to grayscale, as Sobel Operator requires
# input image to be of mode Grayscale (L)
img = img.convert("L")

# Calculating Edges using the passed laplacian Kernel
final = img.filter(ImageFilter.Kernel((3, 3), (-1, -1, -1, -1, 8,
										-1, -1, -1, -1), 1, 0))

final.save("EDGE_sample2.jpg")

