from PIL import Image

img = Image.open(r"akki.jpg")
r, g, b = img.split()
len(r.histogram())
### 256 ###

print(r.histogram())

