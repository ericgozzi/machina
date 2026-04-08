from machina import WHITE, Picture

pic = Picture.from_file("./examples/doggo.jpg")

divisions = 8

width = pic.width
height = pic.height

wi = int(width / divisions)
hi = int(height / divisions)

mosaic = Picture.from_blank(width, height, WHITE)

for x in range(divisions):
    for y in range(divisions):
        quad = pic.copy()
        quad.crop(x * wi, y * hi, wi, hi)
        if (x + y) % 4 == 0:
            quad.rotate_90()
            quad.rotate_90()
        elif (x + y) % 4 == 1:
            pass
        elif (x + y) % 4 == 2:
            quad.rotate_90()
            quad.rotate_90()
        else:
            pass
        quad.paste_on(mosaic, x * wi, y * hi)


mosaic.show()
