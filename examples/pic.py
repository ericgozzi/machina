from machina import Picture

pic = Picture.from_file("examples/checker.jpg")

pic2 = pic.copy()
pic2.crop(0, 0, 2000, 2000)
pic2.rotate_90()

pic = pic2.paste_on(pic, 0, 0)

pic.save("savedpic.png")
