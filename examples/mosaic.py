from machina import BayerDither, Picture

pic = Picture.from_file("./examples/doggo.jpg")

# filter = ShatterBox(divisions=30)
filter = BayerDither(threshold=16)
pic_mosaic = filter(pic)


pic_mosaic.save("./examples/doggo_mosaic.jpg")
