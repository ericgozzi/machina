from machina.pixel import Picture
from machina.pixel import create_grid_of_pictures


a1 = Picture.from_file_path("./img/a1.png")
a2 = Picture.from_file_path("./img/a2.png")
a3 = Picture.from_file_path("./img/a3.png")
a4 = Picture.from_file_path("./img/a4.png")
a5 = Picture.from_file_path("./img/a5.png")
a6 = Picture.from_file_path("./img/a6.png")

grid = create_grid_of_pictures([a1, a2, a3, a4, a5, a6], grid_size=(3, 2), image_size=(1000, 1000),)

grid.save("cover.png")