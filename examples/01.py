import time

from machina import Artist, Line, Point, Polyline

a = Artist()

time.sleep(1)
point = Point(100, 10, 0)
a.draw(point)

line = Line(Point(0, 0, 0), Point(100, 100, 0))
a.draw(line)


polyline = Polyline(
    [Point(1000, 500, 0), Point(50, 100, 0), Point(100, 0, 0), Point(150, 100, 0)]
)
a.draw(polyline)

a.serve_forever()
