import xml.etree.ElementTree as ET


from .color import Color



class Drawing:



    def __init__(self, width=500, height=500):
        self.width = width
        self.height = height

        # Create the root SVG element
        self.svg = ET.Element('svg', xmlns="http://www.w3.org/2000/svg",
                              width = str(width),
                              height = str(height))
        
    
    def add_line(self, x1, y1, x2, y2, **kwargs):
        """
        Add a line element to the SVG.

        Parameters
        ----------
        x1 : float
            The x-coordinate of the start point.
        y1 : float
            The y-coordinate of the start point.
        x2 : float
            The x-coordinate of the end point.
        y2 : float
            The y-coordinate of the end point.

        Keyword Arguments (kwargs)
        -------------------------
        stroke_color : Color, optional
            The stroke color of the line. Must be a Color object. Default is Color.WHITE.
        stroke_width : float, optional
            The width of the line stroke. Default is 1.
        stroke_opacity : float, optional
            Opacity of the stroke (0.0–1.0). Default is 1.0.
        opacity : float, optional
            Overall opacity of the line (0.0–1.0). Default is 1.0.
        stroke_linecap : str, optional
            Style of the line endings. Options: "butt", "round", "square". Default is "butt".
        stroke_linejoin : str, optional
            Style of the line joins. Options: "miter", "round", "bevel". Default is "miter".
        display : str, optional
            Display property of the SVG element. Default is "inline".

        Additional kwargs
        -----------------
        Any other valid SVG attributes (e.g., 'id', 'transform', 'style', 'pointer-events') can also
        be passed via kwargs. These attributes will be added directly to the SVG element.

        Notes
        -----
        - Only Color objects are accepted for stroke_color.
        - All numerical values are converted to strings internally to comply with SVG XML attribute requirements.
        """
        attrs = {
            "x1": str(x1),
            "y1": str(y1),
            "x2": str(x2),
            "y2": str(y2),
            "stroke": kwargs.get("stroke_color", Color.WHITE).hex,
            "stroke-width": str(kwargs.get("stroke_width", 1)),
            "stroke-opacity": str(kwargs.get("stroke_opacity", 1.0)),
            "opacity": str(kwargs.get("opacity", 1.0)),
            "stroke-linecap": kwargs.get("stroke_linecap", "butt"),
            "stroke-linejoin": kwargs.get("stroke_linejoin", "miter"),
            "display": kwargs.get("display", "inline")
        }

        line = ET.Element("line", **attrs)
        self.svg.append(line)


    def add_rectangle(self, x, y, width, height, **kwargs):
        """
        Add a rectangle element to the SVG.

        Parameters
        ----------
        x : float
            The x-coordinate of the top-left corner.
        y : float
            The y-coordinate of the top-left corner.
        width : float
            The width of the rectangle.
        height : float
            The height of the rectangle.

        Keyword Arguments (kwargs)
        -------------------------
        fill_color : Color, optional
            The fill color of the rectangle. Must be a Color object. Default is Color.WHITE.
        stroke_color : Color, optional
            The stroke (border) color of the rectangle. Must be a Color object. Default is Color.WHITE.
        stroke_width : float, optional
            The width of the rectangle's stroke. Default is 1.
        fill_opacity : float, optional
            Opacity of the fill (0.0–1.0). Default is 1.0.
        stroke_opacity : float, optional
            Opacity of the stroke (0.0–1.0). Default is 1.0.
        opacity : float, optional
            Overall opacity of the rectangle (0.0–1.0). Default is 1.0.
        rx : float, optional
            Horizontal corner radius (for rounded corners). Default is 0.
        ry : float, optional
            Vertical corner radius (for rounded corners). Default is 0.
        display : str, optional
            Display property of the SVG element. Default is "inline".
        stroke_linecap : str, optional
            Style of stroke line endings. Default is "butt".
        stroke_linejoin : str, optional
            Style of stroke line joins. Default is "miter".

        Additional kwargs
        -----------------
        Any other valid SVG attributes (e.g., 'id', 'transform', 'style', 'pointer-events') can also
        be passed via kwargs.

        Notes
        -----
        - Only Color objects are accepted for fill_color and stroke_color.
        - All numerical values are converted to strings internally to comply with SVG XML attribute requirements.
        """

        attrs = {
            "x": str(x),
            "y": str(y),
            "width": str(width),
            "height": str(height),
            "rx": str(kwargs.get("rx", 0)),
            "ry": str(kwargs.get("ry", 0)),
            "fill": kwargs.get("fill_color", Color.WHITE).hex,
            "stroke": kwargs.get("stroke_color", Color.WHITE).hex,
            "stroke-width": str(kwargs.get("stroke_width", 1)),
            "fill-opacity": str(kwargs.get("fill_opacity", 1.0)),
            "stroke-opacity": str(kwargs.get("stroke_opacity", 1.0)),
            "opacity": str(kwargs.get("opacity", 1.0)),
            "display": kwargs.get("display", "inline"),
            "stroke-linecap": kwargs.get("stroke_linecap", "butt"),
            "stroke-linejoin": kwargs.get("stroke_linejoin", "miter")
        }

        rect = ET.Element("rect", **attrs)
        self.svg.append(rect)


    def add_circle(self, center_x, center_y, radius, **kwargs):
        """
        Add a circle element to the SVG.

        Parameters
        ----------
        center_x : float
            The x-coordinate of the circle's center.
        center_y : float
            The y-coordinate of the circle's center.
        radius : float
            The radius of the circle.

        Keyword Arguments (kwargs)
        -------------------------
        fill_color : Color, optional
            The fill color of the circle. Must be a Color object. Default is Color.WHITE.
        stroke_color : Color, optional
            The stroke (border) color of the circle. Must be a Color object. Default is Color.WHITE.
        stroke_width : float, optional
            The width of the stroke. Default is 1.
        fill_opacity : float, optional
            Opacity of the fill (0.0–1.0). Default is 1.0.
        stroke_opacity : float, optional
            Opacity of the stroke (0.0–1.0). Default is 1.0.
        opacity : float, optional
            Overall opacity of the circle (0.0–1.0). Default is 1.0.
        display : str, optional
            Display property of the SVG element. Default is "inline".
        stroke_linecap : str, optional
            Style of the stroke line endings. Options: "butt", "round", "square". Default is "butt".
        stroke_linejoin : str, optional
            Style of the stroke line joins. Options: "miter", "round", "bevel". Default is "miter".

        Additional kwargs
        -----------------
        Any other valid SVG attributes (e.g., 'id', 'transform', 'style', 'pointer-events') can also
        be passed via kwargs. These attributes will be added directly to the SVG element.

        Notes
        -----
        - Only Color objects are accepted for fill_color and stroke_color.
        - All numerical values are converted to strings internally to comply with SVG XML attribute requirements.
        """
        attrs = {
            "cx": str(center_x),
            "cy": str(center_y),
            "r": str(radius),
            "fill": kwargs.get("fill_color", Color.WHITE).hex,
            "stroke": kwargs.get("stroke_color", Color.WHITE).hex,
            "stroke-width": str(kwargs.get("stroke_width", 1)),
            "fill-opacity": str(kwargs.get("fill_opacity", 1.0)),
            "stroke-opacity": str(kwargs.get("stroke_opacity", 1.0)),
            "opacity": str(kwargs.get("opacity", 1.0)),
            "display": kwargs.get("display", "inline"),
            "stroke-linecap": kwargs.get("stroke_linecap", "butt"),
            "stroke-linejoin": kwargs.get("stroke_linejoin", "miter")
        }

        circle = ET.Element("circle", **attrs)
        self.svg.append(circle)


    def add_polyline(self, points, **kwargs):
        """
        Add a polyline element to the SVG.

        Parameters
        ----------
        points : list of tuples
            A list of (x, y) coordinate tuples defining the vertices of the polyline.

        Keyword Arguments (kwargs)
        -------------------------
        fill_color : Color, optional
            Fill color (usually none for polyline).
        stroke_color : Color, optional
            Stroke color. Must be a Color object. Default is Color.WHITE.
        stroke_width : float, optional
            Width of the line. Default is 1.
        fill_opacity : float, optional
            Opacity of the fill (0.0–1.0). Default is 0.0.
        stroke_opacity : float, optional
            Opacity of the stroke (0.0–1.0). Default is 1.0.
        opacity : float, optional
            Overall opacity (0.0–1.0). Default is 1.0.
        stroke_linecap : str, optional
            Line ending style. Default is "butt".
        stroke_linejoin : str, optional
            Line join style. Default is "miter".
        display : str, optional
            Display property. Default is "inline".

        Additional kwargs
        -----------------
        Any other valid SVG attributes can be passed and will be added to the element.
        """

        # Convert points to SVG string format
        points_str = " ".join(f"{x},{y}" for x, y in points)

        attrs = {
            "points": points_str,
            "fill": kwargs.get("fill_color", None).hex,
            "stroke": kwargs.get("stroke_color", Color.WHITE).hex,
            "stroke-width": str(kwargs.get("stroke_width", 1)),
            "fill-opacity": str(kwargs.get("fill_opacity", 0.0)),
            "stroke-opacity": str(kwargs.get("stroke_opacity", 1.0)),
            "opacity": str(kwargs.get("opacity", 1.0)),
            "stroke-linecap": kwargs.get("stroke_linecap", "butt"),
            "stroke-linejoin": kwargs.get("stroke_linejoin", "miter"),
            "display": kwargs.get("display", "inline")
        }

        polyline = ET.Element("polyline", **attrs)
        self.svg.append(polyline)


    def add_polygon(self, points, **kwargs):
        """
        Add a polygon element to the SVG.

        Parameters
        ----------
        points : list of tuples
            A list of (x, y) coordinate tuples defining the vertices of the polygon. 
            The polygon is automatically closed.

        Keyword Arguments (kwargs)
        -------------------------
        fill_color : Color, optional
            Fill color. Must be a Color object. Default is Color.WHITE.
        stroke_color : Color, optional
            Stroke color. Must be a Color object. Default is Color.WHITE.
        stroke_width : float, optional
            Width of the stroke. Default is 1.
        fill_opacity : float, optional
            Opacity of the fill (0.0–1.0). Default is 1.0.
        stroke_opacity : float, optional
            Opacity of the stroke (0.0–1.0). Default is 1.0.
        opacity : float, optional
            Overall opacity (0.0–1.0). Default is 1.0.
        stroke_linecap : str, optional
            Line ending style. Default is "butt".
        stroke_linejoin : str, optional
            Line join style. Default is "miter".
        display : str, optional
            Display property. Default is "inline".

        Additional kwargs
        -----------------
        Any other valid SVG attributes can be passed and will be added to the element.
        """

        # Convert points to SVG string format
        points_str = " ".join(f"{x},{y}" for x, y in points)

        attrs = {
            "points": points_str,
            "fill": kwargs.get("fill_color", Color.WHITE).hex,
            "stroke": kwargs.get("stroke_color", Color.WHITE).hex,
            "stroke-width": str(kwargs.get("stroke_width", 1)),
            "fill-opacity": str(kwargs.get("fill_opacity", 1.0)),
            "stroke-opacity": str(kwargs.get("stroke_opacity", 1.0)),
            "opacity": str(kwargs.get("opacity", 1.0)),
            "stroke-linecap": kwargs.get("stroke_linecap", "butt"),
            "stroke-linejoin": kwargs.get("stroke_linejoin", "miter"),
            "display": kwargs.get("display", "inline")
        }


        polygon = ET.Element("polygon", **attrs)
        self.svg.append(polygon)


    def add_ellipse(self, cx, cy, rx, ry, **kwargs):
        """
        Add an ellipse element to the SVG.

        Parameters
        ----------
        cx : float
            The x-coordinate of the ellipse center.
        cy : float
            The y-coordinate of the ellipse center.
        rx : float
            The horizontal radius of the ellipse.
        ry : float
            The vertical radius of the ellipse.

        Keyword Arguments (kwargs)
        -------------------------
        fill_color : Color, optional
            Fill color of the ellipse. Must be a Color object. Default is Color.WHITE.
        stroke_color : Color, optional
            Stroke color of the ellipse. Must be a Color object. Default is Color.WHITE.
        stroke_width : float, optional
            Width of the stroke. Default is 1.
        fill_opacity : float, optional
            Opacity of the fill (0.0–1.0). Default is 1.0.
        stroke_opacity : float, optional
            Opacity of the stroke (0.0–1.0). Default is 1.0.
        opacity : float, optional
            Overall opacity (0.0–1.0). Default is 1.0.
        stroke_linecap : str, optional
            Style of stroke line endings. Default is "butt".
        stroke_linejoin : str, optional
            Style of stroke line joins. Default is "miter".
        display : str, optional
            Display property. Default is "inline".

        Additional kwargs
        -----------------
        Any other valid SVG attributes (e.g., 'id', 'transform', 'style', 'pointer-events') 
        can also be passed via kwargs.

        Notes
        -----
        - Only Color objects are accepted for fill_color and stroke_color.
        - All numerical values are converted to strings internally to comply with SVG XML attribute requirements.
        """

        attrs = {
            "cx": str(cx),
            "cy": str(cy),
            "rx": str(rx),
            "ry": str(ry),
            "fill": kwargs.get("fill_color", Color.WHITE).hex,
            "stroke": kwargs.get("stroke_color", Color.WHITE).hex,
            "stroke-width": str(kwargs.get("stroke_width", 1)),
            "fill-opacity": str(kwargs.get("fill_opacity", 1.0)),
            "stroke-opacity": str(kwargs.get("stroke_opacity", 1.0)),
            "opacity": str(kwargs.get("opacity", 1.0)),
            "stroke-linecap": kwargs.get("stroke_linecap", "butt"),
            "stroke-linejoin": kwargs.get("stroke_linejoin", "miter"),
            "display": kwargs.get("display", "inline")
        }


        ellipse = ET.Element("ellipse", **attrs)
        self.svg.append(ellipse)


    def save(self, filepath='output.svg'):
        """Save the SVG to a file."""
        tree = ET.ElementTree(self.svg)
        tree.write(filepath, encoding="utf-8", xml_declaration=True)












if __name__ == '__main__':


    dwg = Drawing(1000, 1000)


    dwg.add_circle(500, 500, 200, stroke_color=Color.RED, stroke_width=10)
    dwg.add_line(100, 100, 900, 900, stroke_color=Color.BLUE, stroke_width=20)
    dwg.add_polyline([(10, 10), (900, 10), (900, 900), (10, 900)], stroke_width=10, stroke_color=Color.YELLOW)
    
    dwg.save()