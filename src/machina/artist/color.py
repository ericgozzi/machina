import colorsys



class Color:
    
    def __init__(self, r=0, g=0, b=0, a=1.0):
        self.r = r
        self.g = g
        self.b = b
        self.a = a


    def __repr__(self):
        return f'Color: r: {self.r}, g: {self.g}, b: {self.b}, a: {self.a}'
    
    def __str__(self):
        return f'Color: r: {self.r}, g: {self.g}, b: {self.b}, a: {self.a}'



    @property
    def r(self):
        return self._r
    

    @property
    def g(self):
        return self._g
    
    @property
    def b(self):
        return self._b
    
    @property
    def a(self):
        return self._a
    

    @r.setter
    def r(self, value):
        #Automatically detect scale
        if isinstance(value, int):
            value = value / 255
        elif isinstance(value, float) and value > 1.0:
            # Handle floats mistakenly given in 0-255 range
            value = value/255
        # clamp to 0.0-1-0
        self._r = max(0.0, min(1.0, value))


    @g.setter
    def g(self, value):
        #Automatically detect scale
        if isinstance(value, int):
            value = value / 255
        elif isinstance(value, float) and value > 1.0:
            # Handle floats mistakenly given in 0-255 range
            value = value/255
        # clamp to 0.0-1-0
        self._g = max(0.0, min(1.0, value))


    @b.setter
    def b(self, value):
        #Automatically detect scale
        if isinstance(value, int):
            value = value / 255
        elif isinstance(value, float) and value > 1.0:
            # Handle floats mistakenly given in 0-255 range
            value = value/255
        # clamp to 0.0-1-0
        self._b = max(0.0, min(1.0, value))

    
    @a.setter
    def a(self, value):
        #Automatically detect scale
        if isinstance(value, int):
            value = value / 255
        elif isinstance(value, float) and value > 1.0:
            # Handle floats mistakenly given in 0-255 range
            value = value/255
        # clamp to 0.0-1-0
        self._a = max(0.0, min(1.0, value))


    @property
    def rgb(self):
        return (int(self.r*255), int(self.g*255), int(self.b*255))
    
    @property
    def rgba(self):
        return (int(self.r*255), int(self.g*255), int(self.b*255), int(self.a*255))
    
    @property
    def hex(self):
        r, g, b = self.rgb
        return f"#{r:02x}{g:02x}{b:02x}"
    
    @property
    def hexa(self):
        r, g, b, a = self.rgba
        return f"#{r:02x}{g:02x}{b:02x}{a:02x}"
    
    @property
    def hsl(self):
        r, g, b = self._r, self._g, self._b
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        return (h, s, l)
    



    #*********************************
    # CONSTRUCTORS
    #*********************************
    
    @classmethod
    def from_hsl(cls, h, s, l):
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        return cls(r, g, b)





    #*********************************
    # OPERATIONS
    #*********************************

    def lighten(self, amount=0.1):
        h, s, l = self.hsl
        l = max(0.0, min(1.0, l + amount))
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        self.r, self.g, self.b = r, g, b

    
    def darken(self, amount=0.1):
        self.lighten(amount=-amount)
    

    #*********************************
    # STANDARDS
    #*********************************
    

Color.BLACK = Color(0.0, 0.0, 0.0)
Color.GRAY = Color(0.5, 0.5, 0.5)
Color.WHITE = Color(1.0, 1.0, 1.0)
Color.RED   = Color(1.0, 0.0, 0.0)
Color.GREEN = Color(0.0, 1.0, 0.0)
Color.BLUE  = Color(0.0, 0.0, 1.0)
Color.CYAN = Color(0.0, 1.0, 1.0)
Color.MAGENTA = Color(1.0, 0.0, 1.0)
Color.YELLOW = Color(1.0, 1.0, 0.0)

    







if __name__ == "__main__":
    print(Color.WHITE)
