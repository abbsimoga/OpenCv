d = [{"x":26, "y":101, "w":121, "h":59, "pixels":2355, "cx":76, "cy":122, "rotation":0.056336, "code":1, "count":1}, {"x":170, "y":138, "w":35, "h":27, "pixels":732, "cx":186, "cy":152, "rotation":3.103689, "code":1, "count":1}]

print(min([int(abs(rod["cx"]-120) + abs(rod["cy"]-120)) for rod in d]))