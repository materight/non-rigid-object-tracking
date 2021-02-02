from random import randint
import webcolors

def closestColor(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def getColorName(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closestColor(requested_colour)
        actual_name = None
    return actual_name, closest_name

def pickNewColor(color_names_used):
    while(1):
        rgb = (randint(0, 255), randint(0, 255), randint(0, 255))
        actual_name, closest_name = getColorName(rgb)
        rgb = webcolors.name_to_rgb(closest_name)
        if closest_name not in color_names_used:
            break
    return (rgb[0], rgb[1], rgb[2])