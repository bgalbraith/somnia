import array

import numpy as np
from PIL import Image
import pyglet
from scipy import stats

from ldp8 import LPD8Controller


# ref_image = Image.open('data/kowloon-city.jpg')
# width, height = ref_image.size
# som = np.flipud(np.reshape(list(ref_image.getdata()), (height*width, 3))).astype(np.uint8)

height = 512  # 480  # 720, 1080
width = 256  # 854  # 1280, 1920
som = np.random.randint(0, 255, (height*width, 3), dtype=np.uint8)

index = np.arange(height*width).reshape((height, width))
imdata = pyglet.image.ImageData(width, height, 'RGB',
                                array.array('B', som.flatten()).tobytes())

window = pyglet.window.Window(width, height, vsync=True, fullscreen=False)
radius = 64
change_radius = False

alpha = 0.9
change_alpha = False


def set_neighborhood_radius(radius, alpha):
    length = radius*2+1
    neighborhood = np.mgrid[0:length, 0:length] - radius
    distance = np.linalg.norm(neighborhood, axis=0)
    # weight = alpha * np.ones_like(distance)
    # weight = alpha * radius * stats.norm.pdf(distance, 0, radius / 2)
    weight = alpha * radius * (stats.norm.pdf(distance, 0, radius / 2) -
                               0.5*stats.norm.pdf(distance, 0, radius))
    weight[distance > radius] = 0
    weight = weight.reshape((length, length, 1))
    return neighborhood, weight


neighborhood, weight = set_neighborhood_radius(radius, alpha)


source = Image.open('data/kowloon-city.jpg')
cursor = [0, 0]
cycle = 0


def get_sample():
    global cursor, cycle
    pixel = source.getpixel(tuple(cursor))
    cursor[0] += 1
    if cursor[0] >= source.width:
        cursor[0] = 0
        cursor[1] += 1
    if cursor[1] >= source.height:
        cursor = [0, 0]
        cycle += 1
    return pixel


def find_bmu(sample):
    dist = np.linalg.norm(som - sample, axis=1)
    # dmin = np.min(dist)
    # idx = np.random.choice(np.where(dist == dmin)[0])
    idx = np.argmin(dist)
    return np.array([idx // width, idx % width]).reshape((2, 1, 1))


def update_neighborhood(sample, bmu):
    neighbors = neighborhood + bmu
    if False:
        neighbors[0, neighbors[0] >= height] -= height
        neighbors[1, neighbors[1] >= width] -= width
    else:
        neighbors[0, neighbors[0] >= height] = height - 1
        neighbors[1, neighbors[1] >= width] = width - 1
        neighbors[0, neighbors[0] < 0] = 0
        neighbors[1, neighbors[1] < 0] = 0

    update_index = index[neighbors[0], neighbors[1]]
    som[update_index] += (weight * (sample - som[update_index])).astype(np.uint8)
    som[som < 0] = 0
    som[som > 255] = 255


def update(dt):
    global neighborhood, weight, radius, change_radius, alpha, change_alpha
    if change_radius or change_alpha:
        neighborhood, weight = set_neighborhood_radius(radius, alpha)
        change_radius = change_alpha = False
    sample = get_sample()
    bmu = find_bmu(sample)
    update_neighborhood(sample, bmu)
    imdata.set_data('RGB', width*3, array.array('B', som.flatten()).tobytes())


@window.event
def on_draw():
    window.clear()
    imdata.blit(0, 0, 0)


def update_radius(value, dt):
    global radius, change_radius
    radius = int(1.0 * (value+1))
    if radius < 1:
        radius = 1
    change_radius = True


def update_alpha(value, dt):
    global alpha, change_alpha
    alpha = value / 127
    change_alpha = True


def save_screenshot(value, dt):
    n_pixels = cursor[0] + source.width*cursor[1]
    imdata.save('somnia_{}_{}.png'.format(cycle, n_pixels))


controller = LPD8Controller()
try:
    controller.open()
    controller.set_knob_callback(0, update_radius)
    controller.set_knob_callback(1, update_alpha)
    controller.set_pad_down_callback(0, save_screenshot)
except OSError:
    pass

pyglet.clock.schedule_interval(update, 1/30.0)
pyglet.app.run()
