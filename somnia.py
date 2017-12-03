import array

import numpy as np
from PIL import Image
import pyglet
from scipy import stats

from ldp8 import LPD8Controller


class SOMNIA(object):
    def __init__(self, source, height=256, width=256, radius=32, alpha=0.1):
        self.height = height
        self.width = width

        self.som = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        self.imdata = pyglet.image.ImageData(width, height, 'RGB',
                                             array.array('B', self.som.flatten()).tobytes())
        self.radius = radius
        self.alpha = alpha

        self.source = Image.open(source)
        self.cursor = [0, 0]
        self.cycle = 0

        self.neighborhood, self.weight = self.set_neighborhood_radius()
        self.changed = False

    def update(self, dt):
        if self.changed:
            self.neighborhood, self.weight = self.set_neighborhood_radius()
            self.changed = False

        sample = self.get_sample()
        bmu = self.find_bmu(sample)
        self.update_neighborhood(sample, bmu)
        self.imdata.set_data('RGB', self.width*3, array.array('B', self.som.flatten()).tobytes())

    def get_sample(self):
        pixel = self.source.getpixel(tuple(self.cursor))

        self.cursor[0] += 1
        if self.cursor[0] >= self.source.width:
            self.cursor[0] = 0
            self.cursor[1] += 1
        if self.cursor[1] >= self.source.height:
            self.cursor = [0, 0]
            self.cycle += 1

        return pixel

    def find_bmu(self, sample):
        dist = np.linalg.norm(self.som - sample, axis=2)
        idx = np.argmin(dist)
        return np.array([idx // self.width, idx % self.width]).reshape((2, 1, 1))

    def update_neighborhood(self, sample, bmu):
        neighbors = self.neighborhood + bmu
        if False:
            neighbors[0, neighbors[0] >= self.height] -= self.height
            neighbors[1, neighbors[1] >= self.width] -= self.width
        else:
            neighbors[0, neighbors[0] >= self.height] = self.height - 1
            neighbors[1, neighbors[1] >= self.width] = self.width - 1
            neighbors[0, neighbors[0] < 0] = 0
            neighbors[1, neighbors[1] < 0] = 0

        update_index = [neighbors[0], neighbors[1]]

        self.som[update_index] += (self.weight * (sample - self.som[update_index])).astype(np.uint8)
        self.som[self.som < 0] = 0
        self.som[self.som > 255] = 255

    def set_neighborhood_radius(self, mode='rectangle'):
        r = self.radius
        length = r*2+1
        neighborhood = np.mgrid[0:length, 0:length] - r
        distance = np.linalg.norm(neighborhood, axis=0)

        weight = np.zeros_like(distance)
        if mode == 'rectangle':
            weight = np.ones_like(distance)
        elif mode == 'gaussian':
            weight = r * stats.norm.pdf(distance, 0, r / 2)
        elif mode == 'dog':
            weight = r * (stats.norm.pdf(distance, 0, r / 2) - 0.5*stats.norm.pdf(distance, 0, r))

        weight[distance > r] = 0
        weight = self.alpha * weight.reshape((length, length, 1))

        return neighborhood, weight

    def save_screenshot(self, value, dt):
        n_pixels = self.cursor[0] + self.source.width*self.cursor[1]
        self.imdata.save('somnia_{}_{}.png'.format(self.cycle, n_pixels))

    def update_radius(self, value, dt):
        self.radius = int(1.0 * (value+1))
        if self.radius < 1:
            self.radius = 1
        self.changed = True

    def update_alpha(self, value, dt):
        self.alpha = value / 127
        self.changed = True


if __name__ == '__main__':
    somnia = SOMNIA('data/nin-seed2.jpg', width=256, height=256, alpha=0.05,
                    radius=16)
    controller = LPD8Controller()
    try:
        controller.open()
        controller.set_knob_callback(0, somnia.update_radius)
        controller.set_knob_callback(1, somnia.update_alpha)
        controller.set_pad_down_callback(0, somnia.save_screenshot)
    except OSError:
        pass

    window = pyglet.window.Window(somnia.width, somnia.height, vsync=True,
                                  fullscreen=False)

    @window.event
    def on_draw():
        window.clear()
        somnia.imdata.blit(0, 0, 0)

    pyglet.clock.schedule_interval(somnia.update, 1/30)
    pyglet.app.run()
