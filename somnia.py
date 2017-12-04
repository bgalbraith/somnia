import array

import numpy as np
from PIL import Image
import pyglet
from scipy import stats
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ldp8 import LPD8Controller


class SOMNIA(object):
    def __init__(self, source, height=256, width=256, radius=32, alpha=0.1,
                 wrap=(True, True)):
        self.height = height
        self.width = width
        self.k_height = height
        self.k_width = width
        self.wrap = wrap

        self.som = 255*torch.rand(1, 3, height, width)
        self.delta = torch.zeros(1, 3, height, width)
        img = self.som.numpy().squeeze().transpose((1, 2, 0)).flatten()
        self.imdata = pyglet.image.ImageData(width, height, 'RGB',
                                             array.array('B', img).tobytes())
        self.radius = radius
        self.alpha = alpha

        self.dataset = SOMNIADataset(source)

        self.weight = torch.zeros(height, width)
        self.neighborhood, self.template = self.set_neighborhood_radius()
        self.changed = False

    def update(self, dt):
        if self.changed:
            self.neighborhood, self.weight = self.set_neighborhood_radius()
            self.changed = False

        sample = self.get_sample()
        bmu = self.find_bmu(sample)
        self.update_neighborhood(bmu)
        img = self.som.numpy().squeeze().transpose((1, 2, 0)).flatten()
        self.imdata.set_data('RGB', self.width*3, array.array('B', img).tobytes())

    def get_sample(self):
        return next(self.dataset)

    def find_bmu(self, sample):
        self.delta = sample - self.som
        dist = -1*torch.norm(self.delta, 2, 1)
        _, idx = F.max_pool2d(dist, (self.k_height, self.k_width),
                              return_indices=True)

        idx = idx.data.numpy().squeeze()
        y = idx // self.width
        x = idx % self.width

        return torch.from_numpy(np.array([[y], [x]]))

    def update_neighborhood(self, bmu):
        neighbors = self.neighborhood + bmu

        mask = torch.ones(len(self.template)).type(torch.ByteTensor)
        if self.wrap[0]:
            neighbors[0] %= self.height
        else:
            mask *= neighbors[0] >= 0
            mask *= neighbors[0] < self.height

        if self.wrap[1]:
            neighbors[1] %= self.width
        else:
            mask *= neighbors[1] >= 0
            mask *= neighbors[1] < self.width

        neighbors = neighbors.masked_select(mask).view(2, -1)
        template = self.template.masked_select(mask)
        self.weight *= 0
        self.weight[neighbors[0], neighbors[1]] = template
        self.som += self.weight * self.delta
        self.som[self.som < 0] = 0
        self.som[self.som > 255] = 255

    def set_neighborhood_radius(self, mode='dog'):
        r = self.radius
        length = r*2+1
        n = np.mgrid[0:length, 0:length].reshape((2, length*length)) - r
        d = np.linalg.norm(n, axis=0)

        weight = np.zeros(len(d))
        if mode == 'rectangle':
            weight = np.ones(len(d))
        elif mode == 'gaussian':
            weight = r * stats.norm.pdf(d, 0, r / 2)
        elif mode == 'dog':
            g1 = stats.norm.pdf(d, 0, r / 2)
            g2 = 0.5*stats.norm.pdf(d, 0, r)
            weight = r * (g1 - g2)

        weight[d > r] = 0
        weight *= self.alpha

        return torch.from_numpy(n), torch.from_numpy(weight.astype(np.float32))

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


class SOMNIADataset(Dataset):
    def __init__(self, source):
        image = Image.open(source)
        self.data = torch.Tensor(list(image.getdata())).view(-1, 3, 1, 1)
        self.cursor = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item].view(1, 3, 1, 1)

    def __next__(self):
        if self.cursor >= len(self):
            self.cursor = 0
        sample = self[self.cursor]
        self.cursor += 1
        return sample


if __name__ == '__main__':
    somnia = SOMNIA('data/real_nvp.jpeg', width=256, height=256, alpha=0.25,
                    radius=16, wrap=(False, False))
    controller = LPD8Controller()
    try:
        controller.open()
        controller.set_knob_callback(0, somnia.update_radius)
        controller.set_knob_callback(1, somnia.update_alpha)
        controller.set_pad_down_callback(0, somnia.save_screenshot)
    except OSError:
        pass

    window = pyglet.window.Window(somnia.width, somnia.height, vsync=False,
                                  fullscreen=False)

    @window.event
    def on_draw():
        window.clear()
        somnia.imdata.blit(0, 0, 0)

    pyglet.clock.schedule(somnia.update)
    pyglet.app.run()
