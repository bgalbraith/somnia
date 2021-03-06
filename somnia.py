import argparse
import array
from functools import partial
import time

import numpy as np
from PIL import Image
import pyglet
from scipy import stats
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ldp8 import LPD8Controller


parser = argparse.ArgumentParser(description='SOMNIA')
parser.add_argument('--source', type=str, help='image to draw samples from')
parser.add_argument('--height', type=int, default=256,
                    help='SOM height (default: 256)')
parser.add_argument('--width', type=int, default=256,
                    help='SOM width (default: 256)')
parser.add_argument('--shuffle', action='store_true',
                    help='shuffle the input samples')
parser.add_argument('--mode', type=str, default='rectangle',
                    help='initial neighborhood window function, can be rectangle, linear, gaussian, dog (default: rectangle)')
parser.add_argument('--alpha', type=float, default=0.05,
                    help='initial learning rate (default: 0.05)')
parser.add_argument('--radius', type=int, default=16,
                    help='initial neighborhood radius (default: 16)')
parser.add_argument('--wrap_x', action='store_true',
                    help='wrap neighborhood around the x axis')
parser.add_argument('--wrap_y', action='store_true',
                    help='wrap neighborhood around the y axis')
parser.add_argument('--tiling', type=int, default=1,
                    help='sub SOM factor')
parser.add_argument('--lpd8', action='store_true',
                    help='use the AKAI LPD8 controller')
args = parser.parse_args()


class SOMNIA(object):
    def __init__(self, dataset, height=256, width=256, radius=32, alpha=0.1,
                 mode='rectangle', wrap=(True, True), tiling=1):
        self.height = height
        self.width = width
        self.tiling = tiling
        self.k_height = height // self.tiling
        self.k_width = width // self.tiling
        self.wrap = wrap
        self.mode = mode

        self.som = 255*torch.rand(1, 3, height, width)
        self.delta = torch.zeros(1, 3, height, width)
        img = self.som.numpy().squeeze().transpose((1, 2, 0)).flatten()
        self.imdata = pyglet.image.ImageData(width, height, 'RGB',
                                             array.array('B', img).tobytes())
        self.radius = radius
        self.alpha = alpha

        self.dataset = iter(dataset)

        self.weight = torch.zeros(height, width)
        self.neighborhood, self.template = self.set_neighborhood_radius()
        self.changed = False

    def update(self, dt):
        if self.changed:
            self.neighborhood, self.template = self.set_neighborhood_radius()
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

        return torch.from_numpy(np.array([y.flatten(), x.flatten()]))

    def update_neighborhood(self, bmu):
        for i in range(bmu.shape[1]):
            neighbors = self.neighborhood + bmu[:, [i]]

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

    def set_neighborhood_radius(self):
        r = self.radius
        length = r*2+1
        n = np.mgrid[0:length, 0:length].reshape((2, length*length)) - r
        d = np.linalg.norm(n, axis=0)

        weight = np.zeros(len(d))
        if self.mode == 'rectangle':
            weight = np.ones(len(d))
        elif self.mode == 'linear':
            weight = 1 - d / np.max(d)
        elif self.mode == 'gaussian':
            weight = r * stats.norm.pdf(d, 0, r / 3)
        elif self.mode == 'dog':
            g1 = stats.norm.pdf(d, 0, r / 2.5)
            g2 = 0.5*stats.norm.pdf(d, 0, r / 1.5)
            weight = r * (g1 - g2)

        weight[d > r] = 0
        weight *= self.alpha

        return torch.from_numpy(n), torch.from_numpy(weight.astype(np.float32))

    def save_screenshot(self, value, dt):
        self.imdata.save('somnia_{}.png'.format(time.time()))

    def update_radius(self, value, dt):
        self.radius = int(1.0 * (value+1))
        if self.radius < 1:
            self.radius = 1
        self.changed = True

    def update_alpha(self, value, dt):
        self.alpha = value / 127
        if self.alpha < 0:
            self.alpha = 0
        self.changed = True

    def update_tiling(self, value, dt):
        if value < 0:
            value = 0
        self.tiling = value // 16 + 1
        self.k_height = self.height // self.tiling
        self.k_width = self.width // self.tiling

    def change_neighborhood(self, mode, value, dt):
        self.mode = mode
        self.changed = True


class SOMNIADataset(Dataset):
    def __init__(self, source):
        image = Image.open(source)
        self.data = torch.Tensor(list(image.getdata())).view(-1, 3, 1, 1)
        self.cursor = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item].view(3, 1, 1)

    def __next__(self):
        if self.cursor >= len(self):
            self.cursor = 0
        sample = self[self.cursor]
        self.cursor += 1
        return sample


if __name__ == '__main__':
    dataset = SOMNIADataset(args.source)
    loader = DataLoader(dataset, shuffle=args.shuffle)
    somnia = SOMNIA(loader, width=args.width, height=args.height,
                    alpha=args.alpha, radius=args.radius, mode=args.mode,
                    wrap=(args.wrap_y, args.wrap_x), tiling=args.tiling)

    if args.lpd8:
        controller = LPD8Controller()
        try:
            controller.open()
            controller.set_knob_callback(0, somnia.update_radius)
            controller.set_knob_callback(1, somnia.update_alpha)
            controller.set_knob_callback(2, somnia.update_tiling)
            controller.set_pad_down_callback(0, somnia.save_screenshot)
            controller.set_pad_down_callback(4, partial(somnia.change_neighborhood,
                                                        'rectangle'))
            controller.set_pad_down_callback(5, partial(somnia.change_neighborhood,
                                                        'linear'))
            controller.set_pad_down_callback(6, partial(somnia.change_neighborhood,
                                                        'gaussian'))
            controller.set_pad_down_callback(7, partial(somnia.change_neighborhood,
                                                        'dog'))
        except OSError:
            print('LPD8 controller not found')

    window = pyglet.window.Window(somnia.width, somnia.height, vsync=False,
                                  fullscreen=False)

    @window.event
    def on_draw():
        window.clear()
        somnia.imdata.blit(0, 0, 0)

    @window.event
    def on_key_press(key, modifiers):
        if key == pyglet.window.key.RIGHT:
            somnia.update_radius(somnia.radius + 1, None)
        elif key == pyglet.window.key.LEFT:
            somnia.update_radius(somnia.radius - 2, None)
        elif key == pyglet.window.key.UP:
            somnia.update_alpha(somnia.alpha*127 + 1, None)
        elif key == pyglet.window.key.DOWN:
            somnia.update_alpha(somnia.alpha*127 - 1, None)
        elif key == pyglet.window.key.BRACKETRIGHT:
            somnia.update_tiling((somnia.tiling + 1) * 16, None)
        elif key == pyglet.window.key.BRACKETLEFT:
            somnia.update_tiling((somnia.tiling - 2) * 16, None)
        elif key == pyglet.window.key._1:
            somnia.change_neighborhood('rectangle', None, None)
        elif key == pyglet.window.key._2:
            somnia.change_neighborhood('linear', None, None)
        elif key == pyglet.window.key._3:
            somnia.change_neighborhood('gaussian', None, None)
        elif key == pyglet.window.key._4:
            somnia.change_neighborhood('dog', None, None)
        elif key == pyglet.window.key.SPACE:
            somnia.save_screenshot(None, None)

    pyglet.clock.schedule(somnia.update)
    pyglet.app.run()
