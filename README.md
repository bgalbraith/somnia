## SOMINA: Self-Organizing Maps as Neural Interactive Art

How to use:

`python somnia.py --source <image>`

There are a variety of other command line arguments you can pass in to set the
initial properties of the network.

This will launch a window and the model will begin to train. To save a copy of
the state of the network, press the space bar. You can change the neighborhood
function properties in the following ways:

* window function - constant (1), linear (2), Gaussian (3), difference of Gaussians (4)
* radius - increase (right), decrease (left)
* learning rate - increase (up), decrease (down)
* tiling effect - increase (right brace), decrease (left brace)
