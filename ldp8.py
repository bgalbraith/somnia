from rtmidi.midiutil import open_midiinput

KNOB = 176
PAD_DOWN = 144
PAD_UP = 128
PAD_ID_OFFSET = 36


class LPD8Controller(object):
    """
    The AKAI LPD8 has eight velocity sensitive pads and eight knobs
    """
    def __init__(self):
        self._controller = None
        self._knobs = [lambda x, y: None for _ in range(8)]
        self._pads = [{'up': lambda x, y: None, 'down': lambda x, y: None}
                      for _ in range(8)]

    def open(self):
        controller, port = open_midiinput('LPD8')
        controller.set_callback(self.handle_input)
        self._controller = controller

    def handle_input(self, message, data):
        action, dt = message

        if action[0] == KNOB:
            self._knobs[action[1]-1](action[2], dt)
        elif action[0] == PAD_DOWN:
            self._pads[action[1]-PAD_ID_OFFSET]['down'](action[2], dt)
        elif action[0] == PAD_UP:
            self._pads[action[1]-PAD_ID_OFFSET]['up'](action[2], dt)
        else:
            print(action)

    def set_knob_callback(self, knob_id, callback):
        self._knobs[knob_id] = callback

    def set_pad_up_callback(self, pad_id, callback):
        self._pads[pad_id]['up'] = callback

    def set_pad_down_callback(self, pad_id, callback):
        self._pads[pad_id]['down'] = callback
