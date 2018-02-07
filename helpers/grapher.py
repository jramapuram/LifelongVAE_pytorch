import torch
import numpy as np

from visdom import Visdom
from torch.autograd import Variable


def to_data(tensor_or_var):
    '''simply returns the data'''
    if type(tensor_or_var) is Variable:
        return tensor_or_var.data

    return tensor_or_var


class Grapher(object):
    ''' A helper class to assist with plotting to visdom '''
    def __init__(self, env, server, port=8097):
        self.vis = Visdom(server=server,
                          port=port,
                          env=env)
        self.env = env
        self.param_map = self._init_map()
        self.function_map = {
            'line': self._plot_line,
            'imgs': self._plot_imgs,
            'img': self._plot_img,
            'hist': self._plot_hist,
            'video': self._plot_video
        }

        # this is persisted through the lifespan of the object
        # it contains the window objects
        self.registered_lines = {}

    def save(self):
        self.vis.save([self.env])

    def _init_map(self):
        ''' Internal member to return a map of lists '''
        return {
            'line': [],
            'imgs': [],
            'img': [],
            'video': [],
            'hist': []
        }

    def clear(self):
        '''Helper to clear and reset the internal map'''
        if hasattr(self, 'param_map'):
            self.param_map.clear()

        self.param_map = self._init_map()

    def _plot_img(self, img_list):
        for img_map in img_list:
            for key, value in img_map.items():
                self.vis.image(to_data(value).cpu().numpy(),
                               opts=dict(title=key),
                               win=key)

    def _plot_imgs(self, imgs_list):
        for imgs_map in imgs_list:
            for key, value in imgs_map.items():
                self.vis.images(to_data(value).cpu().numpy(),
                                opts=dict(title=key),
                                win=key)

    def _plot_line(self, line_list):
        for line_map in line_list:
            for key, value in line_map.items():
                x = np.asarray(value[0])  # time-point
                y = np.asarray(value[1])  # value
                if len(y.shape) < 1:
                    y = np.expand_dims(y, -1)

                if len(x.shape) < 1:
                    x = np.expand_dims(x, -1)

                if key not in self.registered_lines:
                    self.registered_lines[key] = self.vis.line(
                        Y=y, X=x,
                        opts=dict(title=key),
                        win=key
                    )
                else:
                    self.vis.line(Y=y, X=x,
                                  opts=dict(title=key),
                                  win=self.registered_lines[key],
                                  update='append')

    def _plot_hist(self, hist_list):
        for hist_map in hist_list:
            for key, value in hist_map.items():
                numbins = value[0]
                hist_value = value[1]
                self.vis.histogram(hist_value,
                                   opts=dict(title=key, numbins=numbins),
                                   win=key)

    def _plot_video(self, video_list):
        for video_map in video_list:
            for key, value in video_map.item():
                assert isinstance(value, torch.Tensor), "files not supported"
                self.vis.video(tensor=to_data(value),
                               opts=dict(title=key),
                               win=key)

    def register(self, param_map, plot_types, override=True):
        ''' submit bulk map here, see register_single for detail '''
        assert len(param_map) == len(plot_types)
        if type(override) != list:
            override = [override] * len(param_map)

        for pm, pt, o in zip(param_map, plot_types, override):
            self.register_single(pm, pt, o)

    def _find_and_append(self, param_map, plot_type):
        assert plot_type == 'line', "only line append supported currently"
        exists = False
        for i in range(len(self.param_map[plot_type])):
            list_item = self.param_map[plot_type]
            for key, value in param_map.items():
                for j in range(len(list_item)):
                    if key in list_item[j]:
                        list_item[j][key][0].extend(value[0])
                        list_item[j][key][1].extend(value[1])
                        exists = True

        if not exists:
            self.param_map[plot_type].append(param_map)

    def _find_and_replace(self, param_map, plot_type):
        exists = False
        for i in range(len(self.param_map[plot_type])):
            list_item = self.param_map[plot_type]
            for key, value in param_map.items():
                for j in range(len(list_item)):
                    if key in list_item[j]:
                        list_item[j][key] = value
                        exists = True

        if not exists:
            self.param_map[plot_type].append(param_map)

    def register_single(self, param_map, plot_type='line',
                        append=False, override=True):
        ''' register a single plot which will be added to the current map
            eg: register({'title': value}, 'line')

            plot_type: 'line', 'hist', 'imgs', 'img', 'video'
            override : if True then overwrite an item if it exists
            append   : if True appends to the line. This is mainly useful
                       useful if you are extending a line before show()

            Note: you can't override and append
        '''
        assert len(param_map) == 1, "only one register per call"
        assert not(override is True and append is True), "cant override and append"

        plot_type = plot_type.lower().strip()
        assert plot_type == 'line' \
            or plot_type == 'hist' \
            or plot_type == 'imgs' \
            or plot_type == 'img' \
            or plot_type == 'video'

        if append:
            self._find_and_append(param_map, plot_type)

        if override:
            self._find_and_replace(param_map, plot_type)

    def _check_exists(self, plot_type, param_map):
        for key, _ in param_map.items():  # {'name', value}
            for list_item in self.param_map[plot_type]:  # [{'name': value}, {'name2': value2}]
                return key not in list_item

    def show(self, clear=True):
        ''' This helper is called to actually push the data to visdom'''
        for key, value_list in self.param_map.items():
            self.function_map[key](value_list)

        if clear:  # helper to clear the plot map
            self._init_map()
