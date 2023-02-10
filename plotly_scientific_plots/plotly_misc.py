### Plotly misc

import plotly.offline as pyo
import plotly
import plotly.graph_objs as go
import collections.abc
import numpy as np
import colorlover as cl
from typing import Union, Generic, TypeVar

Shape = TypeVar("Shape")
DType = TypeVar("DType")

class Array(Generic[Shape, DType]):
    """
    Use this to type-annotate numpy arrays, e.g.
        image: Array['H,W,3', np.uint8]
        xy_points: Array['N,2', float]
        nd_mask: Array['...', bool]
    """
    pass

def placeholder_figure(id):
    ''' Creates placholder figure for Dash callbacks '''
    return go.Figure(data=None, layout={'meta': [id]})

def plotOut(fig,
            plot=True,
            mode='auto', # 'auto'/'tab'. If 'tab' forces to plot in seperate tab even if in notebook
            filename='temp_plot.html',
            ):
    """ Standard code snippet to decide whether to return plotly fig object or plot """
    if plot:
        if in_notebook() and mode == 'auto':
            plotfunc = pyo.iplot
        else:
            plotfunc = lambda x: pyo.plot(x, filename=filename)
        plotfunc(fig)
        return fig
    else:
        return fig


def in_notebook():
    """
    Returns ``True`` if the module is running in IPython kernel,
    ``False`` if in IPython shell or other Python shell.
    """
    try:
        return get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
    except:
        return False


def jsonify(plots):
    """
    Completely convert all elements in a list of plotly plots into a dict
    This is used to pickly object in mutliprocessing
    """
    plots_dict = _iterateOverNestedList(plots, jsonifyFigure)

    return plots_dict


def jsonifyFigure(fig):
    """
    Converts a plotly Figure object into a dict.
    Note that pyo.plot() works the same w.r.t. a Figure object or a dict...
    """
    if isinstance(fig, dict) or 'json_format' in fig:    # input already in json format
        fig_dict = fig
    else:                       # input needs to be converted to json format
        fig_dict = {
            'data': fig._data,
            'layout': fig._layout,
            'json_format': True,
        }
    fig_dict = _iterateOverDicts(fig_dict, _tolist)

    return fig_dict


def jsonify_plotly_figure_dict(fig_dict):
    out = '{'
    for name, fig in fig_dict.items():
        out += f'"{name}": {fig.to_json()}'
    out += '}'
    return out


def _iterateOverDicts(ob, func):
    """
    This function iterates a function over a nested dict/list
    see https://stackoverflow.com/questions/32935232/python-apply-function-to-values-in-nested-dictionary
    """
    assert type(ob) == dict, '**** _iterateOverDicts requires a dict input'
    for k, v in ob.items():                         # assumes ob is a dict
        if isinstance(v, collections.abc.Mapping):  # if v dict, go through fields
            _iterateOverDicts(v, func)
        elif isinstance(v, list):                   # if list, check if elements are lists or dicts
            for i in range(len(v)):
                if isinstance(v[i], collections.abc.Mapping):  # if dict apply func to fields
                    v[i] = _iterateOverDicts(v[i], func)
                else:                               # if list, apply func to elements
                    v[i] = func(v[i])
        else:
            ob[k] = func(v)
    return ob


def _iterateOverNestedList(ob, func):
    if isinstance(ob, list):
        return [_iterateOverNestedList(elem, func) for elem in ob]
    else:
        return func(ob)


def _tolist(arr):
    ''' Converts to list if np array'''
    return arr.tolist() if type(arr)==np.ndarray else arr


def _massageData(y: Union[Array[float, 'n_bins'],
                          Array[float, 'n_sigs, n_bins'],
                          Array[list, 'n_sigs']
                         ],
                 x: Union[None,
                          Array[float, 'n_bins'],
                          Array[float, 'n_sigs, n_bins'],
                          Array[list, 'n_sigs']
                         ]=None,
                 z=None,
                 txt=None,
                 names=None
                 ):
    '''
    This massages input data so that all outputs are in the form of:
        [n_sig, n_bin] np arrays (if signals equal length)
        [n_sig] np array where each entry a full array (if signals unequal length)
    '''

    y = np.atleast_2d(np.array(y))

    if y.dtype == 'O':  # arrays are not equally size
        y = y.flatten()
        equal_size = False
        n_sigs = len(y)
        n_bins = [len(s) for s in y]
    else:
        equal_size = True
        n_sigs, n_bins = y.shape

    x, x_info = _massageDataCorrelate(n_sigs, n_bins, equal_size, True, x)
    z, z_info = _massageDataCorrelate(n_sigs, n_bins, equal_size, False, z)

    if names is None:
        names = ['#%d' %(i) for i in range(n_sigs)]

    info = {
        'n_sigs': n_sigs,
        'n_bins': n_bins if x_info['shared'] else [n_bins] * n_sigs,
        'x_info': x_info,
        'z_info': z_info,
    }

    return y, x, z, names, info


def _massageDataCorrelate(n_sigs, n_bins, equal_size, required, data):
    ''' Checks a correlate of main data (eg checks x/z when y provided '''

    def convertEmptyToNone(inp):
        return None if inp==[] else inp

    data = convertEmptyToNone(data)

    if data is None:
        provided = False
        shared = True
        if required:
            assert equal_size == True, 'If y is unequally sized arrays, a unique x data for each must be provided'
        data = [None]
    else:
        data = np.array(data)
        provided = True
        if data.dtype == 'O':
            assert equal_size == False, 'Y is equally sized, but this data is unequally sized (dtype=="O" means data unequally sized)'
            assert len(data) == n_sigs, 'Provided %d y sigs, but only %d data sigs' % (n_sigs, len(data))
            shared = False
            for i in range(n_sigs):
                assert len(data[i]) == n_bins[i], 'For signal %d, len(y)=%d != len(data)=%d' % (i, n_bins[i], len(data[i]))
        else:
            data = np.atleast_2d(data)
            if 1 in data.shape: # data is a vector
            # PREVIOUSLY WAS: if 1 in data.shape and n_sigs == 1:
            # NOTE: removed & n_sigs==1 on 11/26/19 since if provided x, it can be shared between multiple y's
                if data.shape[1] == 1:  # make sure 1d vectors are shape [N, 1]
                    data = data.T
                shared = True
                assert data.shape[1] == n_bins, 'len(y)=%d != len(data)=%d' % (n_bins, data.shape[1])
            else:
                shared = False
                assert data.shape == (n_sigs, n_bins), 'data is shape %s, but y is shape %s' % (str(data.shape), str([n_sigs, n_bins]))

    info = {'shared': shared, 'provided': provided}

    return data, info


def _getCols(n_cols, set='Dark2'):  # use 'Set3' for max amount of colors
    '''  '''
    if n_cols < 3:
        return cl.scales['3']['qual']['Dark2'][:n_cols]
    elif str(n_cols) in cl.scales and set in cl.scales[str(n_cols)]['qual']:
        return  cl.scales[str(max(3, n_cols))]['qual'][set]
    else:   # if n_cols larger than amount of cols in the set, then cycle from the beginning
        max_set_cols = max([i for i in range(3, min(13,n_cols)) if set in cl.scales[str(i)]['qual']])
        return [cl.scales[str(max_set_cols)]['qual'][set][(i+1) % max_set_cols] for i in range(n_cols)]


def _extend_range(minn, maxx, extend_ratio):
    ''' Extends rng for plotting. Takes care of subtelies of pos/neg #s '''
    span = maxx - minn
    extend_span = span * extend_ratio
    minn -= extend_span
    maxx += extend_span
    return minn, maxx