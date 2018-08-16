### Plotly misc

import plotly.offline as pyo
import plotly
import collections.abc
import numpy as np

def plotOut(fig, plot=True, mode='auto'):
    """ Standard code snippet to decide whether to return plotly fig object or plot """
    if plot:
        plotfunc = pyo.iplot if in_notebook() else pyo.plot
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
    if 'json_format' in fig:    # input already in json format
        fig_dict = fig
    else:                       # input needs to be converted to json format
        fig_dict = {
            'data': fig._data,
            'layout': fig._layout,
            'json_format': True,
        }
    fig_dict = _iterateOverDicts(fig_dict, _tolist)

    return fig_dict


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