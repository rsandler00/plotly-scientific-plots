### Plotly misc

import plotly.offline as pyo
import plotly
import collections.abc

def plotOut(fig, plot=True):
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
    plots_dict = _iterateOverNestedList(jsonifyFigure, plots)

    return plots_dict

def jsonifyFigure(fig):
    toDict = lambda v: dict(v) if type(v) in [plotly.graph_objs.graph_objs.PlotlyDict] else v

    fig = dict(fig)
    fig['layout'] = dict(fig['layout'])
    fig['layout'] = {key:toDict(val) for (key,val) in fig['layout'].items()}
    fig['data'] = [dict(d) for d in fig['data']]
    fig['data'] = [{key:toDict(val) for (key,val) in d.items()} for d in fig['data']]
    fig['frames'] = dict(fig['frames'])
    return fig


def _iterateOverDicts(ob, func):
    """
    This function iterates a function over a nested dict/list
    see https://stackoverflow.com/questions/32935232/python-apply-function-to-values-in-nested-dictionary
    """
    for k, v in ob.items():
        if isinstance(v, collections.abc.Mapping):
            _iterateOverDicts(v, func)
        elif isinstance(v, list):
            ob[k] = map(func, v)
        else:
            ob[k] = func(v)

def _iterateOverNestedList(fun, data):
    if isinstance(data, list):
        return [_iterateOverNestedList(fun, elem) for elem in data]
    else:
        return fun(data)