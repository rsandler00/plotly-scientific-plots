import numpy as np

#plotting
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import colorlover as cl

# internal files
from plotly_scientific_plots.plotly_misc import in_notebook, plotOut

## Plotly plot subcomponents
def makeEventLines(times,   # 1d array of timestamps
                   orientation='v', # 'v' or 'h'
                   labels=None, # optional 1d numeric array of event type indices
                   labelmap=None, # optional labelmap in list form, eg: ['car', 'truck']
                   rng=None     # optional filter range for timestamps
                   ):

    # preprocess data
    times = np.array(times)

    # filter labels for relevant timerange
    if rng is not None:
        filt_indxs = (times >= rng[0]) & (times < rng[1])
        times = times[filt_indxs]
        if labels is not None:
            labels = labels[filt_indxs]

    n_events = times.size

    # generate colors
    if labels is not None:
        if labelmap is None:
            labelmap =  [str(x) for x in np.unique(labels)]
        n_types = len(labelmap)
        cols = cl.scales[str(max(3, n_types))]['qual']['Set2']
    else:
        labels = np.zeros(n_events)
        cols = ['red']

    # create line shapes
    lines = []
    for i in range(n_events):
        lines += [abs_line(times[i], orientation, color=cols[labels[i]], name=labelmap[labels[i]])]

    return lines


def abs_line(position, orientation, color='red', width=3, annotation=None, name='abs_line', dash='solid', opacity=0):
    '''
    Creates an absolute line which appears irregardless of panning.
    To use, add output to layout shapes:
        layout.shapes = [hline(line)]
    '''

    if orientation == 'v':
        big = 'x'
        lil = 'y'
    elif orientation == 'h':
        big = 'y'
        lil = 'x'
    else:
        print('ERROR: orientation must be either "v" or "h"')

    shape = {
        'type': 'line',
        big+'ref': big,
        lil+'ref': 'paper',
        big+'0': position,
        big+'1': position,
        lil+'0': 0,
        lil+'1': 1,
        'opacity': opacity,
        'line': {
            'color': color,
            'width': width,
            'dash': dash
        },
        'path': name,
    }

    return shape


def vline(position, **params):
    ''' Creates vertical line shape'''
    return abs_line(position, orientation='v', **params)


def hline(position, **params):
    ''' Creates horizontal line shape'''
    out = abs_line(position, orientation='h', **params)
    return out


def addRect(start, end, orientation='V', color='#ff0000', opacity=0.1, name='rect'):
    ''' This makes a rectangluar background from start til end with shaded color. Useful for highlighting things in plots'''
    if orientation == 'V':
        xref = 'x'
        yref = 'paper'
        x0 = start
        x1 = end
        y0 = 0
        y1 = 1
    elif orientation == 'H':
        yref = 'y'
        xref = 'paper'
        y0 = start
        y1 = end
        x0 = 0
        x1 = 1
    else:
        raise ValueError('Orientation must be either "H" or "V". You input %s' % str(orientation))

    return {
        'type': 'rect',
        'xref': xref,
        'yref': yref,
        'x0': x0,
        'y0': y0,
        'x1': x1,
        'y1': y1,
        'fillcolor': color,
        'opacity': opacity,
        'line': {
            'width': 0,
        },
        'path': str(name)
    }


def _plotSubplots(trace_array,
                  vert_spacing=.1,
                  title = '',
                  ylbl='',  # currently buggy
                  xlbl='',
                  sp_titles=None, # 2d np array of strings for subplot titles
                  plot=True
                  ):
    '''
    Internal function to make subplots based on passed traces, which are in a 2d np array
    '''
    n_rows, n_cols = trace_array.shape

    fig = make_subplots(rows=n_rows,
                        cols=n_cols,
                        shared_xaxes=True,
                        vertical_spacing=vert_spacing,
                        subplot_titles=sp_titles.flatten().tolist(),
                        )

    for r in range(n_rows):
        for c in range(n_cols):
            [fig.append_trace(trace, r+1, c+1) for trace in trace_array[r,c]]


    fig.layout.title = title
    # fig.layout.xaxis = {'title': xlbl}    # this ruins the shared x-axis for some reason
    #fig.layout.yaxis = {'title': ylbl}
    fig.layout.showlegend = True

    return plotOut(fig, plot)


def labelsShading(labels,   # array of labels
                  index=None,   # optional index to plot rect along index scale instead of bins
                  colorscale='Set2',  # colorlover colorscale
                  opacity=.2,
                  exclude_cats=[],  # excludes this category from shading
                  ):
    '''
    Returns plotly rectangle which shade whenever a time-series vector of predicted labels is a given label.
    EX:
        labelsShading(labels, index=x, exclude_cats=[0])
    '''

    cats, indxs, cats_reindexed = np.unique(labels, return_index=True, return_inverse=True)
    n_vals = len(labels)
    n_cats = len(cats)
    cols = cl.scales[str(max(n_cats, 3))]['qual'][colorscale]

    if index is None:
        index  = np.arange(0,n_vals)

    if not isinstance(exclude_cats, list):
        exclude_cats = [exclude_cats]

    # get list of transition points in time-series when label changes
    transition_points, transition_values = labelsToTransitions(cats_reindexed)
    box_cats = cats[transition_values]

    shapes = []
    for i in range(len(transition_points) - 1):
        if box_cats[i] in exclude_cats:
            pass
        else:
            start = index[transition_points[i]]
            end = index[transition_points[i + 1] - 1]
            shapes.append(addRect(start, end, color=cols[transition_values[i]-1], name=box_cats[i], opacity=opacity))

    # Add legend
    label_annots = []
    for i, cat in enumerate(cats):
        if cat not in exclude_cats:
            label_annots += [
                dict(
                    xref='paper',
                    yref='paper',
                    text=cat,
                    showarrow=False,
                    font=dict(
                        size=14,
                        color='#000000'
                    ),
                    align='left',
                    x=20,
                    y=-30,
                    bgcolor=cols[i],
                    opacity=opacity)
            ]

    return shapes, label_annots


def labelsToTransitions(labels,
                        mode='all', # 'all', 'starts', 'stops'
                        ):
    '''
    gets transition points from labels
    Returns:
        transition_points: list of transition points
        transition_values: list same size as transition_points of actual values of labels at those points
    '''
    if mode == 'all':
        diff = np.diff(labels) != 0
    transition_points = list(np.where(diff)[0]+1)
    transition_points = [0] + transition_points + [len(diff)]
    transition_values = labels[transition_points]
    return transition_points, transition_values
