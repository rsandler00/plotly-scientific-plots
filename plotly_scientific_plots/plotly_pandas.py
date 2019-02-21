import numpy as np
import pandas as pd

#plotting
import plotly.graph_objs as go

import colorlover as cl

# internal files
from plotly_scientific_plots.plotly_misc import in_notebook, plotOut
from plotly_scientific_plots.plotly_plot_tools import addRect, _plotSubplots


def plotDF( df,             # pandas DF
            title='',       # title of plot
            ylbl='',       # ylabel
            xlbl=None,        # if None, uses df.index.name
            linemode='lines',   # 'lines'/'markers'/'lines+markers'
            cat_col = None, # if name, then shades BG according to the label
            opacity = .7,   # transparaency of lines. [0.0, 1.0]
            plot=True,      # 1/0 whether we want to plot each of the individual lines
        ):
    """
    This plots a pandas DF.
    NOTE: see also plotly's cufflinks package which makes pnadas plotting super easy!
        cf.go_offline()
        df.iplot(kind='scatter')
    """

    nbins, ncols = df.shape

    # convert cat columns to numeric columns
    for col in df.columns:
        if df[col].dtype.name=='category':
            df[col] = df[col].cat.codes

    # make line colors
    colors = cl.scales[str(max(3, ncols))]['qual']['Set3']
    tcols = ['rgba%s,%.2f)' % (c[3:-1], opacity) for c in colors]

    traces = [go.Scatter(
                x=df.index,
                y=df[col].values,
                name=col,
                mode=linemode,
                line={"color": tcols[i]}
                )
              for i, col in enumerate(df.columns)
              ]

    if xlbl is None:
        xlbl = df.index.name

    layout = go.Layout(title=title,
                       xaxis={'title': xlbl},
                       yaxis={'title': ylbl},
                       showlegend=True,
                       )

    # shade background based on label
    if cat_col is not None:
        cats, cats_reindexed = np.unique(df[cat_col], return_inverse=True)
        n_cats = len(cats)
        cols = cl.scales[str(max(n_cats, 3))]['qual']['Set3']
        # set_trace()
        transition_points = list(np.where(np.diff(cats_reindexed) != 0)[0]) + [df.shape[0] - 1]
        shapes = []
        for i in range(len(transition_points) - 1):
            start = df.iloc[transition_points[i]].name
            end = df.iloc[transition_points[i + 1] - 1].name
            color = cols[cats_reindexed[transition_points[i]]]
            shapes.append(addRect(start, end, color=color))
        layout.shapes = shapes
        # TODO: add legend

    fig = go.Figure(data=traces, layout=layout)

    return plotOut(fig, plot)


def plotDF_Subplots(df,
                    subplot_col_list,  # list of column lists. Ex: [['col1'], ['col2', 'col3'], ['col4']]
                    linemode='lines',   # 'lines'/'markers'/'lines+markers'
                    **kwargs
                    ):
    '''
    Plots specified DF columns in vertically stacked subplots w/ shared x-axis rather than all on top of each other
    '''

    n_subplots = len(subplot_col_list)

    trace_array = np.empty((n_subplots, 1), dtype='O')

    for i, sp_cols in enumerate(subplot_col_list):
        sp_traces = []
        for col in sp_cols:
            sp_traces += [go.Scatter(
                x=df.index,
                y=df[col].values,
                name=col,
                mode=linemode
            )]
        trace_array[i,0] = sp_traces

    return _plotSubplots(trace_array, **kwargs)





