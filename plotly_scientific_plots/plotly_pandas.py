import numpy as np
import pandas as pd

#plotting
import plotly.graph_objs as go

import colorlover as cl

# internal files
from plotly_scientific_plots.plotly_misc import in_notebook, plotOut
from plotly_scientific_plots.plot_subcomponents import addRect, _plotSubplots, labelsShading
from plotly_scientific_plots.misc_computational_tools import norm_mat

def plotDF( df,             # pandas DF
            title='',       # title of plot
            ylbl='',       # ylabel
            xlbl=None,        # if None, uses df.index.name
            linemode='lines',   # 'lines'/'markers'/'lines+markers'
            cat_col = None, # if name, then shades BG according to the label
            opacity = .7,   # transparaency of lines. [0.0, 1.0]
            norm = None,    # None or input to norm_mat
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

    # normalize columns
    if norm is not None:
        for col in df.columns:
            df[col] = norm_mat(df[col].values, method='zscore')

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
        layout.shapes = labelsShading(df[cat_col].values)

    fig = go.Figure(data=traces, layout=layout)

    return plotOut(fig, plot)


def plotDF_Subplots(df,
                    subplot_col_list=None,  # list of column lists. Ex: [['col1'], ['col2', 'col3'], ['col4']]
                    linemode='lines',   # 'lines'/'markers'/'lines+markers'
                    sp_titles=None,     # list of subplot titles. If None then uses df column names
                    opacity=.7,         # line opacity
                    **kwargs
                    ):
    '''
    Plots specified DF columns in vertically stacked subplots w/ shared x-axis rather than all on top of each other
    '''

    if subplot_col_list is None:
        subplot_col_list = [[x] for x in df.columns]

    n_subplots = len(subplot_col_list)

    trace_array = np.empty((n_subplots, 1), dtype='O')

    for i, sp_cols in enumerate(subplot_col_list):
        sp_traces = []
        for col in sp_cols:
            sp_traces += [go.Scatter(
                x=df.index,
                y=df[col].values,
                name=col,
                mode=linemode,
                opacity=opacity
            )]
        trace_array[i,0] = sp_traces

    if sp_titles is None:
        sp_titles = np.transpose([[' + '.join(cols).title() for cols in subplot_col_list]])
    else:
        sp_titles = np.array([sp_titles])

    return _plotSubplots(trace_array, sp_titles=sp_titles, **kwargs)





