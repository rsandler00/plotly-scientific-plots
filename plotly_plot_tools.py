import numpy as np
import scipy as sp
import scipy.stats
import sys
import os
import copy
from numpy.lib.stride_tricks import as_strided

#plotting
import plotly.offline as pyo
import plotly.graph_objs as go
import plotly as py
import matplotlib.pyplot as plt
import colorlover as cl
import dash
import dash_core_components as dcc
import dash_html_components as html
import cufflinks as cf

# personal functions
#module_path = os.path.abspath(os.path.join('..'))
#sys.path.append(module_path + '\\Utils')
#from misc import *

###Scientific Plots
def plotHist(data,              # 1D list/np vector of data
            maxData=1000,       #  max # of points to plot above histogram (if too high, it will be slow)
            plot=True,          #1/0. If 0, returns plotly json object, but doesnt plot
            title='Distribution', # plot title
            xlbl='',            # plot label
            rm_outliers = False, #1/0 whether to remove outliers or not
            density = True,		# whether to plot PDF or count
            boxplot = True,     # 1/0 whether to do upper boxplot
            scatter = True,     # 1/0 add upper scatterplot
            diff_tst = 0):        # 1/0. If 1 assumes we checking for a signif difference from 0
    """
    Plots a 1D histogram using plotly.
    Does the binning w/ numpy to make it go way faster than plotly's inherent histogram function

    Usage:
    x = np.random.normal(0,1,(100))
    plotHist(x, title=Normal Distribution', xlbl='values', diff_tst=1)

    :return: NA
    """

    N = len(data)
    data = np.array(data)

    # remove NaNs/Infs
    data = data[~np.isnan(data)]
    data = data[np.isfinite(data)]

    adj, corr_data, outliers, rng, stats = removeOutliers(data, stdbnd=6, percclip=[5, 95], rmv=rm_outliers)

    hy, hx = np.histogram(data, bins=40, density=density, range=rng)
    top = np.max(hy)*1.1
    jitter = .02

    traces = []
    hist = go.Bar(x=hx, y=hy, name='Hist', opacity=.5,
                       marker=dict(color='red',
                                   line=dict(color='black', width=2)))
    traces += [hist]

    # if data too large only plot a subset
    if scatter:
        if N>maxData:
            Np = maxData
            dataToPlot = np.random.choice(data, Np, replace=False)
        else:
            dataToPlot, Np = data, N
        dataPlot = go.Scatter(x=dataToPlot, y=top+np.random.normal(size=Np)*top*jitter, name='data', mode = 'markers',
                         marker = dict(color='black', size = 2), hoverinfo='x+name')
        traces += [dataPlot]

    #boxplot
    if boxplot:
        bp = boxPlot(stats['med'], np.percentile(data, [25, 75]), rng, mean=stats['mean'],
                      horiz=True, offset=top * 1.2, plot=False, col='red', showleg=True)
        traces += bp

    if diff_tst:
        vertline = go.Scatter(x=[0,0], y=[0,top*1.1], name='x=0', showlegend=1, line=dict(color='black', width=2, dash='dot'))
        traces += [vertline]
        _, Pt = sp.stats.ttest_1samp(data, 0)
        _, Pw = sp.stats.wilcoxon(data)
        title += ' P_t=%.2f. P_w=%.2f' % (Pt, Pw)

    ylbl = 'Probability Density' if density else 'Count'

    fig = go.Figure(data=traces,
                   layout={'title':title,
                            'yaxis':{'title': ylbl},
                            'xaxis':{'title': xlbl, 'range': [rng[0]*.9,rng[1]*1.1]},
                            'bargap':0,
                            'hovermode': 'closest',
                           }
                    )
    if plot:
        plotfunc = pyo.iplot if in_notebook() else pyo.plot
        plotfunc(fig)
    else:
        return fig


def plot2Hists(x1,              # data of 1st histogram
               x2,              # data of 2nd histogram
               names=['A','B'], # legend names of x1, x2 (ex: ['A','B']
               maxData=500,     # max # of points to plot above histogram (if too high, it will be slow)
               normHist=True,   # 1/0. if 1, norms the histogram to a PDF
               samebins=True,   # whether both hists should have same edges
               numbins=40,      # # bins in histogram
               title='Data Distribution', # title of plot
               rm_outliers = False, #1/0 whether to remove outliers or not
               KS=False,        # whether to do 2 sample KS test for different distributions
               MW=False,        # whether to display the Mann-Whitney/Ranksum test for difference of distributions in title
               T=False,         # as MW, but for ttest
               alt='two-sided', # one-sided or two-sided hypothesis testing. See scipy for options
               bp=True,         # whether to add barplot above histograms
               plot=True):      # 1/0. If 0, returns plotly json object, but doesnt plot
    """
    Plots two 1D histograms using plotly.
    Does the binning w/ numpy to make it go way faster than plotly's inherent histogram function

    Usage:

 
    """

    x1=np.array(x1)
    x2=np.array(x2)
    N1, N2 = len(x1), len(x2)

    # Remove NaNs
    x1 = x1[~np.isnan(x1)]
    x2 = x2[~np.isnan(x2)]

    # remove outliers & get basic stats
    adj1, corr_data1, outliers1, rng1, stats1 = removeOutliers(x1, stdbnd=6, percclip=[5, 95], rmv=rm_outliers)
    adj2, corr_data2, outliers2, rng2, stats2 = removeOutliers(x2, stdbnd=6, percclip=[5, 95], rmv=rm_outliers)

    if samebins:
        jointrng = [min(rng1[0], rng2[0]), max(rng1[1], rng2[1])]
        bins=np.linspace(jointrng[0], jointrng[1], numbins)
    else:
        bins=numbins

    hy1, hx1 = np.histogram(x1, bins=bins, density=normHist, range=rng1)
    hy2, hx2 = np.histogram(x2, bins=bins, density=normHist, range=rng2)

    top = np.max(np.hstack((hy1,hy2))) * 1.1

    # hist plots
    traces=[]
    hist1 = go.Bar(x=hx1, y=hy1, name=names[0], legendgroup = names[0], opacity=.5,
                    marker=dict(color='red',
                              line=dict(color='black', width=2)))
    hist2 = go.Bar(x=hx2, y=hy2, name=names[1], legendgroup = names[1], opacity=.5,
                  marker=dict(color='blue',
                              line=dict(color='black', width=2)))
    traces += [hist1, hist2]

    # data plots
    if N1 > maxData:    # if data too large only plot a subset
        Np = maxData
        dataToPlot = np.random.choice(x1, Np, replace=False)
    else:
        dataToPlot, Np = x1, N1
    dataPlot1 = go.Scatter(x=dataToPlot, y=top*1.2 + np.random.normal(size=Np)*top*.03, mode='markers',
                            marker=dict(size=2, color = 'red'), hoverinfo='x+name',
                            name=names[0], legendgroup=names[0], showlegend=False)
    if N2 > maxData:    # if data too large only plot a subset
        Np = maxData
        dataToPlot = np.random.choice(x2, Np, replace=False)
    else:
        dataToPlot, Np = x2, N1
    dataPlot2 = go.Scatter(x=dataToPlot, y=top + np.random.normal(size=Np)*top*.03, mode='markers',
                            marker=dict(size=2, color = 'blue'), hoverinfo='x+name',
                            name=names[1], legendgroup=names[1], showlegend=False)
    traces += [dataPlot1, dataPlot2]

    # Boxplots
    if bp:
        bp1 = boxPlot(stats1['med'], np.percentile(x1, [25,75]), rng1, mean=stats1['mean'],
                    name=names[0], horiz=True, offset=top*1.3, legendGroup=names[0], plot=False, col='red')
        bp2 = boxPlot(stats2['med'], np.percentile(x2, [25, 75]), rng2, mean=stats2['mean'],
                      name=names[1], horiz=True, offset=top * 1.1, legendGroup=names[1], plot=False, col='blue')
        traces = traces + bp1 + bp2

    # Stat testing
    if MW:
        stat, p_MW = sp.stats.mannwhitneyu(x1, x2, alternative=alt)
        title += ' P_MW=%.3f' % (p_MW)
    if T:
        stat, p_T = sp.stats.ttest_ind(x1, x2, equal_var=True, nan_policy='omit')
        title += ' P_T=%.3f' % (p_T)
    if KS:
        stat, p_KS = sp.stats.ks_2samp(x1, x2)
        title += ' P_KS=%.3f' % (p_KS)

    plotrng = [min(rng1[0], rng2[0])*.9, min(rng1[1], rng2[1])*1.1]
    ylbl = 'Denisty' if normHist else 'Count'
    fig = go.Figure(data=traces,
                    layout={'title': title,
                            'yaxis': {'title': ylbl},
                            'xaxis': {'range': plotrng},
                            'barmode': 'overlay',
                            'bargap': 0,
                            'hovermode': 'closest',
                            }
                    )

    if plot:
        plotfunc = pyo.iplot if in_notebook() else pyo.plot
        plotfunc(fig)
    else:
        return fig


def plotPolar(data,         # N-d list/numpy array
              names=None,   # names of cols in data (ex:['A', 'B']
              scatter= True, # whether to do polar scatter plot. Only works if N=1
              maxData=1000, # max # of points to plot above histogram (if too high, it will be slow)
              hist = True,  # 1/0 whether to plot histogram of points
              numbins=40,   # bins in histogram
              normHist=True,# whether to normalize histogram
              title='Polar Distribution',   # title of plot
              plot=True):   # 1/0. If 0, returns plotly json object, but doesnt plot
    """
    This plots a polar plot of data in plotly
    
    Usage:
    x1 = np.random.uniform(-np.pi, np.pi, (100))
    x2 = np.random.uniform(-np.pi, np.pi, (200))
    plotPolar([x1,x2], names=['A', 'B'], numbins=50)
    """

    ## Basic formatting
    if type(data) != np.ndarray:  data = np.array(data)

    if np.issubdtype(data.dtype, np.number):   #given an np array
        data = np.atleast_2d(data)
        N, Lx = data.shape
        Lx = np.matlib.repmat(Lx, 1, N)
    else: #given a data array
        N = len(data)
        Lx = [len(l) for l in data]

    if names is None:
        names = [str(i + 1) for i in range(N)]

    # make sure all data in radians
    [print('All data must be within +-pi') for col in data if (np.min(col)<-np.pi) or (np.max(col)>np.pi)]

    if N>1:
        lg = names
        showleg = True
        cols = cl.scales[str(N)]['qual']['Set1']
    else:
        lg=[None]
        showleg = False
        cols=['blue']

    # scale markersize
    Lxp = np.min([max(Lx), maxData])
    if Lxp > 5000:
        markersize = 1
    elif Lxp > 2000:
        markersize = 2
    elif Lxp > 1000:
        markersize = 3
    elif Lxp > 200:
        markersize = 4
    elif Lxp > 80:
        markersize = 5
    elif Lxp > 25:
        markersize = 7
    else:
        markersize = 9

    traces = []

    ## Histogram
    if hist:
        hy, hx = zip(*[np.histogram(col, bins=numbins, density=normHist, range=[-np.pi, np.pi]) for col in data])
        hy = np.array(hy)
        traces += [go.Scatter(t=hx[n]/np.pi*180, r=hy[n], name=names[n], mode='lines',
                              line={'width': 2, 'color':cols[n]}, hovertext=names[n], hoverinfo='name+r+t')
                    for n in range(N)]
        top = np.max(hy.flatten()) * 1.2
    else:
        top = 1

    ## Scatter
    if scatter and N==1:
        jitter = .05
        # if data too large only plot a subset
        if Lx[0,0] > maxData:
            Np = maxData
            dataToPlot = np.random.choice(data[0], Np, replace=False)
        else:
            dataToPlot, Np = data[0], Lx[0,0]
        traces += [go.Scatter(r = top+np.random.normal(size=Np)*top*jitter, t = data[0]/np.pi*180,
                        mode='markers', name=names[0] + ' scatter', marker={'size': markersize, 'color':cols[n]})]

    ## make fig
    layout = go.Layout(
        title=title,
        showlegend = showleg
    )
    fig = go.Figure(data=traces, layout=layout)
    #pyo.plot(fig)

    if plot:
        plotfunc = pyo.iplot if in_notebook() else pyo.plot
        plotfunc(fig)
    else:
        return fig


def corrPlot(x,                 # 1D data vector or list of 1D dsata vectors
             y,                 # 1D data vector or list of 1D dsata vectors
             z=None,            # optional colors for the lines
             names=None,        # names of x, y (ex:['A', 'B']
             maxdata=2000,      # max # of points to plot above histogram (if too high, it will be slow)
             addCorr=True,      # whether to add correlation statistics into plot (R2, spearmanR2, Pvals, & y=mx+b)
             addCorrLine=True,     # whether to plot correlation line
             addXYline=False,      # whether to plot y=x line
             text=None,         # whether to add additional text to each point
             plot=True,         # if false, just returns plotly json object
             title='Correlation', # title of plot
             xlbl='',           #
             ylbl=''):          #
    """
    Plots x , y data and their trendline using plotly
    """
    #TODO: remove outliers

    # 1st convert t ndarray


    # 1st convert t ndarray
    if type(x) != np.ndarray:  x = np.array(x)
    if type(y) != np.ndarray:  y = np.array(y)

    # (1) get N
    if np.issubdtype(x.dtype, np.number):  # given an np array
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        N, Lx = x.shape
    else:  # given a data array
        N = len(x)

    # (2) remove NaNs
    tmpx, tmpy = [], []
    for n in range(N):
        bad = np.atleast_2d(np.isnan(x[n]) | np.isnan(y[n]))
        tmpx += [x[n][~bad[0]]]
        tmpy += [y[n][~bad[0]]]
    x = np.array(tmpx)
    y = np.array(tmpy)

    # (3) get Lx
    if np.issubdtype(x.dtype, np.number):  # given an np array
        N, Lx = x.shape
        Lx = np.tile(Lx, N)
    else:  # given a data array
        Lx = [len(l) for l in x]
        Ly = [len(l) for l in y]
        if Lx != Ly: raise ValueError('All x & y vectors must be same length!!!')

    # if data has too many points, remove some for speed
    Iplot = [np.arange(Lx[n]) if Lx[n] < maxdata else np.random.choice(Lx[n], size=maxdata, replace=False) for n in range(N)]

    if names is None:
        names = ['Line ' + str(i) for i in range(N)]
    if isinstance(names, str):
        names = [names]

    traces = []

    # determine scatterpoint colors
    if z is not None:
        assert N==1, 'So far coloring only works w/ 1 data series'
        if type(z) != np.ndarray:  z = np.array(z)
        z = np.atleast_2d(z)
        cols = z
        line_col = ['black']
        lg = [None]
        showleg = False
        showscale = True
        scattertext = ['z=%d' % (i) for i in range(Lx[0])]
    else:
        if N>1:
            lg = names
            showleg = False
            cols = cl.scales[str(max(3, N))]['qual']['Set1']
        else:
            lg=[None]
            showleg = True
            cols=['blue']
        line_col = cols
        showscale = False
        if text is None:
            scattertext = ''
        else:
            scattertext = text

    # scale markersize
    Lxp = np.min([max(Lx),maxdata])
    if Lxp > 5000:
        markersize=1
    elif Lxp >2000:
        markersize=2
    elif Lxp > 1000:
        markersize = 3
    elif Lxp > 200:
        markersize = 4
    elif Lxp > 80:
        markersize = 5
    elif Lxp > 25:
        markersize = 7
    else:
        markersize = 9

    scatPlot = [go.Scatter(x=x[n][Iplot[n]], y=y[n][Iplot[n]], name=names[n], legendgroup=lg[n], mode='markers',
                           opacity=.5, text=scattertext,
                           marker={'size': markersize, 'color':cols[n], 'showscale':showscale, 'colorscale':'Portland'})
                for n in range(N)]
    traces += scatPlot

    annots = []
    if addCorr:
        for n in range(N):
            slope, intercept, R2, p_val, std_err = sp.stats.linregress(x[n], y[n])
            R2sp, p_val_sp = sp.stats.spearmanr(x[n], y[n])
            corrtext = 'Pearson [R2, P]=[%.2f,%.2f] <br> ' \
                       'Spearman [R2, P]=[%.2f,%.2f] <br> ' \
                       'y=%.2fx+%.2f' \
                       % (R2, p_val, R2sp, p_val_sp, slope, intercept)
            #if only 1 data record print stats on graph
            if N==1:
                annots = go.Annotations([go.Annotation(
                    x=0.05,
                    y=0.95,
                    showarrow=False,
                    text=corrtext,
                    xref='paper',
                    yref='paper'
                )])

            if addCorrLine:
                x_rng = [np.min(x[0]), np.max(x[0])]
                dx_rng = x_rng[1] - x_rng[0]
                shift = .03 # shift from edges
                xc = np.array([x_rng[0]+dx_rng*shift, x_rng[1]-dx_rng*shift])
                yc = slope*xc + intercept
                corrline = [go.Scatter(x=xc, y=yc, name=names[n]+' corr', legendgroup=lg[n], showlegend=showleg,
                            mode='lines', line={'color':line_col[n]}, hovertext=corrtext, hoverinfo='name+text')]
                traces += corrline

    if addXYline:
        x_rng = [np.min(x[0]), np.max(x[0])]
        dx_rng = x_rng[1] - x_rng[0]
        shift = .03  # shift from edges
        xc = np.array([x_rng[0] + dx_rng * shift, x_rng[1] - dx_rng * shift])
        xyline = [go.Scatter(x=xc, y=xc, name='X=Y', showlegend=True,
                               mode='lines', line={'color': 'black'})]
        traces += xyline

    showleg = False if N==1 else True

    layout = go.Layout(title=title,
                       annotations=annots,
                       xaxis={'title': xlbl},
                       yaxis={'title': ylbl},
                       hovermode='closest',
                       showlegend = showleg,
                       )
    fig = go.Figure(data=traces, layout=layout)

    if plot:
        plotfunc = pyo.iplot if in_notebook() else pyo.plot
        plotfunc(fig)
    else:
        return fig

def scatterHistoPlot(x,
                     y,
                     title = '2D Density Plot',
                     xlbl = '',
                     ylbl = '',
                     plot = True
                    ):
    """
    ...
    """

    scatter_plot = go.Scatter(
        x=x, y=y, mode='markers', name='points',
        marker=dict(color='rgb(102,0,0)', size=2, opacity=0.4)
    )
    contour_plot = go.Histogram2dcontour(
        x=x, y=y, name='density', ncontours=20,
        colorscale='Hot', reversescale=True, showscale=False
    )
    x_density = go.Histogram(
        x=x, name='x density',
        marker=dict(color='rgb(102,0,0)'),
        yaxis='y2'
    )
    y_density = go.Histogram(
        y=y, name='y density', marker=dict(color='rgb(102,0,0)'),
        xaxis='x2'
    )
    data = [scatter_plot, contour_plot, x_density, y_density]

    scatterplot_ratio = .85    # ratio of figure to be taken by scatterplot vs histograms
    layout = go.Layout(
        title=title,
        showlegend=False,
        autosize=False,
        width=600,
        height=550,
        xaxis=dict(
            title = xlbl,
            domain=[0, scatterplot_ratio],
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            title=ylbl,
            domain=[0, scatterplot_ratio],
            showgrid=False,
            zeroline=False
        ),
        margin=dict(
            t=50
        ),
        hovermode='closest',
        bargap=0,
        xaxis2=dict(
            domain=[scatterplot_ratio, 1],
            showgrid=False,
            zeroline=False
        ),
        yaxis2=dict(
            domain=[scatterplot_ratio, 1],
            showgrid=False,
            zeroline=False
        )
    )

    fig = go.Figure(data=data, layout=layout)

    return plotOut(fig, plot)

def basicBarPlot(data,          # list of #'s
                 names=None,    # xtick labels. Can be numeric or str
                 title='',
                 ylbl='',
                 xlbl='',
                 text=None,
                 orient=None,
                 line=None,     # add line perpendicular to bars (eg to show mean)
                 plot=True):
    """
    Makes a basic bar plot where data is [n,1] list of values. No averaging/etc... For that see barPlot or propBarPlot
    """

    traces = [go.Bar(x=names, y=data, text=text, textposition='auto',
                    marker=dict(
                        color='rgb(158,202,225)',
                        line=dict(
                            color='rgb(8,48,107)',
                            width=1.5),
                    ),
                    opacity=0.6)
              ]

    layout = go.Layout(
            title=title,
            yaxis={'title': ylbl},
            xaxis={'title': xlbl},
            hovermode='closest',
    )
    if line:
        layout.shapes = [hline(line)]

    fig = go.Figure(data=traces, layout=layout)

    if plot:
        plotfunc = pyo.iplot if in_notebook() else pyo.plot
        plotfunc(fig)
    else:
        return fig

    return None

def barPlot(data,           # list of 1D data vectors
            names=None,     # names of data vectors
            maxData=500,    # max # of points to plot above histogram (if too high, it will be slow)
            title=' ',      # title of plot
            ylbl='Mean',    # y-label
            bar=True,       # 1/0. If 0, makes boxplot instead of barplot
            stats=[],       # which stat tests to run, including [ttest, MW, ANOVA, KW] (kruchsal-wallis)
            plot=True):     # 1/0. If 0, just returns fig object
    """
    Makes a custom plotly barplot w/ data on side
    :return:
    """
    # TODO: add outlier removal

    data = np.array(data)
    N = len(data)
    Lx = [len(col) for col in data]
    # remove NaNs
    data = [removeNaN(col) for col in data]

    if names is None:
        names = [str(i + 1) for i in range(N)]

    if N<3:
        cols = cl.scales[str(3)]['qual']['Set1'][0:N]
    elif N<10:
        cols = cl.scales[str(N)]['qual']['Set1']
    else:
        cols=[None]*N

    jitter = .03

    means = [np.mean(col) for col in data]
    meds = [np.median(col) for col in data]
    std = [np.std(col) for col in data]

    traces = []
    if bar:
        bars = [go.Bar(
            x=list(range(N)),
            y=means,
            marker=dict(
                color=cols),
            text=['median= %.4f' % (m) for m in meds],
            name='BAR',
            error_y=dict(
                type='data',
                array=std,
                visible=True
            ),
            showlegend=False
        )]
        traces += bars
    else:
        #implement boxplot
        boxwidth = 50
        quartiles = np.array([np.percentile(data[n], [25, 75]) for n in range(N)])
        minmax=np.array([np.percentile(data[n],[5,95]) for n in range(N)])
        boxs = [boxPlot(meds[n], quartiles[n], minmax[n], mean=means[n], outliers=None, name=names[n], horiz=0, offset=n,
                legendGroup='boxplot', showleg=False, plot=False, col=cols[n], width=boxwidth) for n in range(N)]
        traces += sum(boxs,[])

    # scale markersize
    Lxp = np.max(Lx)
    if Lxp > 5000:
        markersize = 1
    elif Lxp > 2000:
        markersize = 2
    elif Lxp > 1000:
        markersize = 3
    elif Lxp > 200:
        markersize = 4
    elif Lxp > 80:
        markersize = 5
    else:
        markersize = 7

    # reduce length of data for plotting
    data_to_plot = [np.random.choice(col, maxData, replace=False) if len(col) > maxData else col for col in data]

    dataPlot = [go.Scatter(x=i + .5 + np.random.normal(size=len(data_to_plot[i])) * jitter,
                           y=data_to_plot[i],
                           mode='markers',
                           marker=dict(size=markersize, color=cols[i]),
                           name=names[i])
                for i in range(N)]
    traces += dataPlot

    xaxis = go.XAxis(
        # title="",
        showgrid=True,
        showline=True,
        ticks="",
        showticklabels=True,
        linewidth=2,
        ticktext=names,
        tickvals=list(range(N)),
        tickfont=dict(size=18)
    )

    # if data has huge outliers, manually bring axes closer to look better
    auto_rng = np.max([np.max(col) for col in data_to_plot]) < 2*np.max(means+std)

    # stats
    statvals = []
    if 'MW' in stats and N==2:
        stat, pval = sp.stats.mannwhitneyu(data[0], data[1], alternative='two-sided')
        statvals += [['MW', pval]]
    if 'ttest' in stats and N==2:
        stat, pval = sp.stats.ttest_ind(data[0], data[1])
        statvals += [['T-test', pval]]
    if 'ANOVA' in stats:
        print('ANOVA not yet implemented')
    if 'KW' in stats:
        print('Kruskal–Wallis test not yet implemented')
    if len(statvals) > 0:
        stat_str = '. '.join(['P(%s)=%.3f' % (x[0], x[1]) for x in statvals])
        title = title + '. ' + stat_str

    layout = go.Layout(
        title=title,
        xaxis=xaxis,
        yaxis={'title': ylbl, 'range': [0, np.max(means+std)*2], 'autorange': auto_rng},
        bargap=.5,
        hovermode='closest',
        showlegend = False,
    )

    fig = go.Figure(data=traces, layout=layout)

    if plot:
        plotfunc = pyo.iplot if in_notebook() else pyo.plot
        plotfunc(fig)
    else:
        return fig

    return None


def propBarPlot(data,           # list of 1D data vectors
            names=None,     # names of data vectors
            title=' ',      # title of plot
            ylbl='Proportion',    # y-label\
            plot=True):
    """
        Makes a custom plotly proportion barplot
        :return:
        """
    data = np.array(data)
    N = len(data)
    Lx = [len(col) for col in data]
    print(Lx)

    if names is None:
        names = [str(i + 1) for i in range(N)]
    if N >= 3:
        cols = cl.scales[str(N)]['qual']['Set1']
    else:
        cols = cl.scales[str(3)]['qual']['Set1'][0:N]
    jitter = .03

    means = [np.mean(col) for col in data]
    std = [(means[n]*(1-means[n])/Lx[n])**.5 for n in range(N)]

    traces = []
    bars = [go.Bar(
        x=list(range(N)),
        y=means,
        marker=dict(
            color=cols),
        text=['N = %d' % (l) for l in Lx],
        name='BAR',
        error_y=dict(
            type='data',
            array=std,
            visible=True
        ),
        showlegend=False
    )]
    traces += bars

    xaxis = go.XAxis(
        # title="",
        showgrid=True,
        showline=True,
        ticks="",
        showticklabels=True,
        linewidth=2,
        ticktext=names,
        tickvals=list(range(N)),
        tickfont=dict(size=18)
    )

    layout = go.Layout(
        title=title,
        xaxis=xaxis,
        yaxis={'title': ylbl},
        bargap=.5,
        hovermode='closest',
        showlegend=False,
    )

    fig = go.Figure(data=traces, layout=layout)

    if plot:
        plotfunc = pyo.iplot if in_notebook() else pyo.plot
        plotfunc(fig)
    else:
        return fig

    return None


def multiLine(data,         # [N,Lx] numpy array or list, where rows are each line
              x=None,       # optional x-data
              lines=True,   # 1/0 whether we want to plot each of the individual lines
              mean=True,    # True/False where want mean+std line
              names=None,   # names of each data list
              plot=True,    # if false, just returns plotly json object
              title='',     # title of plot
              ylbl='',      #
              xlbl='',      #
              norm=None):   # input to norm_mat function if want to norm the data
    """
    Plots bunch of lines + mean in plotly
    """

    data = np.array(data)
    N, Lx = data.shape

    if norm is not None:
        data = norm_mat(data, method=norm)
    if names is None: names = ['#%d' %(i) for i in range(N)]
    if x is None: x=np.array(range(Lx))
    x = np.atleast_2d(x)
    uniquex=True if len(x)>1 else False     #whether same x for all y

    traces = []
    if lines:
        lineplots = [go.Scatter(y=data[i], x=x[i*uniquex], name=names[i], line={'width': 1})
             for i in range(N)]
        traces += lineplots

    if mean and not uniquex:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        plotmean = go.Scatter(x=x[0], y=mean, name='Mean', legendgroup='mean', line={'width': 6})
        ploterror_top = go.Scatter(
            x=x[0],
            y=mean + std,
            fill='none',
            fillcolor='rgba(0,100,80,0.2)',
            mode='lines',
            marker=dict(color="444"),
            line=dict(width=0),
            showlegend=False,
            legendgroup='mean',
            name = 'upper bound',
            opacity = .7,
        )
        ploterror_bottom = go.Scatter(
            x=x[0],
            y=mean - std,
            fill='tonexty',
            fillcolor='rgba(0,100,80,0.2)',
            mode='lines',
            marker=dict(color="444"),
            line=dict(width=0),
            showlegend=False,
            legendgroup='mean',
            name='lower bound',
            opacity=.7,
        )
        traces = [plotmean, ploterror_top, ploterror_bottom] + traces

    if isinstance(x[0][0], str):
        xaxis = go.XAxis(
            title=xlbl,
            showgrid=True,
            showticklabels=True,
            # ticktext=['OFF', 'aDBS OFF', 'aDBS ON', 'Cont'],
            tickvals=x[0],
            tickfont=dict(size=18))
    else:
        xaxis = go.XAxis(title=xlbl)

    layout = go.Layout(title=title,
                       xaxis=xaxis,
                       yaxis={'title': ylbl},
                       )
    fig = go.Figure(data=traces, layout=layout)

    if plot:
        plotfunc = pyo.iplot if in_notebook() else pyo.plot
        plotfunc(fig)
    else:
        return fig


def plotDF( df,             # pandas DF
            title='',       # title of plot
            ylbl ='',       # ylabel
            linemode='lines',   # 'lines'/'markers'/'lines+markers'
            plot=True,      # 1/0 whether we want to plot each of the individual lines
        ):
    """
    This plots a pandas DF.
    NOTE: see also plotly's cufflinks package which makes pnadas plotting super easy!
        cf.go_offline()
        df.iplot(kind='scatter')
    """

    traces = [go.Scatter(x=df.index, y=df[col], name=col, mode=linemode)  for col in df.columns]

    layout = go.Layout(title=title,
                       xaxis={'title': df.index.name},
                       yaxis={'title': ylbl},
                       showlegend=True,
                       )

    fig = go.Figure(data=traces, layout=layout)

    if plot:
        plotfunc = pyo.iplot if in_notebook() else pyo.plot
        plotfunc(fig)
    else:
        return fig



def multiMean(data,
              x=None,
              std=True,
              names=None,
              plot=True,
              title='',
              ylbl='',
              xlbl='',
              norm=None,
              indiv=False,
              indivnames=None):
    """
    Plots means of multiple data matrices
    :param data: list of data matrices
    :param x: optional x-data
    :param std: 1/0. If 1 plots shaded std deviation around mean
    :param names: names of data
    :param plot: if false, just returns plotly json object
    :param title: title of plot
    :param ylbl:
    :param xlbl:
    :param norm: nput to norm_mat function if want to norm the data
    :param indiv: 1/0 whether we want to plot each of the individual lines
    :param indivnames: names of individual line traces
    :return:
    """
    data = [np.atleast_2d(np.array(d)) for d in data]
    N = len(data)
    Ncol, Lx = zip(*[d.shape for d in data])
    if len(np.unique(Lx)) != 1: raise ValueError('Input data sources must be of the same length (Lx)')
    Lx = Lx[0]

    if norm is not None:
        data = [norm_mat(d, method=norm) for d in data]
    if names is None: names = ['#%d' % (i) for i in range(N)]
    if x is None: x = np.array(range(Lx))
    x = np.atleast_2d(x)

    traces = []
    cols = cl.scales[str(max(3, N))]['qual']['Set1']
    tcols = ['rgba' + c[3:-1] + ',.2)' for c in cols]
    for n in range(N):
        mean = np.mean(data[n], axis=0)
        std = np.std(data[n], axis=0)
        plotmean = go.Scatter(x=x[0], y=mean, name=names[n], legendgroup=names[n], line={'width': 4, 'color': cols[n]})
        ploterror_top = go.Scatter(
            x=x[0],
            y=mean + std,
            fill='none',
            fillcolor=tcols[n],
            mode='lines',
            marker=dict(color=tcols[n]),
            line=dict(width=0),
            showlegend=False,
            legendgroup=names[n],
            name=names[n] + ' UB',
            opacity=.7,
        )
        ploterror_bottom = go.Scatter(
            x=x[0],
            y=mean - std,
            fill='tonexty',
            fillcolor=tcols[n],
            mode='lines',
            marker=dict(color=tcols[n]),
            line=dict(width=0),
            showlegend=False,
            legendgroup=names[n],
            name=names[n] + ' LB',
            opacity=.7,
        )
        traces += [plotmean, ploterror_top, ploterror_bottom]
        if indiv and Ncol[n]>1:
            inames = ['']*Ncol[n] if indivnames is None else indivnames
            indivlines = [go.Scatter(x=x[0], y=l, showlegend=c==0, name=names[n] + ' |', legendgroup=names[n] + ' |',
                                     hovertext=inames[c], hoverinfo='text', opacity=.3,
                                     line={'width': 1, 'color': cols[n], 'dash': 'dot'})
            for c, l in enumerate(data[n])]
            traces += indivlines

    layout = go.Layout(title=title,
                       xaxis={'title': xlbl},
                       yaxis={'title': ylbl},
                       hovermode='closest',
                       )
    fig = go.Figure(data=traces, layout=layout)

    if plot:
        plotfunc = pyo.iplot if in_notebook() else pyo.plot
        plotfunc(fig)
    else:
        return fig

def plotHist2D(x,           # 1D vector
               y,           # 1D vector
               bins=[15, 30],   # # of bins in histogram
               xlbl='',
               ylbl='',
               title='',
               log=False,   # whether to log the histogram counts
               mean=False,  # whether to overlay mean + std dhading onto heatmap
               plot=True
               ):
    """
    plots 2D heatmap. Does the binning in np as its faster than plotly 2D hist
    """
    x = np.array(x);
    y = np.array(y)
    maxstd = 8  # if max above this many stddevs from mean, it is clipped
    percclip = [5, 95]  # percentile above which it is clipped
    meanx, stdx, minx, maxx = np.mean(x), np.std(x), np.min(x), np.max(x)
    xbins = np.linspace(*np.percentile(x, percclip),
                        bins[0]) if meanx + maxstd * stdx < maxx or meanx - maxstd * stdx > minx else bins[0]
    meany, stdy, miny, maxy = np.mean(y), np.std(y), np.min(y), np.max(y)
    ybins = np.linspace(*np.percentile(y, percclip),
                        bins[1]) if meany + maxstd * stdy < maxy or meany - maxstd * stdy > miny else bins[1]

    H, xedges, yedges = np.histogram2d(x, y, bins=[xbins, ybins], normed=False)
    H = H.T  # extremely important!!!!!

    if log:
        H[H == 0] = np.nan
        H = np.log10(H);
        zlbl = 'log(Count)'
    else:
        zlbl = 'Count'

    hist = go.Heatmap(
        x=xedges,  # sample to be binned on the x-axis
        y=yedges,  # sample to be binned on of the y-axis
        z=H,
        name='Heatmap',
        showlegend=True,
        zsmooth='best',  # (!) apply smoothing to contours
        colorscale='Portland',  # choose a pre-defined color scale
        colorbar=go.ColorBar(
            titleside='right',  # put title right of colorbar
            ticks='outside',  # put ticks outside colorbar
            title=zlbl,
        )
    )
    plots=[hist]

    # plotting trendline
    if mean:
        Hnorm = copy.deepcopy(H)
        Hnorm[np.isnan(Hnorm)]=0
        Hnorm = Hnorm / np.sum(Hnorm, axis=0)
        Px_given_y = np.atleast_2d(yedges[:-1]) @ Hnorm
        dx = xedges[1]-xedges[0]
        meanLine = [go.Scatter(x=xedges+dx/2, y=Px_given_y[0], name='Trendline', showlegend=True)]
        plots = meanLine + plots

    layout = go.Layout(title=title,
                       xaxis={'title': xlbl},
                       yaxis={'title': ylbl},
                       showlegend=True,
                       )

    fig = go.Figure(data=plots, layout=layout)
    if plot:
        pyo.iplot(fig)
    else:
        return fig

def boxPlot(med, quartiles, minmax, mean=None, outliers=None, name='boxplot', horiz=True, offset=0,
            legendGroup='boxplot', showleg=False, plot=False, col='blue', width=8):
    """
    Makes very light plotly boxplot. Unlike theirs, this can take externally calc'd values rather than just data to make it go much faster.
    :param med:
    :param quartiles:
    :param minmax:
    :param mean:
    :param name:
    :param horiz:
    :param offset:
    :param legendGroup:
    :param plot:
    :param col:
    :return:
    """
    show_indiv_leg=False    #set to true for debug mode
    if horiz:
        wideaxis='x'
        offsetaxis='y'
    else:
        wideaxis = 'y'
        offsetaxis = 'x'

    if mean:
        text='Median=%.3e <br> Mean=%.3e <br> [Q1,Q2]=[%.3e,%.3e] <br> [min, max]=[%.3e,%.3e]' % \
             (med,mean, *quartiles, *minmax)
    else:
        text = 'Median=%.3e <br> [Q1,Q2]=[%.3e,%.3e] <br> [min, max]=[%.2f,%.2f]' \
               % (med, *quartiles, *minmax)

    thickLine = [{wideaxis:quartiles, offsetaxis:[offset]*2,
                    'name':name, 'showlegend':showleg, 'legendgroup':legendGroup, 'type': 'scatter',
                    'line':{'color': col, 'width': width}, 'opacity':.4, 'hovertext':text, 'hoverinfo':'name+text',
                  }]
    thinLine = [{wideaxis:minmax, offsetaxis:[offset]*2,
                    'name':name, 'showlegend':show_indiv_leg, 'legendgroup':legendGroup, 'type': 'scatter',
                    'line': {'color': col, 'width': 2}, 'opacity':.4, 'hovertext':text, 'hoverinfo':'name+text'}]
    medPoint = [{wideaxis:[med], offsetaxis:[offset], 'hovertext':text, 'hoverinfo':'name+text',
                    'name':name, 'showlegend':show_indiv_leg, 'legendgroup':legendGroup, 'mode': 'markers',
                    'marker':{'color':'black', 'symbol':'square', 'size':8}, 'opacity':1}]
    boxPlots = thickLine + thinLine + medPoint
    if mean is not None:
        meanPoint = [{wideaxis: [mean], offsetaxis: [offset], 'hovertext':text, 'hoverinfo':'name+text',
                     'name': name, 'showlegend': show_indiv_leg, 'legendgroup': legendGroup,
                     'mode': 'markers',
                     'marker': {'color': 'white', 'symbol': 'diamond', 'size': 8,
                                'line': {'color':'black', 'width':1}
                               },
                     'opacity': 1,
                     'line': {'color':'black'}}]
        boxPlots += meanPoint
    if outliers is not None:
        outlierplot = [{wideaxis:outliers, offsetaxis:[offset]*len(outliers), 'name':name, 'legendgroup':legendGroup,
                        'mode':'markers', 'marker':dict(size = 2, color=col), 'hoverinfo': wideaxis+'+name'}]
        boxPlots += outlierplot

    if plot:
        fig = go.Figure(data=boxPlots)
        pyo.iplot(fig)
    else:
        return boxPlots

def removeOutliers(data, stdbnd=6, percclip=[5,95], rmv=True):
    N = len(data)
    mean = np.mean(data)
    med = np.median(data)
    std = np.std(data)
    min = np.min(data)
    max = np.max(data)
    rng = [min, max]
    adj = False

    if rmv:
        if mean + stdbnd*std < max:    # if data has large max tail adjust upper bound of rng
            rng[1] = np.percentile(data, percclip[1])
            adj = True
        if mean - stdbnd*std > min:    # if data has large min tail adjust lower bound of rng
            rng[0] = np.percentile(data, percclip[0])
            adj = True
    # remove data outside rng
    # TODO: this can be optimized such that if rmv=0, no searching need be done...
    Igood = (data>rng[0]) & (data < rng[1])
    included_data = data[Igood]
    outliers = data[~Igood]

    stats = {'mean':mean, 'med':med, 'std':std, 'min':min, 'max':max}

    return adj, included_data, outliers, rng, stats

def scattermatrix(df,
                  title = 'Scatterplot Matrix',
                  plot=True):  # if false, just returns plotly json object
    """
    This makes a scattermatrix for data
    """

    cols = df.columns
    N = len(cols)

    fig = py.tools.make_subplots(rows=N, cols=N)

    for n1 in range(1,N+1):
        for n2 in range(1,n1+1):
            #print('n1:%d, n2:%d' %(n1,n2))
            if n1==n2:
                #plot hist
                ff = plotHist(df[cols[n1-1]],  # 1D list/np vector of data
                         maxData=500,  # max # of points to plot above histogram (if too high, it will be slow)
                         plot=False,  # 1/0. If 0, returns plotly json object, but doesnt plot
                         rm_outliers=True,  # 1/0 whether to remove outliers or not
                         density=True,  # whether to plot PDF or count
                         boxplot = 0,
                         scatter = 0,
                         diff_tst=0)
                [fig.append_trace(d, n1, n2) for d in ff.data]
            if n2 < n1:
                # plot scatter
                ff = corrPlot(df[cols[n1-1]],                 # 1D data vector or list of 1D dsata vectors
                     df[cols[n2-1]],                 # 1D data vector or list of 1D dsata vectors
                     maxdata=500,      # max # of points to plot above histogram (if too high, it will be slow)
                     addCorr=False,      # whether to add correlation statistics into plot (R2, spearmanR2, Pvals, & y=mx+b)
                     addCorrLine=False,     # whether to plot correlation line
                     addXYline=False,      # whether to plot y=x line
                     plot=False,         # if false, just returns plotly json object
                )
                [fig.append_trace(d, n1, n2) for d in ff.data]

    fig['layout'].update(title=title)
    fig['layout'].update(showlegend=False)
    [fig['layout']['yaxis' + str((n-1)*N+1)].update(title=cols[n-1]) for n in range(1,N+1)]

    if plot:
        plotfunc = pyo.iplot if in_notebook() else pyo.plot
        plotfunc(fig)
    else:
        return fig

## Plotly plot subcomponents
def abs_line(position, orientation, color='r', width=3, annotation=None):
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
        'line': {
            'color': color,
            'width': width,
        },
    }

    return shape

def vline(position, **params):
    return abs_line(position, orientation='v', **params)

def hline(position, **params):
    out = abs_line(position, orientation='h', **params)
    return out

## misc utility functions
def removeNaN(x):
    # This function removes NaNs
    return x[~np.isnan(x)]

def add_jitter(data,std_ratio=.03):
    "Adds random noise to a data series"
    std = np.std(data)
    data_out = data + np.random.normal(0, std*std_ratio, size=data.shape)
    return data_out

###Dash wrappers
def dashSubplot(plots,
                min_width=18,  # min width of column (in %). If more columns, scrolling is enabled
                max_width=50,  # max width of column (in %).
                indiv_widths=None,  # can specify list of individual column widths
                ):

    # remove empty elements of list
    plots = [[plt for plt in col if plt != []] for col in plots]    # remove empty plots from each column
    for i in range(len(plots)-1, -1, -1):   # remove empty columns
        if plots[i] == []:
            plots.pop(i)
            if indiv_widths is not None:
                indiv_widths.pop(i)

    Ncol = len(plots)  # number of columns

    if indiv_widths is None:
        col_width = [min(max_width, max(int(100/Ncol-2), min_width) )] * Ncol
    else:
        col_width = indiv_widths


    col_style = [{'width': str(col_width[i]) + '%',
             'display': 'inline-block',
             'vertical-align': 'top',
             'margin-right': '25px'} for i in range(Ncol)]

    layout = html.Div(
        [html.Div(plots[i], style=col_style[i]) for i in range(Ncol)],
        style = {'margin-right': '0px',
                 'position': 'absolute',
                 'width': '100%'}
    )

    return layout

def horizontlDiv(dashlist,
                 id='L',    # either single element or list. If single, id of html divs will be this + # (ie 'L1', 'L2', etc..
                 width=50): #either total width or list of indiv widths
    N = len(dashlist)
    if type(width) == int:
        indiv_width = [str(int(width/N))+'%'] * N
    elif type(width) == list:
        indiv_width = [int(w)+'%' for w in width]
    else:
        print('ERROR: width must either be int or list of ints!')

    horiz_div = [html.Div(i, id=id+str(c),
                          style={'width': indiv_width[c],
                                 'display': 'inline-block',
                                 'vertical-align': 'middle'})
                 for c,i in enumerate(dashlist)]
    return horiz_div

def dashSubplot_from_figs(figs):
    n_r = int(np.ceil(np.sqrt(len(figs))))
    i_r = 0
    i_c = 0
    d_plot = [[] for i in range(n_r)]

    for fig in figs:
        i_c += 1
        if i_c >= n_r:
            i_r += 1
            i_c = 0
        da = dcc.Graph(figure=fig, id=' ')
        d_plot[i_r].append(da)
        i_c += 1
        if i_c >= n_r:
            i_r += 1
            i_c = 0

    layout = dashSubplot(d_plot)
    return layout

### Plotly misc

def plotOut(fig, plot):
    """ Standard code snippet to decide whether to return plotly fig object or plot """
    if plot:
        plotfunc = pyo.iplot if in_notebook() else pyo.plot
        plotfunc(fig)
        return None
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

