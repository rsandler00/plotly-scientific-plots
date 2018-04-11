import numpy as np
import scipy as sp
from numpy.lib.stride_tricks import as_strided
from itertools import product
import sys
import os
#import pacpy
from IPython import get_ipython

#plotting
import plotly.offline as pyo
import plotly.graph_objs as go
import plotly as py
import matplotlib.pyplot as plt
import pdb

def norm_mat(X,
             X2=None,
             method='zscore',
             input_bounds=[],
             output_bounds=(0,1)):
    """
    This normalizes each row of a matrix with various norm options
    :param X: [N,Lx] matrix
    :param X2: optional 2nd matrix which will be norm'd by the same scale as the first
    :param method: if a number works by np.linalg.norm function.
                    If 'zscore' then zscores.
                    If 'baseline', than norm'd by 1st element of vector
    :return: Y - norm'd matrix
    """

    if X.ndim == 1:
        X = np.atleast_2d(np.array(X)).T
    nCol, Lx = X.shape
    if not isinstance(method, str):
        Y = np.array([col/np.linalg.norm(col,method) for col in X])
    if method == 'zscore':
        Y = np.array([sp.stats.mstats.zscore(col) for col in X])
    if method == 'baseline':  #normalize by 1st element of vector
        Y = [col/col[0] for col in X]
        if X2 is not None:  # normalize 2nd matrix w/ same methods as 1st
            Y2 = [col / X[i][0] for i, col in enumerate(X2)]
            return Y, Y2
    if method == 'boundedscale':
        if input_bounds == []:    #set to min/max of each column
            input_bounds += [np.min(X, axis=0)]
            input_bounds += [np.max(X, axis=0)]
        x_std = (X - input_bounds[0]) / (input_bounds[1] - input_bounds[0])
        Y = x_std * (output_bounds[1] - output_bounds[0]) + output_bounds[0]
    if method == 'non' or method == None:
        Y = X
    return Y


def calcPAC(lo, hi, ):
    pacpy.pac

def getSTA(trigger, signal, rng,
           lags=1,
           norm='zscore', #how each STA trial is normalized in all_sta
           # plotting parameters
           plot=False,
           getFig = False, #if true returns the figure object instead of plotting
           xtra_times = None, #plots dots relative to on times.
           Fs=1,
           removeOutliers=1,     #1/0. If 1 remove data +- 6 std devs from mean
           title='Stimulus Triggered Average'):
    '''
    Computes stimulus triggered average of signal from trigger
    :param trigger:  trigger points around which to calc STA. 1D vector of bins (thus ints)
    :param signal: 1D vector
    :param rng: [lb, ub] in bins of how long to get the STA for
    :param lags: int of how much bins to skip (thus if 2, takes every other point in the STA)
    :param plot: 1/0. If 1, then plots in plotly
    :param Fs: sampling frequency of data. Only relevant for plotting
    :param title: title of plot
    :return:
    '''
    signal = np.array(signal)
    N = len(trigger)

    if removeOutliers:
        #bounds signal at +-6std. Not most elegant, but works...
        stdbnd = 4
        mn = np.mean(signal)
        std = np.std(signal)
        maxx = mn + stdbnd * std
        signal[signal>maxx]=maxx
        minn = mn - stdbnd * std
        signal[signal < minn] = minn

    all_sta = np.array([signal[t - rng[0]:t + rng[1]:lags] for t in trigger if t + rng[1] < len(signal) and t - rng[0] > 0])
    all_sta = norm_mat(all_sta, method=norm)

    bins = np.arange(-rng[0], rng[1], lags)
    sta = np.mean(all_sta, axis=0)

    if plot:
        sta_rescaled = len(trigger)/(np.max(sta)-np.min(sta))*(sta-np.min(sta))
        # good colormaps are Picnic, Rainbow
        heatmap = go.Heatmap(x=bins/Fs*1000, y= np.arange(1,len(all_sta)+1), z=all_sta, colorscale='Rainbow')
        line = go.Scatter(x=bins/Fs*1000, y=sta_rescaled, line={'color':'black', 'width':3}, name='STA')
        yaxis = go.Scatter(x=[0,0], y=[0,len(trigger)], showlegend=False, line={'color': 'black', 'dash':'dash','width':1})
        if xtra_times is not None:
            dots = [go.Scatter(x=(xtra_times-trigger)/Fs*1000, y=np.arange(N)+1,
                               name='dots',
                               mode='markers',
                               marker=dict(size=4, color='white'),
                               )]
        else:
            dots = []


        layout = {'title':title,
                  'xaxis':{'title': 'Times (ms)', 'range': [-rng[0]/Fs*1000, rng[1]/Fs*1000]},
                  'yaxis': {'title': 'Trial', 'range': [1, N]},
                }
        fig = go.Figure(data=[heatmap, line, yaxis]+dots, layout=layout)

        if getFig:
            return fig
        else:
            plotfunc = pyo.iplot if in_notebook() else pyo.plot
            plotfunc(fig)

    return sta, bins, all_sta


def getStimOnOffTimes(sig, Fs, filt_range=(100,130), perc_thresh=50, trg=None, plot=False, plt_rng=[5000,25000]):
    """
    Gets on/off timestamps of stimulation
    :param sig: stimCpy signal from pbrown unilateral data
    :param Fs: sampling freq
    :param filt_range: where to filt for stimulation, which i ussually 130Hz
    :param perc_thresh: threshold of where to dvidie on/off stim
    :param trg: provided trg sinal from ppbrown unilateral data
    :param plot: 1/0 whether to make plotly plot
    :param plt_rng: range to plot, in bins
    :return:
    """
    sig = np.array(sig)
    b, a = sp.signal.butter(4, [filt_range[0] / Fs * 2, filt_range[1] / Fs * 2], 'bandpass')
    filt = sp.signal.filtfilt(b, a, sig, padlen=150)
    hilb = np.abs(sp.signal.hilbert(filt))

    if trg is None:     # actual stim threshold signal not provided
        thresh = np.percentile(hilb, perc_thresh)
        onOffSig = (np.sign(hilb - thresh))
    else:  # stim threshold signal provided
        trg = np.array(trg)
        thresh = np.mean(trg)
        onOffSig = (np.sign(trg - thresh))
    onOffSig[onOffSig==0]=1 #this needed to avoid bug where sig == thresh
    diff = np.diff(onOffSig)
    onTimes = np.where(diff==2)[0]
    offTimes = np.where(diff==-2)[0]
    # insure starts w/ on & ends w/ off
    if offTimes[0] < onTimes[0]: offTimes=offTimes[1:]
    if offTimes[-1] < onTimes[-1]: onTimes=onTimes[:-1]
    if len(onTimes) != len(offTimes):
        raise  ValueError("onTimes doesnt equal offTimes !!!!")
    onOffTimes = np.array([onTimes, offTimes]).T

    if plot is True:
        rng = plt_rng
        scale = max(hilb)
        opc=.7
        traces = []
        traces += [go.Scatter(y=sig[rng[0]:rng[1]], name='Sig', opacity=opc)]
        traces += [go.Scatter(y=filt[rng[0]:rng[1]], name='filt', opacity=opc)]
        traces += [go.Scatter(y=hilb[rng[0]:rng[1]], name='hilb', opacity=opc)]
        if trg is not None:
            traces += [go.Scatter(y=trg[rng[0]:rng[1]], name='trg', opacity=opc)]
        traces += [go.Scatter(x=[0,rng[1]-rng[0]], y=[thresh]*2, name='threshold', opacity=opc)]
        traces += [go.Scatter(y=onOffSig[rng[0]:rng[1]] * scale, name='onOffSig', opacity=opc)]
        traces += [go.Scatter(y=diff[rng[0]:rng[1]] * scale/2*1.2, name='diff', opacity=opc)]
        #traces += [go.Scatter(y=np.diff(hilb[rng[0]-1:rng[1]])*100, name='d`hilb', opacity=opc)]
        np.hstack(([0], np.diff(hilb)))
        in_rng = onOffTimes[(onOffTimes[:,0]>rng[0]) & (onOffTimes[:,0]<rng[1]),:]
        shapes = [{
            'type': 'rect',
            # x-reference is assigned to the x-values
            'xref': 'x',
            # y-reference is assigned to the plot paper [0,1]
            'yref': 'paper',
            'x0': in_rng[i,0]-rng[0],
            'y0': 0,
            'x1': in_rng[i,1]-rng[0],
            'y1': 1,
            'fillcolor': '#d3d3d3',
            'opacity': 0.4,
            'line': {
                'width': 0,
            }
        } for i in range(in_rng.shape[0])]
        #shapes=[]
        if in_notebook():
            pyo.iplot({'data': traces, 'layout': {'shapes': shapes}})
        else:
            pyo.plot({'data': traces, 'layout': {'shapes': shapes}}, filename='getStimOnOffTimes.html')

    return (onOffTimes, hilb, onOffSig)

def conditionalHist(x,y, Nbins=50, std=True,
                    plot=False, xlbl='X', ylbl='Y', stats=None):
    """
    gives E[y|x]
    :param x:
    :param y:
    :param Nbins: number of bins in the hist
    :param std: whether to calc/plot standard deviation of the hist
    :param plot: 1/0
    :param xlbl:
    :param ylbl:
    :param stats: whether to add trendline to plot
    :return:
    """
    x = np.array(x)
    y = np.array(y)

    # calc min/max xrange (ie make robust to outliers
    maxstd = 8  # if max above this many stddevs from mean, it is clipped
    percclip = [5, 95]  # percentile above which it is clipped
    meanx, stdx, minx, maxx = np.mean(x), np.std(x), np.min(x), np.max(x)
    xrange = np.percentile(x, percclip) if meanx + maxstd * stdx < maxx or meanx - maxstd * stdx > minx  else [minx, maxx]

    bins = np.linspace(*xrange, Nbins+2)
    bins = bins[1:-1] # remove edge effects
    dig = np.digitize(x, bins)
    #remove values outside of range
    Igood = list(dig != 0) and list(dig < Nbins)
    condHist = accum(dig[Igood], y[Igood], func=np.mean, size=Nbins)

    if std:
        condStd = accum(dig[Igood], y[Igood], func=np.std)

    if plot:
        traces = []
        traces += [go.Scatter(x=bins, y=condHist, name='E[Y|X]')]
        if std:
            ploterror_top = go.Scatter(
                x=bins,
                y=condHist + condStd,
                fill='none',
                fillcolor='rgba(200,100,80,0.2)',
                mode='lines',
                marker=dict(color="444"),
                line=dict(width=0),
                showlegend=True,
                legendgroup='bounds',
                name='var[Y|X]',
                opacity=.7,
            )
            ploterror_bottom = go.Scatter(
                x=bins,
                y=condHist - condStd,
                fill='tonexty',
                fillcolor='rgba(200,100,80,0.2)',
                mode='lines',
                marker=dict(color="444"),
                line=dict(width=0),
                showlegend=False,
                legendgroup='bounds',
                name='lower bound',
                opacity=.7,
            )
            traces += [ploterror_top, ploterror_bottom]
        if stats:
            slope, intercept, R2, p_val, std_err = sp.stats.linregress(x, y)
            R2sp, p_val_sp = sp.stats.spearmanr(x,y)
            corrtext = 'Pearson [R2, P]=[%.2f,%.2f] <br> ' \
                       'Spearman [R2, P]=[%.2f,%.2f] <br> ' \
                       'y=%.2fx+%.2f' \
                       % (R2, p_val, R2sp, p_val_sp, slope, intercept)
            print(corrtext)
            annots = go.Annotations([go.Annotation(
                    x=0.05,
                    y=0.95,
                    showarrow=False,
                    text=corrtext,
                    xref='paper',
                    yref='paper'
                )])
        else:
            annots = []

        title = 'Conditional Histogram of ' + xlbl + ' | ' + ylbl
        layout = go.Layout(
            title=title,
            xaxis={'title': xlbl},
            yaxis={'title': ylbl},
            annotations=annots
        )
        if in_notebook():
            pyo.iplot({'data': traces, 'layout': layout})
        else:
            pyo.plot({'data': traces, 'layout': layout}, filename='getStimOnOffTimes.html')

    if std:
        return condHist, bins, condStd
    else:
        return condHist, bins

def getBursts(x, method='feingold', minmax=None, plot=False, plt_rng=[0,10000], threshpar=None, xtrasig=None):
    """
    Finds burst regions in the beta power signal
    :param x: beta power signal (ie hilbert envelope.
    :param method: sor far: 'feingold'/?
    :param plot: 1/0. If 1 makes a plotly plot to visualize algo...
    :return: b - binary signal where 1 is bursting regions
    """

    x = np.array(x)
    Lx = len(x)

    if method == 'basic':
        #basic threshold method. anything above is a beta segment
        if threshpar is None: threshpar = 0
        thresh = threshpar
        b_sig = x > thresh
    if method == 'tinkhauser':
        # used in Tinkhauser et al., 2017. He used simple 75th percentile of signal threshold
        if threshpar is None: threshpar = 75
        thresh = np.percentile(x, threshpar)
        b_sig = x > thresh
    if method == 'feingold':
        # burst detection method of Feingold et al., 2015 PNAS
        # it has 2 thresholds, one low, one high. It considers a beta segment any region which continuously crosses the
        # low threshold and has at least 1 point that crosses the high threshold.
        if threshpar is None: threshpar = [1.7, 1.3]    #high / low threshold, in terms of median. Feingold uses [3, 1.5]
        med = np.median(x)
        thresh = threshpar[0] * med
        lowthresh = threshpar[1] * med
        multithresh_sig = np.zeros(x.shape)
        multithresh_sig[x > lowthresh] = 1
        multithresh_sig[x > thresh] = 3
        diff = np.diff(multithresh_sig)
        high_to_mid = np.where(diff==-2)[0]
        mid_to_low = np.where(diff == -1)[0]
        high_to_low = mid_to_low[np.unique(np.searchsorted(mid_to_low, high_to_mid))[:-1]]
        high_to_low_direct = np.where(diff == -3)[0]
        high_to_low = np.sort(np.hstack((high_to_low, high_to_low_direct)))

        mid_to_high = np.where(diff == 2)[0]
        low_to_mid = np.where(diff == 1)[0]
        low_to_high = low_to_mid[np.unique(np.searchsorted(low_to_mid, mid_to_high))[:-1] - 1]
        low_to_high_direct = np.where(diff == 3)[0]
        low_to_high = np.sort(np.hstack((low_to_high, low_to_high_direct)))

        if len(low_to_high) < len(high_to_low):
            high_to_low = high_to_low[np.searchsorted(high_to_low, low_to_high)]
        if len(high_to_low) < len(low_to_high):  #debug
            low_to_high = low_to_high[np.searchsorted(low_to_high, high_to_low)-1]
        b_sig = np.zeros(x.shape)
        for i in range(len(low_to_high)):
            b_sig[low_to_high[i]:high_to_low[i]]=1

    # get start/stop of each burst
    diff = np.hstack((0, np.diff(b_sig.astype(int))))
    start = np.where(diff == 1)[0]
    stop = np.where(diff == -1)[0]
    if start[-1]>stop[-1]: start=start[:-1]
    if stop[0] < start[0]: stop = stop[1:]
    b_rng = np.array([start, stop])
    if minmax is not None:
        good  = [(np.diff(b_rng.T) > minmax[0]) & (np.diff(b_rng.T) < minmax[1])][0].flatten()
        b_rng = b_rng[:, good]
        b_sig = np.zeros(x.shape) #resete & redo signal
        for i in range(b_rng.shape[1]):
            b_sig[b_rng[0,i]:b_rng[1,i]] = 1

    if plot:
        opc = .7
        traces = []
        traces += [go.Scatter(y=x[plt_rng[0]:plt_rng[1]], name='Beta Power', opacity=opc)]
        if xtrasig is not None:
            traces += [go.Scatter(y=np.array(xtrasig)[plt_rng[0]:plt_rng[1]], name='XtraSig', opacity=opc)]
        traces += [go.Scatter(x=[0, plt_rng[1]-plt_rng[0]], y=[thresh] * 2, name='Threshold', opacity=opc)]
        if method == 'feingold':
            traces += [go.Scatter(x=[0, plt_rng[1]-plt_rng[0]], y=[lowthresh] * 2, name='Low Threshold', opacity=opc)]

        # grey areas
        in_rng = b_rng[:, (b_rng[0, :] > plt_rng[0]) & (b_rng[0, :] < plt_rng[1])]
        shapes = [{
            'type': 'rect',
            # x-reference is assigned to the x-values
            'xref': 'x',
            # y-reference is assigned to the plot paper [0,1]
            'yref': 'paper',
            'x0': in_rng[0, i] - plt_rng[0],
            'y0': 0,
            'x1': in_rng[1, i] - plt_rng[0],
            'y1': 1,
            'fillcolor': '#d3d3d3',
            'opacity': 0.4,
            'line': {
                'width': 0,
            }
        } for i in range(in_rng.shape[1])]
        # shapes=[]

        if in_notebook():
            pyo.iplot({'data': traces, 'layout': {'shapes': shapes}})
        else:
            pyo.plot({'data': traces, 'layout': {'shapes': shapes}}, filename='getStimOnOffTimes.html')

    return b_sig, b_rng

def autocorrelation(x, maxlag):
    """
    Autocorrelation with a maximum number of lags.

    `x` must be a one-dimensional numpy array.

    This computes the same result as
        numpy.correlate(x, x, mode='full')[len(x)-1:len(x)+maxlag]

    The return value has length maxlag + 1.
    """
    x = _check_arg(x, 'x')
    p = np.pad(x.conj(), maxlag, mode='constant')
    T = as_strided(
        p[maxlag:],
        shape=(maxlag + 1, len(x) + maxlag),
        strides=(-p.strides[0], p.strides[0]))
    return T.dot(p[maxlag:].conj())

def crosscorrelation(x, y, lag=None, verbose=True):
    '''Compute lead-lag correlations between 2 time series.

    <x>,<y>: 1-D time series.
    <lag>: lag option, could take different forms of <lag>:
          if 0 or None, compute ordinary correlation and p-value;
          if positive integer, compute lagged correlation with lag
          upto <lag>;
          if negative integer, compute lead correlation with lead
          upto <-lag>;
          if pass in an list or tuple or array of integers, compute
          lead/lag correlations at different leads/lags.

    Note: when talking about lead/lag, uses <y> as a reference.
    Therefore positive lag means <x> lags <y> by <lag>, computation is
    done by shifting <x> to the left hand side by <lag> with respect to
    <y>.
    Similarly negative lag means <x> leads <y> by <lag>, computation is
    done by shifting <x> to the right hand side by <lag> with respect to
    <y>.

    Return <result>: a (n*2) array, with 1st column the correlation
    coefficients, 2nd column correpsonding p values.

    Currently only works for 1-D arrays.
    '''

    import numpy
    from scipy.stats import pearsonr

    if len(x) != len(y):
        raise ('Input variables of different lengths.')

    # --------Unify types of <lag>-------------
    if numpy.isscalar(lag):
        if abs(lag) >= len(x):
            raise ('Maximum lag equal or larger than array.')
        if lag < 0:
            lag = -numpy.arange(abs(lag) + 1)
        elif lag == 0:
            lag = [
                0,
            ]
        else:
            lag = numpy.arange(lag + 1)
    elif lag is None:
        lag = [
            0,
        ]
    else:
        lag = numpy.asarray(lag)

    # -------Loop over lags---------------------
    result = []
    if verbose:
        print
        '\n#<lagcorr>: Computing lagged-correlations at lags:', lag

    for ii in lag:
        if ii < 0:
            result.append(pearsonr(x[:ii], y[-ii:]))
        elif ii == 0:
            result.append(pearsonr(x, y))
        elif ii > 0:
            result.append(pearsonr(x[ii:], y[:-ii]))

    result = numpy.asarray(result)

    return result

def getNarrowBandSig(x, Fs, bnd, order=5, domag=True, dophase=True):
    x = np.array(x)
    N = len(x)
    b, a = sp.signal.butter(order, np.array(bnd)/Fs*2, btype='bandpass')
    xfilt = sp.signal.filtfilt(b, a, x)

    mag, phase = [], []
    if domag or dophase:
        hilb = fastHilbert(xfilt)
        if domag:
            mag = np.abs(hilb)
        if dophase:
            phase = np.unwrap(np.angle(hilb))

    return xfilt, mag, phase

def getNarrowPac(lo_sig,    # 1d np array sig for low freq. phase
                 lo_rng,    # low freq filter range
                 hi_rng,    # hi freq filter range
                 Fs,        # sampling freq (Hz)
                 hi_sig=None, # sig for high freq power. If none, same as lo_sig
                 MI=True,   # 1/0 whether to also calc MI & print it in plot title
                 norm = 'zscore'):  # norm for STA plot.
    """
    This function plots phase-amplitude coupling (PAC) STA as in Canolty et al., 2006 Fig. 1B.
    """

    if hi_sig is None: hi_sig = lo_sig

    # get lo phase signal
    b, a = sp.signal.butter(4, [lo_rng[0] / Fs * 2, lo_rng[1] / Fs * 2], 'bandpass')
    lo_filt = sp.signal.filtfilt(b, a, lo_sig, padlen=150)
    lo_phase = np.angle(fastHilbert(lo_filt))

    # get hi amp signal
    b, a = sp.signal.butter(4, [hi_rng[0] / Fs * 2, hi_rng[1] / Fs * 2], 'bandpass')
    hi_filt = sp.signal.filtfilt(b, a, hi_sig, padlen=150)
    hi_amp = np.angle(fastHilbert(hi_filt))

    # get lo phase triggers
    trgs = np.where(np.diff(lo_phase)<-6)[0]

    # Modulation Index (MI)
    if MI:
        z_array = hi_amp * np.exp(1j * lo_phase)
        MI = np.abs(np.mean(z_array))
        title = 'PAC STA. MI=%.2e' % (MI)
    else:
        MI= np.nan
        title = 'PAC STA'

    PACsta = getSTA(trgs, hi_amp, rng=[150, 160], Fs=Fs, lags=4, plot=1, getFig=1, title=title, norm=norm)

    return PACsta, MI, lo_phase, hi_amp


### Generic python helper functions. Not specifically neuroscience related

def accum(accmap, a, func=None, size=None, fill_value=0, dtype=None):
    """
    An accumulation function similar to Matlab's `accumarray` function.

    Parameters
    ----------
    accmap : ndarray
        This is the "accumulation map".  It maps input (i.e. indices into
        `a`) to their destination in the output array.  The first `a.ndim`
        dimensions of `accmap` must be the same as `a.shape`.  That is,
        `accmap.shape[:a.ndim]` must equal `a.shape`.  For example, if `a`
        has shape (15,4), then `accmap.shape[:2]` must equal (15,4).  In this
        case `accmap[i,j]` gives the index into the output array where
        element (i,j) of `a` is to be accumulated.  If the output is, say,
        a 2D, then `accmap` must have shape (15,4,2).  The value in the
        last dimension give indices into the output array. If the output is
        1D, then the shape of `accmap` can be either (15,4) or (15,4,1)
    a : ndarray
        The input data to be accumulated.
    func : callable or None
        The accumulation function.  The function will be passed a list
        of values from `a` to be accumulated.
        If None, numpy.sum is assumed.
    size : ndarray or None
        The size of the output array.  If None, the size will be determined
        from `accmap`.
    fill_value : scalar
        The default value for elements of the output array.
    dtype : numpy data type, or None
        The data type of the output array.  If None, the data type of
        `a` is used.

    Returns
    -------
    out : ndarray
        The accumulated results.

        The shape of `out` is `size` if `size` is given.  Otherwise the
        shape is determined by the (lexicographically) largest indices of
        the output found in `accmap`.


    Examples
    --------
    >>> from numpy import array, prod
    >>> a = array([[1,2,3],[4,-1,6],[-1,8,9]])
    >>> a
    array([[ 1,  2,  3],
           [ 4, -1,  6],
           [-1,  8,  9]])
    >>> # Sum the diagonals.
    >>> accmap = array([[0,1,2],[2,0,1],[1,2,0]])
    >>> s = accum(accmap, a)
    array([9, 7, 15])
    >>> # A 2D output, from sub-arrays with shapes and positions like this:
    >>> # [ (2,2) (2,1)]
    >>> # [ (1,2) (1,1)]
    >>> accmap = array([
            [[0,0],[0,0],[0,1]],
            [[0,0],[0,0],[0,1]],
            [[1,0],[1,0],[1,1]],
        ])
    >>> # Accumulate using a product.
    >>> accum(accmap, a, func=prod, dtype=float)
    array([[ -8.,  18.],
           [ -8.,   9.]])
    >>> # Same accmap, but create an array of lists of values.
    >>> accum(accmap, a, func=lambda x: x, dtype='O')
    array([[[1, 2, 4, -1], [3, 6]],
           [[-1, 8], [9]]], dtype=object)
    """

    # Check for bad arguments and handle the defaults.
    if accmap.shape[:a.ndim] != a.shape:
        raise ValueError("The initial dimensions of accmap must be the same as a.shape")
    if func is None:
        func = np.sum
    if dtype is None:
        dtype = a.dtype
    if accmap.shape == a.shape:
        accmap = np.expand_dims(accmap, -1)
    adims = tuple(range(a.ndim))
    if size is None:
        size = 1 + np.squeeze(np.apply_over_axes(np.max, accmap, axes=adims))
    size = np.atleast_1d(size)

    # Create an array of python lists of values.
    vals = np.empty(size, dtype='O')
    for s in product(*[range(k) for k in size]):
        vals[s] = []
    for s in product(*[range(k) for k in a.shape]):
        indx = tuple(accmap[s])
        val = a[s]
        vals[indx].append(val)

    # Create the output array.
    out = np.empty(size, dtype=dtype)
    for s in product(*[range(k) for k in size]):
        if vals[s] == []:
            out[s] = fill_value
        else:
            out[s] = func(vals[s])

    return out


def _check_arg(x, xname):
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError('%s must be one-dimensional.' % xname)
    return x

perc = lambda x: np.sum(x)/len(x)*100

rngmap = lambda sig, rngs, func: [func(sig[rngs[0,i]:rngs[1,i]]) for i in range(rngs.shape[1])]

def getMasterDir():
    masterdirfile = os.path.join(os.path.dirname(os.getcwd()), 'masterdir.txt')
    if os.path.isfile(masterdirfile):
        file = open(masterdirfile, 'r')
        masterdir = file.read()
    else:
        # this is the default master directory if no file found. change as needed
        masterdir =  'F:\\Data\\BrownData\\Unilateral Study\\'
    return masterdir

def fastHilbert(signal):
    """
    This speeds up scipy's native hilbert func by automatically zero-padding
    :param signal: 1d np array
    :return: hilb
    """

    N = len(signal)
    result = sp.signal.hilbert(signal, N=sp.fftpack.next_fast_len(N))
    result = result[0:N]

    return result