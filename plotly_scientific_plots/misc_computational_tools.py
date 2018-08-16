import numpy as np
import scipy as sp
from numpy.lib.stride_tricks import as_strided
from itertools import product
#plotting
import plotly.offline as pyo
import plotly.graph_objs as go


def norm_mat(X,         # 2D np.ndarray
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


def removeNaN(x):
    # This function removes NaNs
    return x[~np.isnan(x)]


def addJitter(data,std_ratio=.03):
    "Adds random noise to a data series"
    std = np.std(data)
    data_out = data + np.random.normal(0, std*std_ratio, size=data.shape)
    return data_out


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

def perc(x):
    return np.sum(x)/len(x)*100


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