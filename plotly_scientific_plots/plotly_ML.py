from copy import deepcopy
import numpy as np
import copy

# import sklearn
import sklearn as sk
from sklearn.preprocessing import OneHotEncoder
import sklearn.model_selection
import sklearn.ensemble
import sklearn.metrics

# import plotly
import plotly.graph_objs as go
import plotly.figure_factory as ff
# import internal functions
from plotly_scientific_plots.plotly_misc import plotOut

perc = lambda x: np.sum(x)/len(x)*100



def MultiClassROC(y_true, y_scores, **kwargs):
    ''' Wrapper for plotMultiROC for multiclass classification, where'''
    y_scores = np.array(y_scores)
    y_true = np.array(y_true)
    if y_scores.ndim == 1:
        y_scores = np.atleast_2d(y_scores).T
    if y_true.ndim == 1:
        y_true = np.atleast_2d(y_true).T
    if y_scores.shape[1] == 1:  # assuming just giving scores of binary classifier
        return MultiROC(y_true, y_scores, **kwargs)
    n_samples, n_classes = y_scores.shape
    encoder = OneHotEncoder(sparse=False, categories=np.atleast_2d(np.arange(n_classes)).tolist())
    lbls_exp = encoder.fit_transform(y_true.reshape(-1, 1))
    if n_classes > 2:
        return MultiROC(lbls_exp, y_scores, **kwargs)
    else:
        return MultiROC(lbls_exp[:, 1:], y_scores[:, 1:], **kwargs)


def MultiTrialROC(y_true, y_scores, **kwargs):
    ''' Wrapper for plotMultiROC for multiple trials where lbls is the same but scores are differnt '''
    n_samples, n_classes = y_scores.shape
    return MultiROC(np.tile(np.atleast_2d(y_true).T, (1, n_classes)), y_scores, **kwargs)


def plotMultiROC(*args, **kwargs):
    print('\nplotMultiROC has been replaced by MultiClassROC & MultiTrialROC\n')
    return MultiClassROC(*args, **kwargs)


def MultiROC(y_true,        # list of true labels
                    y_scores,   # array of scores for each class of shape [n_samples, n_classes]
                    title='Multiclass ROC Plot',
                    n_points=100,  # reinterpolates to have exactly N points
                    labels=None,  # list of labels for each class
                    threshdot=None,  # whether to plot a dot @ the threshold
                    return_auc=False,
                    metrics=True,
                    plot=True,  # 1/0. If 0, returns plotly json object, but doesnt plot
                ):
    """
    Makes a multiclass ROC plot. Can also be used for binary ROC plot
    """

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    assert y_true.shape == y_scores.shape, 'y_true and y_scores must have the exact same shape!'
    N, n_classes = y_scores.shape

    # calc ROC curves & AUC
    fpr = dict()
    tpr = dict()
    thresh = dict()
    auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], thresh[i] = sk.metrics.roc_curve(y_true[:, i], y_scores[:, i])
        auc[i] = sk.metrics.auc(fpr[i], tpr[i])
        if n_points is not None:
            x = np.linspace(0, 1, n_points)
            indxs = np.searchsorted(tpr[i], x)
            tpr[i] = tpr[i][indxs]
            fpr[i] = fpr[i][indxs]
            thresh[i] = thresh[i][indxs]
            # Add endpoints for proper AUC calcs
            tpr[i] = np.concatenate(([0], tpr[i], [1]))
            fpr[i] = np.concatenate(([0], fpr[i], [1]))
            thresh[i] = np.concatenate(([np.inf], thresh[i], [-np.inf]))

    thresh_txt = dict()
    if metrics:
        acc = deepcopy(thresh)
        f1 = deepcopy(thresh)
        for i in range(n_classes):
            thresh_txt[i] = []
            for j, th in enumerate(thresh[i]):
                preds = y_scores[:, i] > th
                acc[i][j] = np.mean(preds == y_true[:, i])
                f1[i][j] = sk.metrics.f1_score(y_true[:, i], preds)
                thresh_txt[i] += [f'T={th:.4f}. Acc={acc[i][j]:.4f}. F1={f1[i][j]:.4f}']
    else:
        for i in range(n_classes):
            thresh_txt[i] = ['T=%.4f' % t for t in thresh[i]]

    if labels is not None and len(labels) != n_classes:
        print(f'Warning: have {len(labels)} lables, and {n_classes} classes. Disregarding labels')
        labels = None

    if labels is None:
        labels = ['C%d' % n for n in range(1, n_classes+1)]

    labels = [str(x) for x in labels]  # convert labels to str

    # make traces
    traces = []
    [traces.append(go.Scatter(y=tpr[i], x=fpr[i], name=labels[i] + '. AUC= %.2f' % (auc[i]), text=thresh_txt[i],
                              legendgroup=str(i), line={'width': 1}))
        for i in range(n_classes)]
    traces += [go.Scatter(y=[0, 1], x=[0, 1], name='Random classifier', line={'width': 1, 'dash': 'dot'})]

    if threshdot is not None:
        for i in range(n_classes):
            c_indx = (np.abs(thresh[i]-threshdot)).argmin()
            traces += [go.Scatter(x=[fpr[i][c_indx]]*2, y=[tpr[i][c_indx]]*2, mode='markers',
                                  name='Threshold', legendgroup=str(i), showlegend=False)]

    # make layout
    layout = go.Layout(title=title,
                       xaxis={'title': 'FPR'},
                       yaxis={'title': 'TPR'},
                       legend=dict(x=1),
                       hovermode='closest',
    )

    fig = go.Figure(data=traces, layout=layout)

    if return_auc:
        return plotOut(fig, plot), auc[0]
    else:
        return plotOut(fig, plot)


def MultiClassPR(y_true, y_scores, **kwargs):
    ''' Wrapper for plotMultiROC for multiclass classification, where'''
    y_scores = np.array(y_scores)
    y_true = np.array(y_true)
    if y_scores.ndim == 1:
        y_scores = np.atleast_2d(y_scores).T
    if y_true.ndim == 1:
        y_true = np.atleast_2d(y_true).T
    # if y_true.shape[1] == 1 and y_scores.shape[1] == 2:  # assuming we want to get +,- PR plot
    #     y_true = np.concatenate((1-y_true, y_true), axis=1)
    if y_scores.shape[1] == 1:  # assuming just giving scores of binary classifier
        return MultiPR(y_true, y_scores, **kwargs)
    n_samples, n_classes = y_scores.shape
    encoder = OneHotEncoder(sparse=False, categories=np.atleast_2d(np.arange(n_classes)).tolist())
    lbls_exp = encoder.fit_transform(y_true.reshape(-1, 1))
    return MultiPR(lbls_exp, y_scores, **kwargs)
    # if n_classes > 2:
    #     return MultiPR(lbls_exp, y_scores, **kwargs)
    # else:
    #     return MultiPR(lbls_exp[:, 1:], y_scores[:, 1:], **kwargs)


def MultiTrialPR(y_true, y_scores, **kwargs):
    ''' Wrapper for plotMultiROC for multiple trials where lbls is the same but scores are differnt '''
    n_samples, n_classes = y_scores.shape
    return MultiPR(np.tile(np.atleast_2d(y_true).T, (1, n_classes)), y_scores, **kwargs)


def plotMultiPR(*args, **kwargs):
    print('\nplotMultiPR has been replaced by MultiClassPR & MultiTrialPR\n')
    return MultiClassPR(*args, **kwargs)


def MultiPR(y_true,        # list of true labels
                    y_scores,   # array of scores for each class of shape [n_samples, n_classes]
                    title='Multiclass PR Plot',
                    n_points=100,  # reinterpolates to have exactly N points
                    labels=None,  # list of labels for each class
                    threshdot=None,  # whether to plot a dot @ the threshold
                    return_auc=False,
                    metrics=True,
                    plot=True,  # 1/0. If 0, returns plotly json object, but doesnt plot
                ):
    """
    Makes a multiclass PR plot
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    assert y_true.shape == y_scores.shape, 'y_true and y_scores must have the exact same shape!'
    N, n_classes = y_scores.shape

    # if y_scores.ndim == 1:  # convert to [n_samples, n_classes] even if 1 class
    #     y_scores = np.atleast_2d(y_scores).T
    # N, n_classes = y_scores.shape
    # if n_classes == 1:  # needed to avoid inverting when doing binary classification
    #     y_scores = -1 * y_scores

    # calc curves & AUC
    precision = dict()
    recall = dict()
    thresh = dict()
    auc = dict()
    for i in range(n_classes):
        precision[i], recall[i], thresh[i] = sk.metrics.precision_recall_curve(y_true[:, i], y_scores[:, i])
        #average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])
        auc[i] = np.sum(precision[i][1:] * -np.diff(recall[i]))
        if n_points is not None:
            x = np.linspace(precision[i][0], precision[i][-1], n_points)
            indxs = np.searchsorted(precision[i], x)
            precision[i] = precision[i][indxs]
            recall[i] = recall[i][indxs]
            thresh[i] = thresh[i][np.clip(indxs, 0, thresh[i].size - 1)]
            # Add endpoints for proper AUC calcs
            precision[i] = np.concatenate(([0], precision[i], [1]))
            recall[i] = np.concatenate(([1], recall[i], [0]))
            thresh[i] = np.concatenate(([-np.inf], thresh[i], [np.inf]))

    thresh_txt = dict()
    if metrics:
        acc = deepcopy(thresh)
        f1 = deepcopy(thresh)
        for i in range(n_classes):
            thresh_txt[i] = []
            for j, th in enumerate(thresh[i]):
                preds = y_scores[:, i] > th
                acc[i][j] = np.mean(preds == y_true[:, i])
                f1[i][j] = sk.metrics.f1_score(y_true[:, i], preds)
                thresh_txt[i] += [f'T={th:.4f}. Acc={acc[i][j]:.4f}. F1={f1[i][j]:.4f}']
    else:
        for i in range(n_classes):
            thresh_txt[i] = ['T=%.4f' % t for t in thresh[i]]

    if labels is not None and len(labels) != n_classes:
        print(f'Warning: have {len(labels)} labels, and {n_classes} classes. Disregarding labels')
        labels = None

    if labels is None:
        labels = ['C%d' % n for n in range(1, n_classes+1)]

    labels = [str(x) for x in labels]  # convert labels to str

    # make traces
    traces = []
    [traces.append(go.Scatter(y=precision[i], x=recall[i], name=labels[i] + '. AUC= %.2f' % (auc[i]),
                        text=thresh_txt[i], legendgroup=str(i), line={'width': 1})) for i in range(n_classes)]

    if threshdot is not None:
        for i in range(n_classes):
            c_indx = (np.abs(thresh[i]-threshdot)).argmin()
            traces += [go.Scatter(x=[recall[i][c_indx]]*2, y=[precision[i][c_indx]]*2, mode='markers',
                                  name='Threshold', legendgroup=str(i), showlegend=False)]

    # make layout
    layout = go.Layout(title=title,
                       yaxis={'title': 'Precision = P(y=1 | yp=1)',
                              'range': [0, 1]},   # 'Precision = P(yp=y | yp=1)'
                       xaxis={'title': 'Recall = TPR = P(yp=1 | y=1)',
                              'range': [0, 1]}, # 'Recall = TPR = P(yp=y | y=1)'
                       legend=dict(x=1),
                       hovermode='closest',
    )

    fig = go.Figure(data=traces, layout=layout)

    if return_auc:
        return plotOut(fig, plot),
    else:
        return plotOut(fig, plot)


def plotConfusionMatrix(y_true,         # list of true labels
                        y_pred,         # list of predicted labels
                        conf_matrix=None,   # optional mode to directly provide confusion matrix
                        title=None,
                        labels=None,    # list of labels for each class
                        binarized=None,  # if int/str then makes 1vsAll confusion matrix of that class
                        add_totals=True,  # whether to add an extra row for class totals
                        plot=True,      # 1/0. If 0, returns plotly json object, but doesnt plot
                        fontsize=18,    # axis font
                        norm='rows',    # how to norm matrix colors. either 'all'/'rows'/'columns'
                ):
    """
    Plots either a full or binarized confusion matrix

    EX: plotConfusionMatrix(y_true, y_pred, labels)
    """

    if conf_matrix is None:
        n_classes = len(labels) if labels is not None else len(np.unique(np.concatenate((y_pred, y_true))))
        conf_matrix = sk.metrics.confusion_matrix(y_true, y_pred, labels=range(n_classes))
    else:
        n_classes = conf_matrix.shape[0]

    if labels is None:
        labels = ['C%d' % n for n in range(1, n_classes + 1)]

    acc = np.diag(conf_matrix).sum() / np.sum(conf_matrix) * 100

    if binarized is not None:
        # identify index of 1vsAll category
        if type(binarized) == str:
            bin_indx = labels.index(binarized)
        else:
            bin_indx = binarized
        tp = np.sum(np.delete(np.delete(conf_matrix, bin_indx, axis=0), bin_indx, axis=1))
        fp = np.sum(np.delete(conf_matrix[bin_indx, :], bin_indx))
        fn = np.sum(np.delete(conf_matrix, bin_indx, axis=0)[:, bin_indx])
        tn = conf_matrix[bin_indx, bin_indx]
        conf_matrix = np.array([[tp, fn], [fp, tn]])
        labels = ['T', 'F']
        n_classes = 2

    labels = [str(x) for x in labels]   # convert to str
    labels = ['['+x+']' if len(x) == 1 else x for x in labels]    #needed for stupid plotly bug

    # adds an extra row for matrix totals
    conf_matrix_tots = copy.deepcopy(conf_matrix)
    if add_totals:
        pred_tots = np.sum(conf_matrix, 0)
        conf_matrix_tots = np.vstack((conf_matrix, pred_tots))
        true_tots = np.sum(conf_matrix_tots, 1, keepdims=True)
        conf_matrix_tots = np.hstack((conf_matrix_tots, true_tots ))
        labels = labels + ['TOTAL']

    # shorten labels
    labels_short = [x[:10] if type(x) == str else x for x in labels]

    # numeric labels
    num_labels = list(range(len(labels)))

    def normMatByTotal(mat, axis=0):
        ''' This normalzies a matrix by its row (axis=1) or column (axis=0) totals'''
        axis_sums = np.sum(mat, axis=axis, keepdims=True).astype('float32')
        axis_sums[axis_sums == 0] = np.nan  # this avoids divide by 0.
        mat = np.nan_to_num(mat / axis_sums)
        return mat

    # percentage hover labels
    row_percs = normMatByTotal(conf_matrix, axis=1)
    col_percs = normMatByTotal(conf_matrix, axis=0)

    # normalize matrix
    color_mat = copy.deepcopy(conf_matrix_tots)
    if norm != 'all':
        norm_conf_matrix = row_percs if norm=='rows' else col_percs
    else:
        norm_conf_matrix = conf_matrix
    color_mat = color_mat.astype(float)
    color_mat[:norm_conf_matrix.shape[0],:norm_conf_matrix.shape[1]] = norm_conf_matrix

    # hover text
    txt_format = '%d<br><b>Pred:</b> %s <br><b>True:</b> %s <br><b>Row norm:</b> %.3f%% <br><b>Col norm:</b> %.3f%%'
    htext = np.array([[txt_format % (conf_matrix[r,c], labels[c], labels[r], row_percs[r,c]*100, col_percs[r,c]*100)
                       for c in range(n_classes)] for r in range(n_classes)])

    # Adjust Total rows
    if add_totals:
        totals_row_shading = .0    # range 0 to 1. 0=darkest, 1=lightest
        tot_val = np.min(norm_conf_matrix) + (np.max(norm_conf_matrix) - np.min(norm_conf_matrix))*totals_row_shading
        color_mat[-1, :] = tot_val
        color_mat[:, -1] = tot_val
        pred_tot_text = np.array(['<b>%% of Predictions:</b> %.2f%%' % x for x in pred_tots/sum(pred_tots)*100])
        true_tot_text = np.array([['<b>%% of True Data:</b> %.2f%%' % x] for x in true_tots[:-1]/sum(true_tots[:-1])*100]+[['Total Samples']])
        htext = np.hstack((np.vstack((htext, pred_tot_text)), true_tot_text))

    fig = ff.create_annotated_heatmap(color_mat, x=num_labels, y=num_labels,
                                      colorscale='Greys', annotation_text=conf_matrix_tots)

    fig.layout.yaxis.title = 'True'
    fig.layout.xaxis.title = 'Predicted (Total accuracy = %.3f%%)' % acc
    fig.layout.xaxis.titlefont.size = fontsize
    fig.layout.yaxis.titlefont.size = fontsize
    fig.layout.xaxis.tickfont.size = fontsize - 2
    fig.layout.yaxis.tickfont.size = fontsize - 2
    fig.layout.showlegend = False
    # Add label text to axis values
    fig.layout.xaxis.tickmode = 'array'
    fig.layout.xaxis.range = [-.5, n_classes+.5] if add_totals else [-.5, n_classes - .5]
    fig.layout.xaxis.tickvals = num_labels
    fig.layout.xaxis.ticktext = labels_short
    fig.data[0].hoverlabel.bgcolor = 'rgb(188,202,225)'
    if title is not None:
        fig.layout.title = title

    # fig.layout.yaxis.autorange = 'reversed'
    fig.layout.yaxis.tickmode = 'array'
    fig.layout.yaxis.range = [n_classes + .5, -.5] if add_totals else [n_classes - .5, -.5]
    fig.layout.yaxis.tickvals = num_labels
    fig.layout.yaxis.ticktext = labels_short
    fig.layout.margin.l = 120   # adjust left margin to avoid ylbl overlaying tick str's

    fig['data'][0]['xgap'] = 1
    fig['data'][0]['ygap'] = 1
    ## Change annotation font (& text)
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = fontsize-3
        #fig.layout.annotations[i].text = str(conf_matrix_tots.flatten()[i])

    # add hover text
    fig.data[0].text = htext
    fig.data[0].hoverinfo = 'text'

    ### Adjust totals fontstyle
    if add_totals:
        # get totals indxs
        n = n_classes
        last_column_indxs = [(n + 1) * x - 1 for x in range(1, n + 1)]
        last_row_indxs = list(range((n + 1) * (n), (n + 1) ** 2))
        totals_annot_indxs = last_row_indxs + last_column_indxs
        # adjust totals font size & color
        for i in totals_annot_indxs:
            fig['layout']['annotations'][i]['font'] = dict(size=fontsize, color='#000099')

        # Add border lines for total row/col
        data = list(fig['data'])
        data += [go.Scatter(x=[n_classes - .5, n_classes - .5], y=[-.5, n_classes + .5], showlegend=False,
                            hoverinfo='none', line=dict(color='red', width=4, dash='solid'))]
        data += [go.Scatter(y=[n_classes - .5, n_classes - .5], x=[-.5, n_classes + .5], showlegend=False,
                            hoverinfo='none', line=dict(color='red', width=4, dash='solid'))]
        fig = go.Figure(data=data, layout=fig['layout'])

    return plotOut(fig, plot)
