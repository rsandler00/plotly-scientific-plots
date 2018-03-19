import numpy as np
import scipy as sp
import sys
import os
import copy
# import sklearn
import sklearn as sk
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.ensemble
import sklearn.metrics
# import plotly
import plotly.offline as pyo
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly as py
import matplotlib.pyplot as plt
from plotly_plot_tools import in_notebook
import colorlover as cl

perc = lambda x: np.sum(x)/len(x)*100


def plotMultiROC(y_true,        # list of true labels
                    y_scores,   # array of scores for each class of shape [n_samples, n_classes]
                    title = 'Multiclass ROC Plot',
                    labels = None, # list of labels for each class
                    plot=True,  # 1/0. If 0, returns plotly json object, but doesnt plot
                ):
    """
    Makes a multiclass ROC plot
    """

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    N, n_classes = y_scores.shape
    if n_classes == 1:  # needed to avoid inverting when doing binary classification
        y_scores = -1*y_scores

    # calc ROC curves & AUC
    fpr = dict()
    tpr = dict()
    thresh = dict()
    thresh_txt = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], thresh[i] = sk.metrics.roc_curve(y_true == i, y_scores[:, i])
        roc_auc[i] = sk.metrics.auc(fpr[i], tpr[i])
        thresh_txt[i] = ['T=%.4f' % t for t in thresh[i]]

    # make traces
    
    traces = []
    [traces.append(go.Scatter(y=tpr[i], x=fpr[i], name=labels[i] + '. AUC= %.2f' % (roc_auc[i]), text=thresh_txt[i],
                              line={'width': 1})) for i in range(n_classes)]
    traces += [go.Scatter(y=[0, 1], x=[0, 1], name='Random classifier', line={'width': 1, 'dash': 'dot'})]

    # make layout
    layout = go.Layout(title=title,
                       xaxis={'title': 'FPR'},
                       yaxis={'title': 'TPR'},
                       legend=dict(x=1),
                       hovermode='closest',
    )

    fig = go.Figure(data=traces, layout=layout)

    if plot:
        plotfunc = pyo.iplot if in_notebook() else pyo.plot
        plotfunc(fig)
    else:
        return fig


def plotMultiPR(y_true,        # list of true labels
                    y_scores,   # array of scores for each class of shape [n_samples, n_classes]
                    title = 'Multiclass PR Plot',
                    labels = None, # list of labels for each class
                    plot=True,  # 1/0. If 0, returns plotly json object, but doesnt plot
                ):
    """
    Makes a multiclass ROC plot
    """

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    N, n_classes = y_scores.shape
    if n_classes == 1:  # needed to avoid inverting when doing binary classification
        y_scores = -1*y_scores

    # calc ROC curves & AUC
    precision = dict()
    recall = dict()
    pr_auc = dict()
    thresh = dict()
    thresh_txt = dict()
    for i in range(n_classes):
        precision[i], recall[i], thresh[i] = sk.metrics.precision_recall_curve(y_true == i, y_scores[:, i])
        #average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])
        #pr_auc[i] = sk.metrics.auc(precision[i], recall[i])
        pr_auc[i] = 1
        thresh_txt[i] = ['T=%.4f' % t for t in thresh[i]]

    # make traces
    traces = []
    [traces.append(go.Scatter(y=precision[i], x=recall[i], name=labels[i] + '. AUC= %.2f' % (pr_auc[i]), 
                        text=thresh_txt[i], line={'width': 1})) for i in range(n_classes)]

    # make layout
    layout = go.Layout(title=title,
                       xaxis={'title': 'Precision = P(yp=y | yp=1)'},
                       yaxis={'title': 'Recall = TPR = P(yp=y | y=1)'},
                       legend=dict(x=1),
                       hovermode='closest',
    )

    fig = go.Figure(data=traces, layout=layout)

    if plot:
        plotfunc = pyo.iplot if in_notebook() else pyo.plot
        plotfunc(fig)
    else:
        return fig

def plotConfusionMatrix(y_true, # list of true labels
                        y_pred, # list of predicted labels
                        title = 'Confusion Matrix',
                        labels = None, # list of labels for each class
                        binarized = None, # if int/str then makes 1vsAll confusion matrix of that class
                        add_totals = True, # whether to add an extra row for class totals
                        plot = True, # 1/0. If 0, returns plotly json object, but doesnt plot
                        fontsize=18,
                ):
    """
    Plots either a full or binarized confusion matrix
    """

    n_classes = len(labels) if labels is not None else len(np.unique(y_true))

    conf_matrix = sk.metrics.confusion_matrix(y_true, y_pred, labels=range(n_classes))

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
        labels = ['T','F']

    labels = ['['+x+']' if len(x)==1 else x for x in labels]    #needed for stupid plotly bug

    # adds an extra row for matrix totals
    if add_totals:
        conf_matrix = np.vstack((conf_matrix, np.sum(conf_matrix, 0)))
        conf_matrix = np.hstack((conf_matrix, np.atleast_2d(np.sum(conf_matrix, 1)).T ))
        labels = labels + ['TOTAL']

    # shorten labels
    labels_short = [x[:10] if type(x) == str else x for x in labels]

    fig = ff.create_annotated_heatmap(conf_matrix, x=labels_short, y=labels_short, colorscale='Viridis')
    fig.layout.yaxis.autorange = 'reversed'
    fig.layout.yaxis.title = 'True'
    fig.layout.xaxis.title = 'Predicted (Total accuracy = %.3f%%)' % perc(y_true==y_pred)
    fig.layout.xaxis.titlefont.size = fontsize
    fig.layout.yaxis.titlefont.size = fontsize
    fig.layout.xaxis.tickfont.size = fontsize - 2
    fig.layout.yaxis.tickfont.size = fontsize - 2
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = fontsize-3

    if plot:
        plotfunc = pyo.iplot if in_notebook() else pyo.plot
        plotfunc(fig)
    else:
        return fig
