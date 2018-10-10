import numpy as np
import copy

# import sklearn
import sklearn as sk
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.ensemble
import sklearn.metrics

# import plotly
import plotly.graph_objs as go
import plotly.figure_factory as ff
# import internal functions
from plotly_scientific_plots.plotly_misc import plotOut

perc = lambda x: np.sum(x)/len(x)*100


def plotMultiROC(y_true,        # list of true labels
                    y_scores,   # array of scores for each class of shape [n_samples, n_classes]
                    title = 'Multiclass ROC Plot',
                    labels = None, # list of labels for each class
                    threshdot = None,
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

    labels = [str(x) for x in labels]  # convert labels to str

    # make traces
    traces = []
    [traces.append(go.Scatter(y=tpr[i], x=fpr[i], name=labels[i] + '. AUC= %.2f' % (roc_auc[i]), text=thresh_txt[i],
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

    return plotOut(fig, plot)


def plotMultiPR(y_true,        # list of true labels
                    y_scores,   # array of scores for each class of shape [n_samples, n_classes]
                    title = 'Multiclass PR Plot',
                    labels = None, # list of labels for each class
                    threshdot=None, # whether to plot a dot @ the threshold
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

    labels = [str(x) for x in labels]  # convert to str

    # make traces
    traces = []
    [traces.append(go.Scatter(y=precision[i], x=recall[i], name=labels[i] + '. AUC= %.2f' % (pr_auc[i]), 
                        text=thresh_txt[i], legendgroup=str(i), line={'width': 1})) for i in range(n_classes)]

    if threshdot is not None:
        for i in range(n_classes):
            c_indx = (np.abs(thresh[i]-threshdot)).argmin()
            traces += [go.Scatter(x=[recall[i][c_indx]]*2, y=[precision[i][c_indx]]*2, mode='markers',
                                  name='Threshold', legendgroup=str(i), showlegend=False)]

    # make layout
    layout = go.Layout(title=title,
                       xaxis={'title': 'Precision = P(y=1 | yp=1)'},   # 'Precision = P(yp=y | yp=1)'
                       yaxis={'title': 'Recall = TPR = P(yp=1 | y=1)'}, # 'Recall = TPR = P(yp=y | y=1)'
                       legend=dict(x=1),
                       hovermode='closest',
    )

    fig = go.Figure(data=traces, layout=layout)

    return plotOut(fig, plot)


def plotConfusionMatrix(y_true, # list of true labels
                        y_pred, # list of predicted labels
                        title = 'Confusion Matrix',
                        labels = None, # list of labels for each class
                        binarized = None, # if int/str then makes 1vsAll confusion matrix of that class
                        add_totals = True, # whether to add an extra row for class totals
                        plot = True, # 1/0. If 0, returns plotly json object, but doesnt plot
                        fontsize=18,    # axis font
                        norm='rows',     # how to norm matrix colors. either 'all'/'rows'/'columns'
                ):
    """
    Plots either a full or binarized confusion matrix
    """

    n_classes = len(labels) if labels is not None else len(np.unique(y_true))

    if labels is None:
        labels = ['C%d' % n for n in range(1, n_classes+1)]

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
        n_classes = 2

    labels = [str(x) for x in labels]   # convert to str
    labels = ['['+x+']' if len(x)==1 else x for x in labels]    #needed for stupid plotly bug

    # adds an extra row for matrix totals
    conf_matrix_tots =  copy.deepcopy(conf_matrix)
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
    color_mat[:-1,:-1] = norm_conf_matrix

    # hover text
    txt_format = '<b>Pred:</b> %s <br><b>True:</b> %s <br><b>Row norm:</b> %.3f%% <br><b>Col norm:</b> %.3f%%'
    htext = np.array([[txt_format % (labels[c], labels[r], row_percs[r,c]*100, col_percs[r,c]*100)
                       for c in range(n_classes)] for r in range(n_classes)])

    # Adjust Total rows
    if add_totals:
        totals_row_shading = .97    # range 0 to 1. 0=darkest, 1=lightest
        tot_val = np.min(norm_conf_matrix) + (np.max(norm_conf_matrix) - np.min(norm_conf_matrix))*totals_row_shading
        color_mat[-1, :] = tot_val
        color_mat[:, -1] = tot_val
        pred_tot_text = np.array(['<b>%% of Predictions:</b> %.2f%%' % x for x in pred_tots/sum(pred_tots)*100])
        true_tot_text = np.array([['<b>%% of True Data:</b> %.2f%%' % x] for x in true_tots[:-1]/sum(true_tots[:-1])*100]+[['Total Samples']])
        htext = np.hstack((np.vstack((htext, pred_tot_text)), true_tot_text))

    fig = ff.create_annotated_heatmap(color_mat, x=num_labels, y=num_labels, colorscale='Greys', annotation_text=conf_matrix_tots)

    acc = perc(np.squeeze(y_true) == np.squeeze(y_pred))

    fig.layout.yaxis.title = 'True'
    fig.layout.xaxis.title = 'Predicted (Total accuracy = %.3f%%)' % acc
    fig.layout.xaxis.titlefont.size = fontsize
    fig.layout.yaxis.titlefont.size = fontsize
    fig.layout.xaxis.tickfont.size = fontsize - 2
    fig.layout.yaxis.tickfont.size = fontsize - 2
    fig.layout.showlegend = False
    # Add label text to axis values
    fig.layout.xaxis.tickmode = 'array'
    fig.layout.xaxis.range = [-.5, n_classes+.5]
    fig.layout.xaxis.tickvals = num_labels
    fig.layout.xaxis.ticktext = labels_short
    fig.data[0].hoverlabel.bgcolor = 'rgb(188,202,225)'

    # fig.layout.yaxis.autorange = 'reversed'
    fig.layout.yaxis.tickmode = 'array'
    fig.layout.yaxis.range = [n_classes+.5, -.5]
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
        # adjust font
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
