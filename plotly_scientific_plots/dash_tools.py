from multiprocessing import Process
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import json
import pickle
from plotly_scientific_plots.plotly_misc import jsonify



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


def startDashboardSerial(figs,
                        min_width = 18,  # min width of column (in %). If more columns, scrolling is enabled
                        max_width = 50,  # max width of column (in %).
                        indiv_widths = None,
                        port = 8050
                  ):
    """
    This starts the dash layout
    :param figs: a nested list of plotly figure objects. Each outer list is a column in the dashboard, and each
                        element within the outer list is a row within that column.
    :return:
    """



    # convert plotly fig objects to dash graph objects
    graphs = []
    for c_num, col in enumerate(figs):
        g_col = []
        for r_num, f in enumerate(col):
            if f != []:
                g_col += [dcc.Graph(figure=f, id='row_%d_col_%d' % (r_num, c_num))]
            else:
                g_col += [[]]
        graphs += [g_col]

    app = dash.Dash()
    app.layout = dashSubplot(graphs, min_width, max_width, indiv_widths)
    app.run_server(port=port, debug=False)

    return None

def startDashboard(figs,
                   parr=False,  # T/F. If True, will spin seperate python process for Dash webserver
                   save=None,  # either None or save_path
                   **kwargs,    # additional optional params for startDashboardSerial (e.g. min_width)
                  ):

    # First convert to json format to allow pkling for multiprocessing
    figs_dictform = jsonify(figs)

    # save if nessesary (currently only saves in pkl format)
    if save is not None:
        # Note, can also use _dump_json, but its about 3x bigger filesize
        _dump_pkl(figs_dictform, save)

    if parr:
        p = Process(target=startDashboardSerial, args=(figs_dictform,), kwargs=kwargs)
        p.start()
        return p
    else:
        startDashboardSerial(figs_dictform, **kwargs)
        return None



def _dump_pkl(obj, file_path):
    ''' Saves a pkl file '''
    with open(file_path, 'wb') as dfile:
        pickle.dump(obj, dfile, protocol = 2)

def _dump_json(obj, file_path):
    ''' Saves a json file '''
    with open(file_path, 'w') as dfile:
        json.dump(obj, dfile, indent = 4)