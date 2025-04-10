from multiprocessing import Process
import numpy as np
import io
from base64 import b64encode
import dash
from dash import dcc
from dash import html
# import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import json
import pickle
from plotly_scientific_plots.plotly_misc import jsonify


def create_html_download_button(figs, file_name="plotly_graph", button_name="Download as HTML"):
    """
    Creates a download button for Plotly figures
    """
    # Convert from dict to Plotly figs as needed
    plotly_figs = []
    for fig in figs:
        if isinstance(fig, dict):
            plotly_figs.append(go.Figure(fig))
        else:
            plotly_figs.append(fig)
    figs = plotly_figs

    # Handle multiple figures
    if isinstance(figs, list) and len(figs) > 1:
        figs = [fig for fig in figs if fig is not None]
        main_buffer = io.StringIO()
        outputs = []

        # Write first figure with full HTML
        _buffer = io.StringIO()
        figs[0].write_html(_buffer, full_html=True, include_plotlyjs='cdn')
        outputs.append(_buffer)

        # Write remaining figures as divs
        for fig in figs[1:]:
            _buffer = io.StringIO()
            fig.write_html(_buffer, full_html=False)
            outputs.append(_buffer)

        main_buffer.write(''.join([i.getvalue() for i in outputs]))
    else:
        # Handle single figure
        main_buffer = io.StringIO()
        if isinstance(figs, list):
            figs[0].write_html(main_buffer, include_plotlyjs='cdn')
        else:
            figs.write_html(main_buffer, include_plotlyjs='cdn')

    # Convert to base64
    html_bytes = main_buffer.getvalue().encode()
    encoded = b64encode(html_bytes).decode()

    # Create download button
    download_html = html.A(
        button_name,
        href="data:text/html;base64," + encoded,
        download=file_name + ".html",
        style={
            'background-color': '#4CAF50',
            'border': 'none',
            'color': 'white',
            'padding': '10px 20px',
            'text-align': 'center',
            'text-decoration': 'none',
            'display': 'inline-block',
            'font-size': '16px',
            'margin': '4px 2px',
            'cursor': 'pointer',
            'border-radius': '4px'
        }
    )

    return download_html

###Dash wrappers
def dashSubplot(plots,
                min_width=18,  # min width of column (in %). If more columns, scrolling is enabled
                max_width=50,  # max width of column (in %).
                indiv_widths=None,  # can specify list of individual column widths
                title=''        # str or list of strs
                ):

    if isinstance(title, str):
        title = [title]

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

    title = sum([[i, html.Br()] for i in title], [])[:-1]

    col_style = [{'width': str(col_width[i]) + '%',
             'display': 'inline-block',
             'vertical-align': 'top',
             'margin-right': '25px'} for i in range(Ncol)]

    plot_divs = html.Div([html.Div(plots[i], style=col_style[i]) for i in range(Ncol)])
    title_div = html.H3(title)
    layout = html.Div(html.Div([title_div, plot_divs]),
                      style={'margin-right': '0px', 'position': 'absolute', 'width': '100%'})

    return layout


def horizontlDiv(dashlist,
                 id='L',    # either single element or list. If single, id of html divs will be this + # (ie 'L1', 'L2', etc..
                 width=50): #either total width or list of indiv widths
    ''' Creates a horizontal Div line '''
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
                 for c, i in enumerate(dashlist)]
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
                        host=None,    # set to '0.0.0.0' to run as a server. Default val is None (localhost)
                        title='',
                        port=8050,
                        add_download_button=True,
                        download_filename="plotly_dashboard",
                        run=True,
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
            if f == []:
                g_col += [[]]
            elif isinstance(f, dash.development.base_component.Component):
                g_col += [f]
            else:
                if 'meta' in f['layout'] and f['layout']['meta'] is not None:
                    id = f['layout']['meta']
                else:
                    id = ['row_%d_col_%d' % (r_num, c_num)]
                g_col += [dcc.Graph(figure=f, id=id[0])]
        graphs += [g_col]

    app = dash.Dash()

    # Create layout with optional download button
    dashboard_layout = dashSubplot(graphs, min_width, max_width, indiv_widths, title)

    if add_download_button:
        # Extract the original figures before conversion to dash components
        original_figs = []
        for col in figs:
            for fig in col:
                if fig != [] and not isinstance(fig, dash.development.base_component.Component):
                    original_figs.append(fig)

        # Create the download button for all figures
        download_button = create_html_download_button(
            original_figs,
            file_name=download_filename,
            button_name="Download Dashboard as HTML"
        )

        # Add download button to layout
        app.layout = html.Div([
            html.Div(download_button, style={'margin': '10px'}),
            dashboard_layout
        ])
    else:
        app.layout = dashboard_layout

    if run:
        app.run_server(port=port, debug=False, host=host)

    return app


def startDashboard(figs,
                   parr=False,  # T/F. If True, will spin seperate python process for Dash webserver
                   save=None,  # either None or save_path
                   **kwargs    # additional optional params for startDashboardSerial (e.g. min_width)
                  ):

    # First convert to json format to allow pkling for multiprocessing
    figs_dictform = jsonify(figs)

    # save if nessesary (currently only saves in pkl format)
    if save is not None and not False:
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