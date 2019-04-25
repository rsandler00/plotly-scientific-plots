
from plotly_scientific_plots.plotly_misc import *
from plotly_scientific_plots.plotly_plot_tools import *
from plotly_scientific_plots.plotly_ML import *
from plotly_scientific_plots.misc_computational_tools import *
from plotly_scientific_plots.dash_tools import *
from plotly_scientific_plots.plot_subcomponents import *

import sys
if 'pandas' in sys.modules:
    from plotly_scientific_plots.plotly_pandas import *

# this command enables visualizations in jupyter notebook
if in_notebook():
    pyo.init_notebook_mode(connected=True)