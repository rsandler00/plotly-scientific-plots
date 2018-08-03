# plotly-scientific-plots

This python library is meant to augment the plotly and dash visualization libraries.
It is designed to facilitate rapid and beautiful data visualizations for research scientists and data scientists.

Its advantages over naive plotly are:
* One-line commands to make plots
* Integrated scatistical testing above plots
* Expanded plot types (such as confusion amtrices, ROC plots)
* more 'Matlab-like' interface for those making the Matlab --> python transition

## Requirements and installation

Required packages:
* numpy
* scipy
* plotly
* colorlover
* dash
* dash_core_components
* dash_html_components

To install, simply use `pip install plotly-scientific-plots`

To import use `import plotly_scientific_plots as psp`

## Examples and Usage

Plotly's key strength is its ability to do interactive visualizations. 
For scientists, this allows novel ways of exploring data with mouse clicks and hovers.

To see a full lsit of plotly-scientific-tools examples and their descriptions, go through the `examples.ipynb` 
in nbviewer by clicking [here][1]

Below, are a limited set of examples to give the feel of how `psp` works:

##### Two dataset histograms:

```python
psp.plot2Hists(data_source_1, data_source_2, names=['Data 1','Data 2'],
            normHist=True, title='Comparison of 2 Data Sources',
            KS=True, MW=True, T=True)
```
![plot2Hist_1](images/plot2Hist_1.png?raw=true "plot2Hist_1")


##### Scatter + Contour Plot:

```python
psp.scatterHistoPlot(data_source_1, data_source_3, title='Contour of x_var & y_var', 
            xlbl='x_var label', ylbl='y_var label')
```
![plot2Hist_1](images/contour_and_scatter.png?raw=true "contour_and_scatter")


##### Multiple Dataset Correlations + Stats:

```python
psp.corrPlot([data_source_1, data_source_11, data_source_12], [data_source_3, data_source_31, 
            data_source_32], names=['datasetA', 'datasetB', 'datasetC'],addCorr=True, 
            addCorrLine=True, title='Correlation of x_var & y_var', xlbl='x_var label', 
            ylbl='y_var label')
```
![plot2Hist_1](images/corrPlot_multi.png?raw=true "corrPlot_multi")


##### Polar Plot

```python
psp.plotPolar([polar1], numbins=20, title='Polar Distribution')
```
![plot2Hist_1](images/polar1.png?raw=true "polar1")

[1]: https://nbviewer.jupyter.org/github/rsandler00/plotly-scientific-plots/blob/master/examples.ipynb
 