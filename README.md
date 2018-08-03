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

To see a full lsit of plotly-scientific-tools examples, go through the `examples.ipynb` 
in nbviewer by clicking [here][1]

Plots 2 overlapping normalized histograms, including overhead boxplots and data points.

Furthermore performs statistical testing to differentiate the two population samples.
Tests are:
* KS: Kolmogorov-Smirnov statistic on 2 samples.
* MW: Mann-Whitney rank test on two samples.
* T: T-test for the means of *two independent* samples of scores.

All tests are done via `scipy`

```python
psp.plot2Hists(data_source_1, data_source_2, names=['Data 1','Data 2'],
           normHist=True, title='Comparison of 2 Data Sources',
           KS=True, MW=True, T=True))
```
![plot2Hist_1](images/plot2Hist_1.png?raw=true "plot2Hist_1")

[1]: https://nbviewer.jupyter.org/github/rsandler00/plotly-scientific-plots/blob/master/examples.ipynb
 