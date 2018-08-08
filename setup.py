from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='plotly-scientific-plots',
    url='https://github.com/rsandler00/plotly-scientific-tools',
    author='Roman Sandler',
    author_email='rsandler00@gmail.com',
    packages=['plotly_scientific_plots'],
    install_requires=['numpy', 'scipy', 'plotly', 'colorlover',
                      'dash', 'dash_core_components', 'dash_html_components'],
    version='0.1.0.4',
    license='MIT',
    description='Python package extending plotly for scientific computing and visualization',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)