#!/usr/bin/env python
#
# Trading UNAM documentation build configuration file.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

import trading_unam

# -- General configuration ---------------------------------------------

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.viewcode']

templates_path = ['_templates']

source_suffix = '.rst'

master_doc = 'index'

# General information about the project.
project = 'Trading UNAM'
copyright = "2024, Equipo Python FE"
author = "Equipo Python FE"

version = trading_unam.__version__
release = trading_unam.__version__

language = None

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

pygments_style = 'sphinx'

todo_include_todos = False


# -- Options for HTML output -------------------------------------------

html_theme = 'alabaster'

html_static_path = ['_static']


# -- Options for HTMLHelp output ---------------------------------------

htmlhelp_basename = 'trading_unamdoc'


# -- Options for LaTeX output ------------------------------------------

latex_elements = {}

latex_documents = [
    (master_doc, 'trading_unam.tex',
     'Trading UNAM Documentation',
     'Equipo Python FE', 'manual'),
]


# -- Options for manual page output ------------------------------------

man_pages = [
    (master_doc, 'trading_unam',
     'Trading UNAM Documentation',
     [author], 1)
]


# -- Options for Texinfo output ----------------------------------------

texinfo_documents = [
    (master_doc, 'trading_unam',
     'Trading UNAM Documentation',
     author,
     'trading_unam',
     'Trading UNAM - Paquete de optimización de portafolios y analítica financiera.',
     'Miscellaneous'),
]
