import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

project = 'Your Project Name'
author = 'Your Name or Organization'
release = '0.1'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser',  # Lägg till detta för att hantera Markdown-filer
]

templates_path = ['_templates']
html_static_path = ['_static']
