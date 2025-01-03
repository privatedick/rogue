import os

# Sätt grundläggande sökvägar
docs_dir = "docs"
source_dir = os.path.join(docs_dir, "source")
build_dir = os.path.join(docs_dir, "build")
text_dir = os.path.join(build_dir, "text")

# Skapa nödvändiga kataloger
os.makedirs(source_dir, exist_ok=True)
os.makedirs(build_dir, exist_ok=True)
os.makedirs(text_dir, exist_ok=True)

# Definiera innehållet för varje .rst-fil
rst_files = {
    "index.rst": """Welcome to Your Project's documentation!
============================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   api
   core
   modules
   tools

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
""",

    "api.rst": """API
===

Innehåll för API-dokumentation...
""",

    "core.rst": """Core
====

Dokumentation för Core-modulen...
""",

    "installation.rst": """Installation
============

Innehåll för installationsanvisningar...
""",

    "modules.rst": """Modules
=======

Dokumentation för Modules-modulen...
""",

    "tools.rst": """Tools
=====

Dokumentation för Tools...
""",

    "usage.rst": """Usage
=====

Innehåll för användningsanvisningar...
"""
}

# Skapa .rst-filer med innehållet
for filename, content in rst_files.items():
    file_path = os.path.join(source_dir, filename)
    with open(file_path, "w") as f:
        f.write(content)
    print(f"Skapade fil: {file_path}")

# Skapa conf.py för Sphinx
conf_py_content = """
# -- Project information -----------------------------------------------------
project = 'Your Project Name'
copyright = '2025, Your Name'
author = 'Your Name'

# -- General configuration ---------------------------------------------------
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon']

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'alabaster'

# -- Options for LaTeX output ------------------------------------------------
latex_engine = 'xelatex'
"""

conf_py_path = os.path.join(source_dir, "conf.py")
with open(conf_py_path, "w") as f:
    f.write(conf_py_content)
print(f"Skapade fil: {conf_py_path}")

# Skapa Makefile för att bygga dokumentationen
makefile_content = """
.PHONY: help clean text

help:
\t@echo "Generera textdokumentation med 'make text'"

clean:
\trm -rf build/*

text:
\tsphinx-build -b text source build/text
"""

makefile_path = os.path.join(docs_dir, "Makefile")
with open(makefile_path, "w") as f:
    f.write(makefile_content)
print(f"Skapade fil: {makefile_path}")

# Skapa .bat-fil för Windows (om du använder Windows)
make_bat_content = """
@echo off
call sphinx-build -b text source build/text
"""

make_bat_path = os.path.join(docs_dir, "make.bat")
with open(make_bat_path, "w") as f:
    f.write(make_bat_content)
print(f"Skapade fil: {make_bat_path}")
