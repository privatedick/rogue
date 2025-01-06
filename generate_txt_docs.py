import os
import subprocess
import sys


def install_dependencies():
    """Install necessary dependencies."""
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "sphinx", "sphinx_rtd_theme"]
    )


def create_rst_files():
    """Create or replace .rst files with the required content."""
    rst_files = {
        "index.rst": """
Welcome to Your Project's documentation!
========================================

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
        "installation.rst": """
Installation
============

Innehåll för installationsanvisningar...
""",
        "usage.rst": """
Usage
=====

Innehåll för användningsanvisningar...
""",
        "api.rst": """
API
===

Innehåll för API-dokumentation...
""",
        "core.rst": """
Core
====

Dokumentation för Core-modulen...
""",
        "modules.rst": """
Modules
=======

Dokumentation för Modules-modulen...
""",
        "tools.rst": """
Tools
=====

Dokumentation för Tools...
""",
    }

    for file_name, content in rst_files.items():
        with open(os.path.join("docs/source", file_name), "w") as file:
            file.write(content.strip())


def generate_restructuredtext_files():
    """Generate ReStructuredText files from source code."""
    subprocess.check_call(["sphinx-apidoc", "-o", "docs/source/", "../src/"])


def build_text_documentation():
    """Build text documentation with Sphinx."""
    subprocess.check_call(
        ["sphinx-build", "-b", "text", "docs/source", "docs/build/text"]
    )


def main():
    """Main function to orchestrate the documentation generation."""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    install_dependencies()
    create_rst_files()
    generate_restructuredtext_files()
    build_text_documentation()
    print("Documentation has been generated in text format.")


if __name__ == "__main__":
    main()
