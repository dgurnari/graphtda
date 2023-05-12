import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

print(os.getcwd())


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "graphTDA"
copyright = "2023, Davide Gurnari"
author = "Davide Gurnari"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = []

autoclass_content = "both"

autodoc_mock_imports = [
    'numpy',
    "pandas",
    "networkx",
    "matplotlib",
    "sklearn",
    "scipy",
    "tqdm",
    'pyrivet',
    'ot'
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
# html_theme = 'classic'

html_static_path = ["_static"]
