#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

NAME = 'siren-noise-suppression'
DESCRIPTION = 'Dictionary class with advanced functionality'
URL = 'https://github.com/ShuYuHuang/Siren_Noise_Supression'
EMAIL = 'b123767195@gmail.com'
AUTHOR = 'ShuYuHuang'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.1.0'
REQUIRED = [
    # 'requests', 'maya', 'records',
]

    
EXTRAS = {
    # 'fancy feature': ['django'],
}

here = os.path.abspath(os.path.dirname(__file__))
print(here)
about = {}
# project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
project_slug="sirenns"
with open(os.path.join(here, project_slug, '__version__.py')) as f:
    exec(f.read(), about)    
if __name__ == '__main__':
    setup(
        name=NAME,
        version=about['__version__'],
        author=AUTHOR,
        author_email=EMAIL,
        python_requires=REQUIRES_PYTHON,
        include_package_data=True,
        license='MIT',
        url=URL,
        classifiers=[
            # Trove classifiers
            # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: Implementation :: CPython',
            'Programming Language :: Python :: Implementation :: PyPy'
        ],
        packages=find_packages(exclude=["tests",
                                        "data",
                                        "*.tests",
                                        "*.tests.*",
                                        "tests.*"])
    )