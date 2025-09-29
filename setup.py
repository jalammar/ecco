#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import re
from pathlib import Path

from setuptools import find_packages
from setuptools import setup


def read(file):
    return Path(file).read_text(encoding='utf-8')


setup(
    name='ecco',
    version='0.1.2',
    license='BSD-3-Clause',
    description='Visualization tools for NLP machine learning models.',
    long_description='%s\n%s' % (
        re.compile('^.. start-badges.*^.. end-badges', re.M | re.S).sub('', read('README.rst')),
        re.sub(':[a-z]+:`~?(.*?)`', r'``\1``', read('CHANGELOG.rst'))
    ),
    author='Jay Alammar',
    author_email='alammar@gmail.com',
    url='https://github.com/jalammar/ecco',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[p.stem for p in Path('src').glob('*.py')],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Utilities',
    ],
    project_urls={
        'Changelog': 'https://github.com/jalammar/ecco/blob/master/CHANGELOG.rst',
        'Issue Tracker': 'https://github.com/jalammar/ecco/issues',
    },
    keywords=[
        'Natural Language Processing', 'Explainable AI', 'keyword3',
    ],
    python_requires='>=3.7',
    install_requires=[
        "transformers<4.47",
        "seaborn>=0.13",
        "scikit-learn>=0.25",
        "PyYAML>=6",
        "captum>=0.4"
    ],
    extras_require={
        "dev": [
            "pytest>=7",
        ],
    },
    entry_points={
        'console_scripts': [
            'ecco = ecco.cli:main',
        ]
    },
    dependency_links=[
        "https://download.pytorch.org/whl/torch_stable.html"
    ]
)
