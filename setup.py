#!/usr/bin/env python

"""Package setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = ["numpy", "matplotlib", "gsw", "scipy", "pyyaml"]

setup(
    name='niskine',
    author="Amy & Gunnar",
    license="MIT License",
    url='https://github.com/modscripps/niskine',
    version='0.1.0',
    # Requirements
    python_requires='>=3.8',
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
    ],
    description="NISKINe data analysis package",
    long_description=readme,
    include_package_data=True,
    packages=find_packages(include=['niskine', 'niskine.*'], exclude=["*.tests"]),
    zip_safe=False,
)
