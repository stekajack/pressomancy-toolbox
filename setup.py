from setuptools import setup, find_packages

setup(
    name="pressomancy-toolbox",
    version="0.0.1",
    packages=find_packages(),
    author="Deniz Mostarac",
    author_email="deniz.mostarac@ed.ac.uk",
    description="wip",
    install_requires=[
        "numpy",
        "h5py",
        "igraph",
        "vg",
        "pressomancy",
    ],
    python_requires='>=3.10',  # Specify the Python version requirement
)
