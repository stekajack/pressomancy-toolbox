from setuptools import setup

setup(
    name="pressomancy-toolbox",
    version="0.0.1",
    author="Deniz Mostarac",
    author_email="deniz.mostarac@ed.ac.uk",
    description="wip",
    install_requires=[
        "pressomancy @ git+https://github.com/stekajack/pressomancy.git",
        "matplotlib",
        "scipy",
        "pandas",
        "igraph",
    ],
    python_requires='>=3.6',  # Specify the Python version requirement
)
