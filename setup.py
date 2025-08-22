from setuptools import setup, find_packages

setup(#beta, ignore
    name="FRM",
    version="0.1.0",
    description="Quantitative tools and scripts for FRM Level 1 preparation",
    author="Baumann Bence",
    packages=find_packages(),  # This finds 'my_utils' and any other packages
    install_requires=[
        "yfinance",
        "pandas",
        "scipy",
        "numpy"
    ],
    python_requires=">=3.8",
)
