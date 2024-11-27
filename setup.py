from setuptools import setup, find_packages

setup(
    name="probpy",
    version="0.3.0",
    description="A library for probability-related functionalities.",
    author="Frederik Danielsen",
    author_email="danielsen.contact@gmail.com",
    url="https://github.com/FrederikDanielsen/probpy",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20",      
        "scipy>=1.7",
        "matplotlib>=3.4",
        "networkx>=2.5",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)