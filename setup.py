import setuptools
from pathlib import Path

setuptools.setup(
    name="old-fashioned-nlp",
    version="0.0.1",
    author="Chenghao Mou",
    author_email="mouchenghao@gmail.com",
    description="Sklearn base nlp models",
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    py_modules = ['old_fashioned_nlp'],
    url="<https://github.com/authorname/templatepackage>",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)