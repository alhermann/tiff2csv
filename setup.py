import setuptoolswith open("README.md", "r") as fh:    long_description = fh.read()setuptools.setup(    name="tiff2csv", # Replace with your own username    version="0.0.2",    author="Alexander Hermann",    author_email="alexander.hermann@hzg.de",    description="A lightweight python package for processing .tif image data files.",    long_description=long_description,    long_description_content_type="text/markdown",    url="https://github.com/alhermann/TIFF2Vox.git",    packages=setuptools.find_packages(),    classifiers=[        "Programming Language :: Python :: 3",        "License :: OSI Approved :: MIT License",        "Operating System :: OS Independent",    ],    python_requires='>=3.6',)