import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MED3DRESNET",  # Replace with your own username
    version="1.0",
    author="Timothy Burt, Luben Popov, Yuan Zi",
    author_email="taburt@uh.edu",
    description="Cardiovascular risk computed via Deep Learning (DL) on thoracic CT scans (COSC 7373 Team 1 project F19 UH)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lpopov101/ACVProject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)