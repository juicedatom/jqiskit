import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="jqiskit-juicedatom",
    version="0.0.1",
    author="Joshua Manela",
    author_email="juicedatom@gmail.com",
    description="A small example framework for learning quantum computing.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/juicedatom/jqiskit",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'sympy',
        'numpy',
    ],
    python_requires='>=3.7',
)
