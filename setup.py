from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="Set-Output",
    version="0.0.1",
    author="L.A.C.E. Lab",
    description="Set to set prediction.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/etalvitie/set-output",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Unlicense License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)