from setuptools import setup, find_packages

setup(
    name="drugdetector",
    version="0.3.1",
    packages=find_packages(),
    install_requires=[
        "accelerate", 
        "guidance",
        "llama-cpp-python"
        "transformers",
        "optimum",
    ],
    author="Fabrice Harel-Canada",
    author_email="fabricehc@cs.ucla.edu",
    description="A simple wrapper to support drug detection in medical texts.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/fabriceyhc/drugdetector",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
