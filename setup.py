import setuptools

__version__ = "0.0.2"

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="strong_augment",
    version=__version__,
    author="jopo666",
    author_email="jopo@birdlover.com",
    description="StrongAugment implementation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jopo666/StrongAugment",
    packages=setuptools.find_packages(
        include=["strong_augment", "strong_augment.*"]
    ),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    keywords="augmentation",
    python_requires=">=3.9",
    install_requires=[
        "torchvision>=0.12",
        "Pillow>=9.1",
    ],
)
