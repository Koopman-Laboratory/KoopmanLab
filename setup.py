import io
import setuptools

with io.open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(
    name="koopmanlab",
    version="1.0.0",
    author="Wei Xiong, Tian Yang",
    author_email="xiongw21@mails.tsinghua.edu.cn",
    description="A library for Koopman Neural Operator with Pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Koopman-Laboratory/KoopmanLab",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL-3.0 License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8.5',
    install_requires = [
        'torch>=1.10',
        'torchvision>=0.13.1',
        'matplotlib>=3.3.2',
        'numpy>=1.14.5',
        'einops==0.5.0',
        'timm==0.6.11',
        'scipy==1.7.3',
        'h5py==3.7.0',
    ]
)
