from setuptools import setup, find_packages

setup(
    name="part2_detection",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "scikit-learn>=1.2",
        "torch>=2.0.0",
        "torchmetrics>=0.11",
        "numpy>=1.24",
    ],
    description="Part 2: Detection Model & Expalinability for AI Voice Detection",
    python_requires=">=3.10",
)
