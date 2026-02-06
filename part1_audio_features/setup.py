from setuptools import setup, find_packages

setup(
    name="part1_audio_features",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "librosa>=0.9.2",
        "torchaudio>=2.0",
        "torch>=2.0.0",
        "transformers>=4.30",
        "soundfile>=0.12",
        "pydub>=0.25",
        "praat-parselmouth>=0.4.0",
        "ffmpeg-python>=0.2.0",
    ],
    description="Part 1: Audio Ingestion & Feature Engineering for AI Voice Detection",
    python_requires=">=3.10",
)
