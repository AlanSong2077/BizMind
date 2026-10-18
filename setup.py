from setuptools import setup, find_packages

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="bizmind-ml",
    version="0.1.0",
    packages=find_packages(include=["bizmind", "bizmind.*"]),
    install_requires=requirements,
    python_requires=">=3.11",
    description="Business-driven, self-iterating local machine learning engineering",
    author="AlanSong2077",
    url="https://github.com/AlanSong2077/BizMind",
    license="MIT",
)
