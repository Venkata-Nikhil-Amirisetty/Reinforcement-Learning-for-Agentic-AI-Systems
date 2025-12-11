from setuptools import setup, find_packages

setup(
    name="rl-agentic-systems",
    version="0.1.0",
    description="Reinforcement Learning for Agentic AI Systems",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "gym>=0.26.0",
        "tensorboard>=2.13.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    python_requires=">=3.8",
)

