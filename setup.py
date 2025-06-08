from setuptools import setup, find_packages

setup(
    name="bhgy-cluster",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "scipy",
        "nltk",
        "sentence-transformers",
        "matplotlib",
        "seaborn",
        "python-dotenv",
        "pytest"
    ],
) 