from setuptools import setup, find_packages

setup(
    name="crar-pytorch",
    version="0.1",
    packages=find_packages(),
    package_data={
        # If any package contains *.txt or *.yaml files, include them:
        "": ["*.txt", "*.yaml"],
    },
)
