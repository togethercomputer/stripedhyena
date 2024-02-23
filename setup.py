from setuptools import find_packages, setup

with open("README.md") as f:
    readme = f.read()

with open("requirements.txt") as f:
    requirements = f.read().split("\n")

setup(
    name="stripedhyena",
    version="0.2.1",
    description="Model and inference code for beyond Transformer architectures",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Michael Poli",
    url="http://github.com/togethercomputer/stripedhyena",
    license="Apache-2.0",
    packages=find_packages(where="stripedhyena"),
    install_requires=requirements,
)
