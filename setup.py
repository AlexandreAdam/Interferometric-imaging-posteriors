from setuptools import setup, find_packages

setup(
    name="ipa",
    version="0.1",
    description="Package intended to perform interferometric-imaging and posterior estimation at the pixel level "
                "with generative modeling by estimating the gradient of data distributions.",
    packages=find_packages(),
    python_requires=">=3.7"
)
