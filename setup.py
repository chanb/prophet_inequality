from setuptools import setup, find_packages

setup(
    name="prophet_inequality",
    description="Prophet Inequality Implementation",
    version="0.1",
    python_requires=">=3.10",
    install_requires=["matplotlib", "numpy", "scipy", "torch", "tqdm"],
    packages=find_packages(),
    include_package_data=True,
    license=None,
    url="http://github.com/chanb/cmput676_winter2022",
)
