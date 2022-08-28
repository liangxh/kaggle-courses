"""
官方文档: https://setuptools.pypa.io/en/latest/userguide/quickstart.html
"""

from setuptools import setup, find_packages


setup(
    name="mlbox",
    version="1.0",
    author="kabu",
    author_email="kabu@email.com",
    description="Machine Learning Toolbox",
    packages=find_packages(where="src", exclude=[]),
    package_dir={"": "src"},
    include_package_data=True
)
