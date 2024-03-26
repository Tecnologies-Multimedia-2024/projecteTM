from pathlib import Path
from setuptools import setup

# Read the contents of the requirements file:
requirements = Path('requirements.txt').read_text().strip().split('\n')

setup(
    name='tmproject',
    version='0.1.0',
    packages=['tmproject'],
    install_requires=requirements,
    url='',
    license='',
)
