# setup.py
from setuptools import setup

setup(
    name='ift6163',
    version='0.1.0',
    packages=['ift6163'],
    license='MIT License',
    long_description=open('README.md').read(),
    install_requires=open('requirements.txt').read()
)