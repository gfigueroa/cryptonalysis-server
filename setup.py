__author__ = 'Gerardo Figueroa'
__description__ = 'Crypto predictor server'

from setuptools import setup


with open('requirements.txt') as f:
    requirements = f.readlines()

with open('requirements.testing.txt') as f:
    requirements_testing = f.readlines()


setup(
    name='cryptonalysis_server',
    version='0.1.3.feature.predictor.states',
    packages=['cryptonalysis'],
    include_package_data=True,
    license='',
    author=__author__,
    author_email='gerardo_ofc@yahoo.com',
    description=__description__,
    install_requires=requirements,
    tests_require=requirements_testing,
    test_suite='tests'
)
