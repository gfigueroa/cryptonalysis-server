from setuptools import setup

setup(
    name='CryptonalysisServer',
    version='0.1.0',
    packages=['cryptonalysis'],
    include_package_data=True,
    install_requires=[
        'flask',
    ],
    tests_require=[
        'nose==1.3.7'
    ],
    test_suite='nose.collector'
)
