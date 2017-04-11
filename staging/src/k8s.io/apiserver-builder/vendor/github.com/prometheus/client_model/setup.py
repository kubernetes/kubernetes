#!/usr/bin/python

from setuptools import setup

setup(
    name = 'prometheus_client_model',
    version = '0.0.1',
    author = 'Matt T. Proud',
    author_email = 'matt.proud@gmail.com',
    description = 'Data model artifacts for the Prometheus client.',
    license = 'Apache License 2.0',
    url = 'http://github.com/prometheus/client_model',
    packages = ['prometheus', 'prometheus/client', 'prometheus/client/model'],
    package_dir = {'': 'python'},
    requires = ['protobuf(==2.4.1)'],
    platforms = 'Platform Independent',
    classifiers = ['Development Status :: 3 - Alpha',
                   'Intended Audience :: Developers',
                   'Intended Audience :: System Administrators',
                   'License :: OSI Approved :: Apache Software License',
                   'Operating System :: OS Independent',
                   'Topic :: Software Development :: Testing',
                   'Topic :: System :: Monitoring'])
