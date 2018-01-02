#!/usr/bin/env python

from distutils.core import setup

setup(name='Google Certificate Transparency',
      version='0.9',
      description='Certificate Transparency python client and monitor library',
      url='https://github.com/google/certificate-transparency',
      packages=['ct', 'ct.crypto', 'ct.crypto.asn1', 'ct.client'],
     )
