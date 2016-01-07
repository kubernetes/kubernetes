#!/usr/bin/env python

# Copyright 2015 The Kubernetes Authors All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import requests
import socket

from urlparse import urlparse


def CheckServiceAddress(address):
  hostname = urlparse(address).hostname
  service_address = socket.gethostbyname(hostname)
  print service_address


def GetServerResponse(address):
  print 'Send request to:', address
  response = requests.get(address)
  print response
  print response.content


def Main():
  parser = argparse.ArgumentParser()
  parser.add_argument('address')
  args = parser.parse_args()
  CheckServiceAddress(args.address)
  GetServerResponse(args.address)


if __name__ == "__main__":
  Main()
