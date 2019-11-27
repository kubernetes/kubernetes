#!/usr/bin/env python

# Copyright 2016 The Kubernetes Authors.
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

import boilerplate
import unittest
from io import StringIO
import os
import sys

class TestBoilerplate(unittest.TestCase):
  """
  Note: run this test from the hack/boilerplate directory.

  $ python -m unittest boilerplate_test
  """

  def test_boilerplate(self):
    os.chdir("test/")

    class Args(object):
      def __init__(self):
        self.filenames = []
        self.rootdir = "."
        self.boilerplate_dir = "../"
        self.verbose = True

    # capture stdout
    old_stdout = sys.stdout
    sys.stdout = StringIO.StringIO()

    boilerplate.args = Args()
    ret = boilerplate.main()

    output = sorted(sys.stdout.getvalue().split())

    sys.stdout = old_stdout

    self.assertEquals(
        output, ['././fail.go', '././fail.py'])
