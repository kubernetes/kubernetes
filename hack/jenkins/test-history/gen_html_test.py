#!/usr/bin/env python

# Copyright 2016 The Kubernetes Authors All rights reserved.
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

"""Tests for gen_html."""

import json
import os
import shutil
import tempfile
import unittest

import gen_html

TEST_DATA = {
  "test1":
      {"kubernetes-release": [{"build": 3, "failed": False, "time": 3.52},
                              {"build": 4, "failed": True, "time": 63.21}],
       "kubernetes-debug": [{"build": 5, "failed": False, "time": 7.56},
                            {"build": 6, "failed": False, "time": 8.43}],
      },
  "test2":
      {"kubernetes-debug": [{"build": 6, "failed": True, "time": 3.53}]},
}

class GenHtmlTest(unittest.TestCase):
    def gen_html(self, *args):
        return gen_html.gen_html(TEST_DATA, *args)[0]

    def testGenHtml(self):
        html = self.gen_html('')
        self.assertIn("test1", html)
        self.assertIn("test2", html)
        self.assertIn("release", html)
        self.assertIn("debug", html)

    def testGenHtmlFilter(self):
        html = self.gen_html('release')
        self.assertIn("release", html)
        self.assertIn('skipped">\ntest2', html)
        self.assertNotIn("debug", html)

    def testGenHtmlFilterExact(self):
        html = self.gen_html('release', True)
        self.assertNotIn('debug', html)

    def testMain(self):
        temp_dir = tempfile.mkdtemp(prefix='kube-test-hist-')
        try:
            tests_json = os.path.join(temp_dir, 'tests.json')
            with open(tests_json, 'w') as f:
                json.dump(TEST_DATA, f)
            gen_html.main(['--suites', '--prefixes', ',rel,deb',
                           '--output-dir', temp_dir, '--input', tests_json])
            for page in ('index', 'suite-kubernetes-debug', 'tests', 'tests-rel', 'tests-deb'):
                self.assertTrue(os.path.exists('%s/%s.html' % (temp_dir, page)))
        finally:
            shutil.rmtree(temp_dir)

if __name__ == '__main__':
    unittest.main()
