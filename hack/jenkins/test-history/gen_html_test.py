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
  'test1':
      {'kubernetes-release': [{'build': 3, 'failed': False, 'time': 3.52},
                              {'build': 4, 'failed': True, 'time': 63.21}],
       'kubernetes-debug': [{'build': 5, 'failed': False, 'time': 7.56},
                            {'build': 6, 'failed': False, 'time': 8.43}],
      },
  'test2':
      {'kubernetes-debug': [{'build': 6, 'failed': True, 'time': 3.53}]},
}

class GenHtmlTest(unittest.TestCase):
    """Unit tests for gen_html.py."""
    # pylint: disable=invalid-name

    def testHtmlHeader_NoScript(self):
        result = '\n'.join(gen_html.html_header('', False))
        self.assertNotIn('<script', result)

    def testHtmlHeader_NoTitle(self):
        def Test(title):
            result = '\n'.join(gen_html.html_header(title, False))
            self.assertNotIn('<title', result)
        Test('')
        Test(None)

    def testHtmlHeader_Title(self):
        lines = gen_html.html_header('foo', False)
        for item in lines:
          if '<title' in item:
            self.assertIn('foo', item)
            break
        else:
          self.fail('No foo in: %s' % '\n'.join(lines))

    def testHtmlHeader_Script(self):
        lines = gen_html.html_header('', True)
        for item in lines:
          if '<script' in item:
            break
        else:
          self.fail('No script in: %s' % '\n'.join(lines))

    @staticmethod
    def gen_html(*args):
        """Call gen_html with TEST_DATA."""
        return gen_html.gen_html(TEST_DATA, *args)[0]

    def testGenHtml(self):
        """Test that the expected tests and jobs are in the results."""
        html = self.gen_html('')
        self.assertIn('test1', html)
        self.assertIn('test2', html)
        self.assertIn('release', html)
        self.assertIn('debug', html)

    def testGenHtmlFilter(self):
        """Test that filtering to just the release jobs works."""
        html = self.gen_html('release')
        self.assertIn('release', html)
        self.assertIn('skipped">\ntest2', html)
        self.assertNotIn('debug', html)

    def testGenHtmlFilterExact(self):
        """Test that filtering to an exact name works."""
        html = self.gen_html('release', True)
        self.assertIn('release', html)
        self.assertNotIn('debug', html)

    def testGetOptions(self):
        """Test argument parsing works correctly."""

        def check(args, expected_output_dir, expected_input):
            """Check that args is parsed correctly."""
            options = gen_html.get_options(args)
            self.assertEquals(expected_output_dir, options.output_dir)
            self.assertEquals(expected_input, options.input)


        check(['--output-dir=foo', '--input=bar'], 'foo', 'bar')
        check(['--output-dir', 'foo', '--input', 'bar'], 'foo', 'bar')
        check(['--input=bar', '--output-dir=foo'], 'foo', 'bar')

    def testGetOptions_Missing(self):
        """Test missing arguments raise an exception."""
        def check(args):
            """Check that args raise an exception."""
            with self.assertRaises(SystemExit):
                gen_html.get_options(args)

        check([])
        check(['--output-dir=foo'])
        check(['--input=bar'])

    def testMain(self):
        """Test main() creates pages."""
        temp_dir = tempfile.mkdtemp(prefix='kube-test-hist-')
        try:
            tests_json = os.path.join(temp_dir, 'tests.json')
            with open(tests_json, 'w') as buf:
                json.dump(TEST_DATA, buf)
            gen_html.main(tests_json, temp_dir)
            for page in (
                    'index',
                    'tests-kubernetes',
                    'suite-kubernetes-release',
                    'suite-kubernetes-debug'):
                self.assertTrue(os.path.exists('%s/%s.html' % (temp_dir, page)))
        finally:
            shutil.rmtree(temp_dir)

if __name__ == '__main__':
    unittest.main()
