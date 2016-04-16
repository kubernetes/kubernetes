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

"""Tests for gen_json."""

import unittest

import gen_json


class GenJsonTest(unittest.TestCase):
    """Unit tests for gen_json.py."""
    # pylint: disable=invalid-name

    def testGetOptions(self):
        """Test argument parsing works correctly."""
        def check(args, expected_server, expected_match):
            """Check that all args are parsed as expected."""
            options = gen_json.get_options(args)
            self.assertEquals(expected_server, options.server)
            self.assertEquals(expected_match, options.match)


        check(['--server=foo', '--match=bar'], 'foo', 'bar')
        check(['--server', 'foo', '--match', 'bar'], 'foo', 'bar')
        check(['--match=bar', '--server=foo'], 'foo', 'bar')

    def testGetOptions_Missing(self):
        """Test missing arguments raise an exception."""
        def check(args):
            """Check that missing args raise an exception."""
            with self.assertRaises(SystemExit):
                gen_json.get_options(args)

        check([])
        check(['--server=foo'])
        check(['--match=bar'])



if __name__ == '__main__':
    unittest.main()
