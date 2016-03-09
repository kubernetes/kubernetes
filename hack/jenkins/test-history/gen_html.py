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

"""Creates an HTML report for all jobs starting with a given prefix.

Reads the JSON from tests.json, and prints the HTML to stdout.

This code is pretty nasty, but gets the job done.

It would be really spiffy if this used an HTML template system, but for now
we're old-fashioned. We could also generate these with JS, directly from the
JSON. That would allow custom filtering and stuff like that.
"""

from __future__ import print_function

import json
import string
import sys

def gen_tests(data, prefix):
    """Creates the HTML for all test cases.

    Args:
        data: Parsed JSON data that was created by gen_json.py.
        prefix: Considers Jenkins jobs that start with this.

    Returns:
        The HTML as a list of elements along with the number of passing,
        unstable, failing, and skipped tests.
    """
    html = ['<ul class="test">']
    total_okay = 0
    total_unstable = 0
    total_failed = 0
    total_skipped = 0
    for test in sorted(data, key=string.lower):
        test_html = ['<ul class="suite">']
        has_test = False
        has_failed = False
        has_unstable = False
        for suite in sorted(data[test]):
            if not suite.startswith(prefix):
                continue
            has_test = True
            num_failed = 0
            num_builds = 0
            total_time = 0
            for build in data[test][suite]:
                num_builds += 1
                if build['failed']:
                    num_failed += 1
                total_time += build['time']
            avg_time = total_time / num_builds
            unit = 's'
            if avg_time > 60:
                avg_time /= 60
                unit = 'm'
            if num_failed == num_builds:
                has_failed = True
                status = 'failed'
            elif num_failed > 0:
                has_unstable = True
                status = 'unstable'
            else:
                status = 'okay'
            test_html.append('<li class="suite">')
            test_html.append('<span class="{}">{}/{}</span>'.format(status, str(num_builds - num_failed), str(num_builds)))
            test_html.append('<span class="time">{}</span>'.format(str(int(avg_time)) + unit))
            test_html.append(suite)
            test_html.append('</li>')
        if has_failed:
            status = 'failed'
            total_failed += 1
        elif has_unstable:
            status = 'unstable'
            total_unstable += 1
        elif has_test:
            status = 'okay'
            total_okay += 1
        else:
            status = 'skipped'
            total_skipped += 1
        html.append('<li class="test {}">{}'.format(status, test))
        html.extend(test_html)
        html.append('</ul>')
        html.append('</li>')
    html.append('</ul>')
    return html, total_okay, total_unstable, total_failed, total_skipped

def gen_html(data, prefix):
    """Creates the HTML for the entire page.

    Args: Same as gen_tests.
    Returns: Just the list of HTML elements.
    """
    tests_html, okay, unstable, failed, skipped = gen_tests(data, prefix)
    html = ['<html>', '<head>']
    html.append('<link rel="stylesheet" type="text/css" href="style.css" />')
    html.append('<script src="script.js"></script>')
    html.append('</head>')
    html.append('<body>')
    if len(prefix) > 0:
        html.append('<div id="header">Suites starting with {}:'.format(prefix))
    else:
        html.append('<div id="header">All suites:')
    html.append('<span class="total okay" onclick="toggle(\'okay\');">{}</span>'.format(str(okay)))
    html.append('<span class="total unstable" onclick="toggle(\'unstable\');">{}</span>'.format(str(unstable)))
    html.append('<span class="total failed" onclick="toggle(\'failed\');">{}</span>'.format(str(failed)))
    html.append('<span class="total skipped" onclick="toggle(\'skipped\');">{}</span>'.format(str(skipped)))
    html.append('</div>')
    html.extend(tests_html)
    html.append('</body>')
    html.append('</html>')
    return html

if __name__ == '__main__':
    prefix = ''
    if len(sys.argv) == 2:
        prefix = sys.argv[1]
    with open('tests.json', 'r') as f:
        print('\n'.join(gen_html(json.load(f), prefix)))
