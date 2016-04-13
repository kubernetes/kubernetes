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

import argparse
import json
import os
import string
import sys
import time

def gen_tests(data, prefix, exact_match):
    """Creates the HTML for all test cases.

    Args:
        data: Parsed JSON data that was created by gen_json.py.
        prefix: Considers Jenkins jobs that start with this.
        exact_match: Only match Jenkins jobs with name equal to prefix.

    Returns:
        The HTML as a list of elements along with a tuple of the number of
        passing, unstable, failing, and skipped tests.
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
            if exact_match and suite != prefix:
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
            test_html.append('<span class="%s">%d/%d</span>' % (status, num_builds - num_failed, num_builds))
            test_html.append('<span class="time">%.0f%s</span>' % (avg_time, unit))
            test_html.append(suite)
            test_html.append('</li>')
        test_html.append('</ul>')
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
        html.append('<li class="test %s">' % status)
        if exact_match and len(test_html) > 2:
            if not (test_html[2].startswith('<span') and test_html[3].startswith('<span')):
                raise ValueError("couldn't extract suite results for prepending")
            html.extend(test_html[2:4])
            html.append(test)
        else:
            html.append(test)
            html.extend(test_html)
        html.append('</li>')
    html.append('</ul>')
    return '\n'.join(html), (total_okay, total_unstable, total_failed, total_skipped)

def html_header():
    html = ['<html>', '<head>']
    html.append('<link rel="stylesheet" type="text/css" href="style.css" />')
    html.append('<script src="script.js"></script>')
    html.append('</head>')
    html.append('<body>')
    return html

def gen_html(data, prefix, exact_match=False):
    """Creates the HTML for the entire page.

    Args: Same as gen_tests.
    Returns: Same as gen_tests.
    """
    tests_html, (okay, unstable, failed, skipped) = gen_tests(data, prefix, exact_match)
    html = html_header()
    if exact_match:
        html.append('<div id="header">Suite %s' % prefix)
    elif len(prefix) > 0:
        html.append('<div id="header">Suites starting with %s:' % prefix)
    else:
        html.append('<div id="header">All suites:')
    html.append('<span class="total okay" onclick="toggle(\'okay\');">%s</span>' % okay)
    html.append('<span class="total unstable" onclick="toggle(\'unstable\');">%d</span>' % unstable)
    html.append('<span class="total failed" onclick="toggle(\'failed\');">%d</span>' % failed)
    html.append('<span class="total skipped" onclick="toggle(\'skipped\');">%d</span>' % skipped)
    html.append('</div>')
    html.append(tests_html)
    html.append('</body>')
    html.append('</html>')
    return '\n'.join(html), (okay, unstable, failed, skipped)

def gen_metadata_links(suites):
    html = []
    for (name, target), (okay, unstable, failed, skipped) in sorted(suites.iteritems()):
        html.append('<a class="suite-link" href="%s">' % target)
        html.append('<span class="total okay">%d</span>' % okay)
        html.append('<span class="total unstable">%d</span>' % unstable)
        html.append('<span class="total failed">%d</span>' % failed)
        html.append(name)
        html.append('</a>')
    return html

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--suites', action='store_true',
                        help='output test results for each suite')
    parser.add_argument('--prefixes',
                        help='comma-separated list of suite prefixes to create pages for')
    parser.add_argument('--output-dir', required=True,
                        help='where to write output pages')
    parser.add_argument('--input', required=True,
                        help='JSON test data to read for input')
    options=parser.parse_args(args)

    with open(options.input) as f:
        data = json.load(f)

    if options.prefixes:
        # the empty prefix means "all tests"
        options.prefixes = options.prefixes.split(',')
        prefix_metadata = {}
        for prefix in options.prefixes:
            if prefix:
                path = 'tests-%s.html' % prefix
                prefix = 'kubernetes-%s' % prefix
            else:
                path = 'tests.html'
            html, prefix_metadata[prefix or 'kubernetes', path] = gen_html(data, prefix, False)
            with open(os.path.join(options.output_dir, path), 'w') as f:
                f.write(html)
    if options.suites:
        suites_set = set()
        for test, suites in data.iteritems():
            suites_set.update(suites.keys())
        suite_metadata = {}
        for suite in sorted(suites_set):
            path = 'suite-%s.html' % suite
            html, suite_metadata[suite, path] = gen_html(data, suite, True)
            with open(os.path.join(options.output_dir, path), 'w') as f:
                f.write(html)
    html = html_header()
    html.append('<h1>Kubernetes Tests</h1>')
    html.append('Last updated %s' % time.strftime('%F'))
    if options.prefixes:
        html.append('<h2>All suites starting with:</h2>')
        html.extend(gen_metadata_links(prefix_metadata))
    if options.suites:
        html.append('<h2>Specific suites:</h2>')
        html.extend(gen_metadata_links(suite_metadata))
    html.extend(['</body>', '</html>'])
    with open(os.path.join(options.output_dir, 'index.html'), 'w') as f:
        f.write('\n'.join(html))

if __name__ == '__main__':
    main(sys.argv[1:])
