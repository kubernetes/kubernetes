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
import cgi
import collections
import json
import os
import string
import sys
import time


TestMetadata = collections.namedtuple('TestMetadata', [
    'okay',
    'unstable',
    'failed',
    'skipped',
])


def gen_tests(data, prefix, exact_match):
    """Creates the HTML for all test cases.

    Args:
        data: Parsed JSON data that was created by gen_json.py.
        prefix: Considers Jenkins jobs that start with this.
        exact_match: Only match Jenkins jobs with name equal to prefix.

    Returns:
        (html, TestMetadata) for matching tests
    """
    html = ['<ul class="test">']
    totals = collections.defaultdict(int)
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
            test_html.append('<span class="%s">%d/%d</span>' % (
                status, num_builds - num_failed, num_builds))
            test_html.append(
                '<span class="time">%.0f%s</span>' % (avg_time, unit))
            test_html.append(suite)
            test_html.append('</li>')
        test_html.append('</ul>')
        if has_failed:
            status = 'failed'
        elif has_unstable:
            status = 'unstable'
        elif has_test:
            status = 'okay'
        else:
            status = 'skipped'
        totals[status] += 1
        html.append('<li class="test %s">' % status)
        if exact_match and len(test_html) > 2:
            if not (test_html[2].startswith('<span') and
                    test_html[3].startswith('<span')):
                raise ValueError(
                    'couldn\'t extract suite results for prepending')
            html.extend(test_html[2:4])
            html.append(test)
        else:
            html.append(test)
            html.extend(test_html)
        html.append('</li>')
    html.append('</ul>')
    return '\n'.join(html), TestMetadata(
        totals['okay'], totals['unstable'], totals['failed'], totals['skipped'])


def html_header(title, script):
    """Return html header items."""
    html = ['<html>', '<head>']
    html.append('<link rel="stylesheet" type="text/css" href="style.css" />')
    if title:
        html.append('<title>%s</title>' % cgi.escape(title))
    if script:
        html.append('<script src="script.js"></script>')
    html.append('</head>')
    html.append('<body>')
    return html


def gen_html(data, prefix, exact_match=False):
    """Creates the HTML for the entire page.

    Args:
        Same as gen_tests.
    Returns:
        Same as gen_tests.
    """
    tests_html, meta = gen_tests(data, prefix, exact_match)
    if exact_match:
        msg = 'Suite %s' % cgi.escape(prefix)
    elif prefix:
        msg = 'Suites starting with %s' % cgi.escape(prefix)
    else:
        msg = 'All suites'
    html = html_header(title=msg, script=True)
    html.append('<div id="header">%s:' % msg)
    fmt = '<span class="total %s" onclick="toggle(\'%s\');">%s</span>'
    html.append(fmt % ('okay', 'okay', meta.okay))
    html.append(fmt % ('unstable', 'unstable', meta.unstable))
    html.append(fmt % ('failed', 'failed', meta.failed))
    html.append(fmt % ('skipped', 'skipped', meta.skipped))
    html.append('</div>')
    html.append(tests_html)
    html.append('</body>')
    html.append('</html>')
    return '\n'.join(html), meta


def gen_metadata_links(suites):
    """Write clickable pass, ustabled, failed stats."""
    html = []
    for (name, target), meta in sorted(suites.iteritems()):
        html.append('<a class="suite-link" href="%s">' % target)
        html.append('<span class="total okay">%d</span>' % meta.okay)
        html.append('<span class="total unstable">%d</span>' % meta.unstable)
        html.append('<span class="total failed">%d</span>' % meta.failed)
        html.append(name)
        html.append('</a>')
    return html


def write_html(outdir, path, html):
    """Write html to outdir/path."""
    with open(os.path.join(outdir, path), 'w') as buf:
        buf.write(html)


def write_metadata(infile, outdir):
    """Writes tests-*.html and suite-*.html files.

    Args:
      infile: the json file created by gen_json.py
      outdir: a path to write the html files.
    """
    with open(infile) as buf:
        data = json.load(buf)

    prefix_metadata = {}
    prefixes = [
        'kubernetes',
        'kubernetes-e2e',
        'kubernetes-soak',
        'kubernetes-e2e-gce',
        'kubernetes-e2e-gke',
        'kubernetes-upgrade',
    ]
    for prefix in prefixes:
        path = 'tests-%s.html' % prefix
        html, metadata = gen_html(data, prefix, False)
        write_html(outdir, path, html)
        prefix_metadata[prefix or 'kubernetes', path] = metadata

    suite_metadata = {}
    suites = set()
    for suite_names in data.values():
        suites.update(suite_names.keys())
    for suite in sorted(suites):
        path = 'suite-%s.html' % suite
        html, metadata = gen_html(data, suite, True)
        write_html(outdir, path, html)
        suite_metadata[suite, path] = metadata

    blocking = {
        'kubelet-gce-e2e-ci',
        'kubernetes-build',
        'kubernetes-e2e-gce',
        'kubernetes-e2e-gce-scalability',
        'kubernetes-e2e-gce-slow',
        'kubernetes-e2e-gke',
        'kubernetes-e2e-gke-slow',
        'kubernetes-kubemark-5-gce',
        'kubernetes-kubemark-500-gce',
        'kubernetes-test-go',
    }
    blocking_suite_metadata = {
        k: v for (k, v) in suite_metadata.items() if k[0] in blocking}

    return prefix_metadata, suite_metadata, blocking_suite_metadata


def write_index(outdir, prefixes, suites, blockers):
    """Write the index.html with links to each view, including stat summaries.

    Args:
      outdir: the path to write the index.html file
      prefixes: the {(prefix, path): TestMetadata} map
      suites: the {(suite, path): TestMetadata} map
      blockers: the {(suite, path): TestMetadata} map of blocking suites
    """
    html = html_header(title='Kubernetes Test Summary', script=False)
    html.append('<h1>Kubernetes Tests</h1>')
    html.append('Last updated %s' % time.strftime('%F'))

    html.append('<h2>Tests from suites starting with:</h2>')
    html.extend(gen_metadata_links(prefixes))

    html.append('<h2>Blocking suites:</h2>')
    html.extend(gen_metadata_links(blockers))

    html.append('<h2>All suites:</h2>')
    html.extend(gen_metadata_links(suites))

    html.extend(['</body>', '</html>'])
    write_html(outdir, 'index.html', '\n'.join(html))


def main(infile, outdir):
    """Use infile to write test, suite and index html files to outdir."""
    prefixes, suites, blockers = write_metadata(infile, outdir)
    write_index(outdir, prefixes, suites, blockers)


def get_options(argv):
    """Process command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True,
                        help='where to write output pages')
    parser.add_argument('--input', required=True,
                        help='JSON test data to read for input')
    return parser.parse_args(argv)


if __name__ == '__main__':
    OPTIONS = get_options(sys.argv[1:])
    main(OPTIONS.input, OPTIONS.output_dir)
