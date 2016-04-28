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

"""Generates a JSON file containing test history for the last day.

Writes the JSON out to tests.json.
"""

from __future__ import print_function

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib2
from xml.etree import ElementTree
import zlib


def get_json(url):
    """Does an HTTP GET to url and parses the JSON response. None on failure."""
    try:
        content = urllib2.urlopen(url).read().decode('utf-8')
        return json.loads(content)
    except urllib2.HTTPError:
        return None


def get_jobs(server):
    """Generates all job names running on the server."""
    jenkins_json = get_json('{}/api/json'.format(server))
    if not jenkins_json:
        return
    for job in jenkins_json['jobs']:
        yield job['name']

def get_builds(server, job):
    """Generates all build numbers for a given job."""
    job_json = get_json('{}/job/{}/api/json'.format(server, job))
    if not job_json:
        return
    for build in job_json['builds']:
        yield build['number']


def get_build_info(server, job, build):
    """Returns building status along with timestamp for a given build."""
    path = '{}/job/{}/{}/api/json'.format(server, job, str(build))
    build_json = get_json(path)
    if not build_json:
        return True, 0
    return build_json['building'], build_json['timestamp']


def gcs_ls(path):
    """Lists objects under a path on gcs."""
    try:
        result = subprocess.check_output(
            ['gsutil', 'ls', path],
            stderr=open(os.devnull, 'w'))
    except subprocess.CalledProcessError:
        result = b''
    for subpath in result.decode('utf-8').split():
        yield subpath

def gcs_ls_build(job, build):
    """Lists all files under a given job and build path."""
    url = 'gs://kubernetes-jenkins/logs/{}/{}'.format(job, str(build))
    for path in gcs_ls(url):
        yield path


def gcs_ls_artifacts(job, build):
    """Lists all artifacts for a build."""
    for path in gcs_ls_build(job, build):
        if path.endswith('artifacts/'):
            for artifact in gcs_ls(path):
                yield artifact


def gcs_ls_junit_paths(job, build):
    """Lists the paths of JUnit XML files for a build."""
    for path in gcs_ls_artifacts(job, build):
        if re.match(r'.*/junit.*\.xml$', path):
            yield path


def gcs_get_tests(path):
    """Generates test data out of the provided JUnit path.

    Returns None if there's an issue parsing the XML.
    Yields name, time, failed, skipped for each test.
    """
    try:
        data = subprocess.check_output(
            ['gsutil', 'cat', path], stderr=open(os.devnull, 'w'))
    except subprocess.CalledProcessError:
        return

    try:
        data = zlib.decompress(data, zlib.MAX_WBITS | 16)
    except zlib.error:
        # Don't fail if it's not gzipped.
        pass

    try:
        root = ElementTree.fromstring(data)
    except ElementTree.ParseError:
        return

    for child in root:
        name = child.attrib['name']
        ctime = float(child.attrib['time'])
        failed = False
        skipped = False
        for param in child:
            if param.tag == 'skipped':
                skipped = True
            elif param.tag == 'failure':
                failed = True
        yield name, ctime, failed, skipped


def get_tests_from_junit_path(path):
    """Generates all tests in a JUnit GCS path."""
    for test in gcs_get_tests(path):
        if not test:
            continue
        yield test


def get_tests_from_build(job, build):
    """Generates all tests for a build."""
    for junit_path in gcs_ls_junit_paths(job, build):
        for test in get_tests_from_junit_path(junit_path):
            yield test


def get_daily_builds(server, matcher):
    """Generates all (job, build) pairs for the last day."""
    now = time.time()
    for job in get_jobs(server):
        if not matcher(job):
            continue
        for build in reversed(sorted(get_builds(server, job))):
            building, timestamp = get_build_info(server, job, build)
            # Skip if it's still building.
            if building:
                continue
            # Quit once we've walked back over a day.
            if now - timestamp / 1000 > 60*60*24:
                break
            yield job, build


def get_tests(server, matcher):
    """Returns a dictionary of tests to be JSON encoded."""
    tests = {}
    for job, build in get_daily_builds(server, matcher):
        print('{}/{}'.format(job, str(build)))
        for name, duration, failed, skipped in get_tests_from_build(job, build):
            if name not in tests:
                tests[name] = {}
            if skipped:
                continue
            if job not in tests[name]:
                tests[name][job] = []
            tests[name][job].append({
                'build': build,
                'failed': failed,
                'time': duration
            })
    return tests


def main(server, match):
    """Collect test info in matching jobs."""
    print('Finding tests in jobs matching {} at server {}'.format(
        match, server))
    matcher = re.compile(match)
    tests = get_tests(server, matcher)
    with open('tests.json', 'w') as buf:
        json.dump(tests, buf, sort_keys=True)


def get_options(argv):
    """Process command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--server',
        help='hostname of jenkins server',
        required=True,
    )
    parser.add_argument(
        '--match',
        help='filter to job names matching this re',
        required=True,
    )
    return parser.parse_args(argv)


if __name__ == '__main__':
    OPTIONS = get_options(sys.argv[1:])
    main(OPTIONS.server, OPTIONS.match)
