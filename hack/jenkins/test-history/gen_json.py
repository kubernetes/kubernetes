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

import json
import os
import re
import subprocess
import shutil
import sys
import tempfile
import time
import urllib2
import xml.etree.ElementTree as ET
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
    """Retrieves build numbers, statuses, and timestamps for a given job."""
    job_json = get_json('{}/job/{}/api/json?tree={}'.format(
        server, job, 'builds[number,timestamp,building]'))
    if not job_json:
        return
    for build in job_json['builds']:
        yield build['number'], build['building'], build['timestamp']

def gcs_cp_junit(job, build, outdir):
    try:
        subprocess.check_output(['gsutil', '-m', 'cp',
            'gs://kubernetes-jenkins/logs/{}/{}/artifacts/junit_*.xml'.format(
                job, build), outdir], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        pass  # no artifacts matched

def get_tests_from_junit(path):
    """Generates test data out of the provided JUnit file.

    Returns None if there's an issue parsing the XML.
    Yields name, time, failed, skipped for each test.
    """
    data = open(path).read()

    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        print("bad xml:", path)
        return

    for child in root:
        name = child.attrib['name']
        time = float(child.attrib['time'])
        failed = False
        skipped = False
        for param in child:
            if param.tag == 'skipped':
                skipped = True
            elif param.tag == 'failure':
                failed = True
        yield name, time, failed, skipped

def get_tests_from_build(job, build):
    """Generates all tests for a build."""
    tmpdir = tempfile.mkdtemp(prefix='kube-test-history-')
    try:
        gcs_cp_junit(job, build, tmpdir)
        for junit_path in os.listdir(tmpdir):
            for test in get_tests_from_junit(
                    os.path.join(tmpdir, junit_path)):
                yield test
    finally:
        shutil.rmtree(tmpdir)

def get_daily_builds(server, prefix):
    """Generates all (job, build) pairs for the last day."""
    now = time.time()
    for job in get_jobs(server):
        if not job.startswith(prefix):
            continue
        for build, building, timestamp in sorted(
                get_builds(server, job), reverse=True):
            # Skip if it's still building.
            if building:
                continue
            # Quit once we've walked back over a day.
            if now - timestamp / 1000 > 60*60*24:
                break
            yield job, build

def get_tests(server, prefix):
    """Returns a dictionary of tests to be JSON encoded."""
    tests = {}
    for job, build in get_daily_builds(server, prefix):
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

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: {} <server> <prefix>'.format(sys.argv[0]))
        sys.exit(1)
    server, prefix = sys.argv[1:]
    print('Finding tests prefixed with {} at server {}'.format(prefix, server))
    tests = get_tests(server, prefix)
    with open('tests.json', 'w') as f:
        json.dump(tests, f, sort_keys=True)
