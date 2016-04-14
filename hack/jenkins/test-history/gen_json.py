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
import sys
import time
import urllib2


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

def get_tests_from_build(server, job, build):
    """Generates all tests for a build."""
    report = get_json('{}/job/{}/{}/testReport/api/json?tree={}'.format(
        server, job, build, 'suites[cases[name,status,duration]]'))
    if report is None:
        return
    for suite in report['suites']:
        for case in suite['cases']:
            status = case['status']
            failed = status == 'FAILED'
            skipped = status == 'SKIPPED'
            yield case['name'], case['duration'], failed, skipped

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

def builds_for_tests(tests):
    builds_have = set()
    for test in tests.itervalues():
        for builds in test.itervalues():
            for build in builds:
                builds_have.add(build['build'])
    return builds_have

def remove_unwanted(tests, builds_wanted):
    for test in tests.values():
        for job, builds in test.items():  # intentional copy
            builds[:] = [b for b in builds if b['build'] in builds_wanted]
            if not builds:
                test.pop(job)

def get_tests(server, prefix):
    """Returns a dictionary of tests to be JSON encoded."""
    tests = {}
    builds_have = set()
    builds_wanted = set()
    if os.path.exists('tests.json'):
        tests = json.load(open('tests.json'))
        builds_have = builds_for_tests(tests)
    for job, build in get_daily_builds(server, prefix):
        builds_wanted.add(build)
        if build in builds_have:
            continue
        print('{}/{}'.format(job, str(build)))
        for name, duration, failed, skipped in get_tests_from_build(
                server, job, build):
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
    remove_unwanted(tests, builds_wanted)
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
