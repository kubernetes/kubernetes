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

import argparse
import collections
import csv
import re
import json
import os
import random
import subprocess
import sys
import time
import urllib2
import zlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OWNERS_PATH = os.path.abspath(
    os.path.join(BASE_DIR, '..', 'test', 'test_owners.csv'))
GCS_URL_BASE = 'https://storage.googleapis.com/kubernetes-test-history/'
SKIP_MAINTAINERS = {
    'a-robinson', 'aronchick', 'bgrant0607-nocc', 'david-mcmahon',
    'goltermann', 'sarahnovotny'}


def normalize(name):
    name = re.sub(r'\[.*?\]|\{.*?\}', '', name)
    name = re.sub(r'\s+', ' ', name)
    return name.strip()


def get_test_history(days_ago):
    url = time.strftime(GCS_URL_BASE + 'logs/%Y-%m-%d.json',
                        time.gmtime(time.time() - days_ago * 24 * 60 * 60))
    resp = urllib2.urlopen(url)
    content = resp.read()
    if resp.headers.get('content-encoding') == 'gzip':
        content = zlib.decompress(content, 15 | 16)
    return json.loads(content)


def get_test_names_from_test_history():
    test_names = set()
    for days_ago in range(4):
        test_history = get_test_history(days_ago)
        test_names.update(normalize(name) for name in test_history['test_names'])
    return test_names


def get_test_names_from_local_files():
    tests_json = subprocess.check_output(['go', 'run', 'test/list/main.go', '-json'])
    tests = json.loads(tests_json)
    return {normalize(t['Name'] + (' ' + t['TestName'] if 'k8s.io/' not in t['Name'] else ''))
            for t in tests}


def load_owners(fname):
    owners = {}
    with open(fname) as f:
        for n, (name, owner, random_assignment) in enumerate(csv.reader(f)):
            if n == 0:
                continue  # header
            owners[normalize(name)] = (owner, int(random_assignment))
        return owners


def write_owners(fname, owners):
    with open(fname, 'w') as f:
        out = csv.writer(f, lineterminator='\n')
        out.writerow(['name', 'owner', 'auto-assigned'])
        sort_key = lambda (k, v): (k != 'DEFAULT', k)  # put 'DEFAULT' first.
        items = sorted(owners.items(), key=sort_key)
        for name, (owner, random_assignment) in items:
            out.writerow([name, owner, int(random_assignment)])


def get_maintainers():
    # Github doesn't seem to support team membership listing without a key with
    # org admin privileges. Instead, we do it manually:
    # Open https://github.com/orgs/kubernetes/teams/kubernetes-maintainers
    # Run this in the js console:
    # [].slice.call(document.querySelectorAll('.team-member-username a')).map(
    #     e => e.textContent.trim())
    ret = {"alex-mohr", "apelisse", "aronchick", "bgrant0607", "bgrant0607-nocc",
           "bprashanth", "brendandburns", "caesarxuchao", "childsb", "cjcullen",
           "david-mcmahon", "davidopp", "dchen1107", "deads2k", "derekwaynecarr",
           "eparis", "erictune", "fabioy", "fejta", "fgrzadkowski", "freehan",
           "gmarek", "grodrigues3", "ingvagabund", "ixdy", "janetkuo", "jbeda",
           "jessfraz", "jingxu97", "jlowdermilk", "jsafrane", "jszczepkowski",
           "justinsb", "kargakis", "Kashomon", "kevin-wangzefeng", "krousey",
           "lavalamp", "liggitt", "luxas", "madhusudancs", "maisem", "matchstick",
           "mbohlool", "mikedanese", "mml", "mtaufen", "mwielgus", "ncdc",
           "nikhiljindal", "piosz", "pmorie", "pwittrock", "Q-Lee", "quinton-hoole",
           "Random-Liu", "rmmh", "roberthbailey", "saad-ali", "smarterclayton",
           "soltysh", "spxtr", "sttts", "thelinuxfoundation", "thockin",
           "timothysc", "timstclair", "vishh", "wojtek-t", "xiang90", "yifan-gu",
           "yujuhong", "zmerlynn"}
    return sorted(ret - SKIP_MAINTAINERS)


def detect_github_username():
    origin_url = subprocess.check_output(['git', 'config', 'remote.origin.url'])
    m = re.search(r'github.com[:/](.*)/', origin_url)
    if m and m.group(1) != 'kubernetes':
        return m.group(1)
    raise ValueError('unable to determine GitHub user from '
                     '`git config remote.origin.url` output, run with --user instead')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', action='store_true', help='Generate test list from result history.')
    parser.add_argument('--user', help='User to assign new tests to (or RANDOM, default: current GitHub user).')
    parser.add_argument('--addonly', action='store_true', help='Only add missing tests, do not change existing.')
    parser.add_argument('--check', action='store_true', help='Exit with a nonzero status if the test list has changed.')
    options = parser.parse_args()

    if options.history:
        test_names = get_test_names_from_test_history()
    else:
        test_names = get_test_names_from_local_files()
    test_names.add('DEFAULT')
    test_names = sorted(test_names)
    owners = load_owners(OWNERS_PATH)

    outdated_tests = sorted(set(owners) - set(test_names))
    new_tests = sorted(set(test_names) - set(owners))
    maintainers = get_maintainers()

    print '# OUTDATED TESTS (%d):' % len(outdated_tests)
    print  '\n'.join('%s -- %s%s' %
                     (t, owners[t][0], ['', ' (random)'][owners[t][1]])
                      for t in outdated_tests)
    print '# NEW TESTS (%d):' % len(new_tests)
    print  '\n'.join(new_tests)

    if options.check:
        if new_tests or outdated_tests:
            print
            print 'ERROR: the test list has changed'
            sys.exit(1)
        sys.exit(0)

    if not options.user:
        options.user = detect_github_username()

    for name in outdated_tests:
        owners.pop(name)

    if not options.addonly:
        print '# UNEXPECTED MAINTAINERS ',
        print '(randomly assigned, but not in kubernetes-maintainers)'
        for name, (owner, random_assignment) in sorted(owners.iteritems()):
            if random_assignment and owner not in maintainers:
                print '%-16s %s' % (owner, name)
                owners.pop(name)
        print

    owner_counts = collections.Counter(
        owner for name, (owner, random) in owners.iteritems()
        if owner in maintainers)
    for test_name in set(test_names) - set(owners):
        random_assignment = True
        if options.user.lower() == 'random':
            new_owner, _count = random.choice(owner_counts.most_common()[-4:])
        else:
            new_owner = options.user
            random_assignment = False
        owner_counts[new_owner] += 1
        owners[test_name] = (new_owner, random_assignment)

    if options.user.lower() == 'random':
        print '# Tests per maintainer:'
        for owner, count in owner_counts.most_common():
            print '%-20s %3d' % (owner, count)

    write_owners(OWNERS_PATH, owners)


if __name__ == '__main__':
    main()
