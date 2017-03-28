#!/usr/bin/env python

# Copyright 2017 The Kubernetes Authors.
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

"""Extract strings from command files and externalize into translation files.
Expects to be run from the root directory of the repository.

Usage:
   extract.py pkg/kubectl/cmd/apply.go

"""
import fileinput
import sys
import re

class MatchHandler(object):
    """ Simple holder for a regular expression and a function
    to run if that regular expression matches a line.
    The function should expect (re.match, file, linenumber) as parameters
    """
    def __init__(self, regex, replace_fn):
        self.regex = re.compile(regex)
        self.replace_fn = replace_fn

def short_replace(match, file, line_number):
    """Replace a Short: ... cobra command description with an internationalization
    """
    sys.stdout.write('{}i18n.T({}),\n'.format(match.group(1), match.group(2)))

SHORT_MATCH = MatchHandler(r'(\s+Short:\s+)("[^"]+"),', short_replace)

def import_replace(match, file, line_number):
    """Add an extra import for the i18n library.
    Doesn't try to be smart and detect if it's already present, assumes a
    gofmt round wil fix things.
    """
    sys.stdout.write('{}\n"k8s.io/kubernetes/pkg/util/i18n"\n'.format(match.group(1)))

IMPORT_MATCH = MatchHandler('(.*"k8s.io/kubernetes/pkg/kubectl/cmd/util")', import_replace)


def string_flag_replace(match, file, line_number):
    """Replace a cmd.Flags().String("...", "", "...") with an internationalization
    """
    sys.stdout.write('{}i18n.T("{})"))\n'.format(match.group(1), match.group(2)))

STRING_FLAG_MATCH = MatchHandler('(\s+cmd\.Flags\(\).String\("[^"]*", "[^"]*", )"([^"]*)"\)', string_flag_replace)

def replace(filename, matchers):
    """Given a file and a set of matchers, run those matchers
    across the file and replace it with the results.
    """
    # Run all the matchers
    line_number = 0
    for line in fileinput.input(filename, inplace=True):
        line_number += 1
        matched = False
        for matcher in matchers:
            match = matcher.regex.match(line)
            if match:
                matcher.replace_fn(match, filename, line_number)
                matched = True
                break
        if not matched:
            sys.stdout.write(line)
    sys.stdout.flush()

    # gofmt the file again
    from subprocess import call
    call(["goimports", "-w", filename])

replace(sys.argv[1], [SHORT_MATCH, IMPORT_MATCH, STRING_FLAG_MATCH])
