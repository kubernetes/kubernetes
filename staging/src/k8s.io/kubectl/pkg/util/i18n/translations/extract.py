#!/usr/bin/env python3

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
    sys.stdout.write('{}\n"k8s.io/kubectl/pkg/util/i18n"\n'.format(match.group(1)))

IMPORT_MATCH = MatchHandler('(.*"k8s.io/kubectl/pkg/cmd/util")', import_replace)


def string_flag_replace(match, file, line_number):
    """Replace a cmd.Flags().String("...", "", "...") with an internationalization
    """
    sys.stdout.write('{}i18n.T("{})"))\n'.format(match.group(1), match.group(2)))

STRING_FLAG_MATCH = MatchHandler('(\s+cmd\.Flags\(\).String\("[^"]*", "[^"]*", )"([^"]*)"\)', string_flag_replace)


def long_string_replace(match, file, line_number):
    return '{}i18n.T({}){}'.format(match.group(1), match.group(2), match.group(3))

LONG_DESC_MATCH = MatchHandler('(LongDesc\()(`[^`]+`)([^\n]\n)', long_string_replace)

EXAMPLE_MATCH = MatchHandler('(Examples\()(`[^`]+`)([^\n]\n)', long_string_replace)

def replace(filename, matchers, multiline_matchers):
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
    with open(filename, 'r') as datafile:
        content = datafile.read()
        for matcher in multiline_matchers:
            match = matcher.regex.search(content)
            while match:
                rep = matcher.replace_fn(match, filename, 0)
                # Escape back references in the replacement string
                # (And escape for Python)
                # (And escape for regex)
                rep = re.sub('\\\\(\\d)', '\\\\\\\\\\1', rep)
                content = matcher.regex.sub(rep, content, 1)
                match = matcher.regex.search(content)
        sys.stdout.write(content)

    # gofmt the file again
    from subprocess import call
    call(["goimports", "-w", filename])

replace(sys.argv[1], [SHORT_MATCH, IMPORT_MATCH, STRING_FLAG_MATCH], [LONG_DESC_MATCH, EXAMPLE_MATCH])
