#!/usr/bin/env python

# Copyright 2015 The Kubernetes Authors All rights reserved.
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

from __future__ import print_function

import json
import mmap
import os
import re
import sys

def PrintError(*err):
  print(*err, file=sys.stderr)

def file_passes(filename, extension, ref, regexs):
    try:
        f = open(filename, 'r')
    except:
        return False

    data = f.read()

    # remove build tags from the top of Go file
    if extension == "go":
        p = regexs["go_build_constraints"]
        (data, found) = p.subn("", data, 1)

    data = data.splitlines()

    # if our test file is smaller than the reference it surely fails!
    if len(ref) > len(data):
        return False

    # trim our file to the same number of lines as the reference file
    data = data[:len(ref)]

    # Replace all occurances of the regex "2015" with "2014"
    p = regexs["date"]
    for i, d in enumerate(data):

        (data[i], found) = p.subn( '2014', d)
        if found != 0:
            break

    # if we don't match the reference at this point, fail
    if ref != data:
        return False

    return True

def main():
    if len(sys.argv) < 3:
        PrintError("usage: %s extension FILENAME [FILENAMES]" % sys.argv[0])
        return False

    basedir = os.path.dirname(os.path.abspath(__file__))

    extension = sys.argv[1]
    # argv[0] is the binary, argv[1] is the extension (go, sh, py, whatever)
    filenames = sys.argv[2:]

    ref_filename = basedir + "/boilerplate." + extension + ".txt"
    try:
        ref_file = open(ref_filename, 'r')
    except:
        # No boilerplate template is success
        return True
    ref = ref_file.read().splitlines()

    regexs = {}
    # dates can be 2014 or 2015, company holder names can be anything
    regexs["date"] = re.compile( '(2014|2015)' )
    # strip // +build \n\n build constraints
    regexs["go_build_constraints"] = re.compile(r"^(// \+build.*\n)+\n", re.MULTILINE)

    for filename in filenames:
        if not file_passes(filename, extension, ref, regexs):
            print(filename, file=sys.stdout)

if __name__ == "__main__":
  sys.exit(main())
