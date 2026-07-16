#!/usr/bin/env python3

# Copyright 2015 The Kubernetes Authors.
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
import os
import re
import sys

parser = argparse.ArgumentParser()
parser.add_argument("filenames", help="list of files to check, all files if unspecified", nargs='*')
args = parser.parse_args()

# Cargo culted from http://stackoverflow.com/questions/898669/how-can-i-detect-if-a-file-is-binary-non-text-in-python
def is_binary(pathname):
    """Return true if the given filename is binary.
    @raise EnvironmentError: if the file does not exist or cannot be accessed.
    @attention: found @ http://bytes.com/topic/python/answers/21222-determine-file-type-binary-text on 6/08/2010
    @author: Trent Mick <TrentM@ActiveState.com>
    @author: Jorge Orpinel <jorge@orpinel.com>"""
    try:
        with open(pathname, 'r') as f:
            CHUNKSIZE = 1024
            while True:
                chunk = f.read(CHUNKSIZE)
                if '\0' in chunk: # found null byte
                    return True
                if len(chunk) < CHUNKSIZE:
                    break # done
    except:
        return True

    return False

def get_all_files(rootdir):
    all_files = []
    for root, dirs, files in os.walk(rootdir):
        # don't visit certain dirs
        if 'vendor' in dirs:
            dirs.remove('vendor')
        if 'staging' in dirs:
            dirs.remove('staging')
        if '_output' in dirs:
            dirs.remove('_output')
        if 'third_party' in dirs:
            dirs.remove('third_party')
        if '.git' in dirs:
            dirs.remove('.git')

        for name in files:
            pathname = os.path.join(root, name)
            if not is_binary(pathname):
                all_files.append(pathname)
    return all_files

# Collects all the flags used in golang files and verifies the flags do
# not contain underscore. If any flag needs to be excluded from this check,
# need to add that flag in hack/verify-flags/excluded-flags.txt.
def check_underscore_in_flags(rootdir, files):
    # preload the 'known' flags which don't follow the - standard
    pathname = os.path.join(rootdir, "hack/verify-flags/excluded-flags.txt")
    f = open(pathname, 'r')
    excluded_flags = set(f.read().splitlines())
    f.close()

    regexs = [ re.compile('Var[P]?\([^,]*, "([^"]*)"'),
               re.compile('.String[P]?\("([^"]*)",[^,]+,[^)]+\)'),
               re.compile('.Int[P]?\("([^"]*)",[^,]+,[^)]+\)'),
               re.compile('.Bool[P]?\("([^"]*)",[^,]+,[^)]+\)'),
               re.compile('.Duration[P]?\("([^"]*)",[^,]+,[^)]+\)'),
               re.compile('.StringSlice[P]?\("([^"]*)",[^,]+,[^)]+\)') ]

    new_excluded_flags = set()
    # walk all the files looking for any flags being declared
    for pathname in files:
        if not pathname.endswith(".go"):
            continue
        f = open(pathname, 'r')
        data = f.read()
        f.close()
        matches = []
        for regex in regexs:
            matches = matches + regex.findall(data)
        for flag in matches:
            if any(x in flag for x in excluded_flags):
                continue
            if "_" in flag:
                new_excluded_flags.add(flag)
    if len(new_excluded_flags) != 0:
        print("Found a flag declared with an _ but which is not explicitly listed as a valid flag name in hack/verify-flags/excluded-flags.txt")
        print("Are you certain this flag should not have been declared with an - instead?")
        l = list(new_excluded_flags)
        l.sort()
        print(("%s" % "\n".join(l)))
        sys.exit(1)

def main():
    rootdir = os.path.dirname(__file__) + "/../"
    rootdir = os.path.abspath(rootdir)

    if len(args.filenames) > 0:
        files = args.filenames
    else:
        files = get_all_files(rootdir)

    check_underscore_in_flags(rootdir, files)

if __name__ == "__main__":
  sys.exit(main())
