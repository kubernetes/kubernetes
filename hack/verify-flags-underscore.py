#!/usr/bin/env python

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

from __future__ import print_function

import json
import mmap
import os
import re
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filenames", help="list of files to check, all files if unspecified", nargs='*')
parser.add_argument("-e", "--skip-exceptions", help="ignore hack/verify-flags/exceptions.txt and print all output", action="store_true")
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
            while 1:
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
        if '_gopath' in dirs:
            dirs.remove('_gopath')
        if 'third_party' in dirs:
            dirs.remove('third_party')
        if '.git' in dirs:
            dirs.remove('.git')
        if '.make' in dirs:
            dirs.remove('.make')
        if 'BUILD' in files:
           files.remove('BUILD')
        if 'exceptions.txt' in files:
            files.remove('exceptions.txt')
        if 'known-flags.txt' in files:
            files.remove('known-flags.txt')

        for name in files:
            pathname = os.path.join(root, name)
            if is_binary(pathname):
                continue
            all_files.append(pathname)
    return all_files

def normalize_files(rootdir, files):
    newfiles = []
    a = ['Godeps', '_gopath', 'third_party', '.git', 'exceptions.txt', 'known-flags.txt']
    for f in files:
        if any(x in f for x in a):
            continue
        if f.endswith(".svg"):
            continue
        if f.endswith(".gliffy"):
            continue
        if f.endswith(".md"):
            continue
        if f.endswith(".yaml"):
            continue
        newfiles.append(f)
    for i, f in enumerate(newfiles):
        if not os.path.isabs(f):
            newfiles[i] = os.path.join(rootdir, f)
    return newfiles

def line_has_bad_flag(line, flagre):
    results  = flagre.findall(line)
    for result in results:
        if not "_" in result:
            return False
        # this should exclude many cases where jinja2 templates use kube flags
        # as variables, except it uses _ for the variable name
        if "{% set" + result + "= \"" in line:
            return False
        if "pillar[" + result + "]" in line:
            return False
        if "grains" + result in line:
            return False
         # something common in juju variables...
        if "template_data[" + result + "]" in line:
            return False
        return True
    return False

def check_known_flags(rootdir):
    pathname = os.path.join(rootdir, "hack/verify-flags/known-flags.txt")
    f = open(pathname, 'r')
    flags = set(f.read().splitlines())
    f.close()

    illegal_known_flags = set()
    for flag in flags:
        if len(flag) > 0:
            if not "-" in flag:
                illegal_known_flags.add(flag)

    if len(illegal_known_flags) != 0:
        print("All flags in hack/verify-flags/known-flags.txt should contain character -, found these flags without -")
        l = list(illegal_known_flags)
        l.sort()
        print("%s" % "\n".join(l))
        sys.exit(1)


# The list of files might not be the whole repo. If someone only changed a
# couple of files we don't want to run all of the golang files looking for
# flags. Instead load the list of flags from hack/verify-flags/known-flags.txt
# If running the golang files finds a new flag not in that file, return an
# error and tell the user to add the flag to the flag list.
def get_flags(rootdir, files):
    # preload the 'known' flags
    pathname = os.path.join(rootdir, "hack/verify-flags/known-flags.txt")
    f = open(pathname, 'r')
    flags = set(f.read().splitlines())
    f.close()

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

    new_flags = set()
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
            if not "-" in flag:
                continue
            if flag not in flags:
                new_flags.add(flag)
    if len(new_excluded_flags) != 0:
        print("Found a flag declared with an _ but which is not explicitly listed as a valid flag name in hack/verify-flags/excluded-flags.txt")
        print("Are you certain this flag should not have been declared with an - instead?")
        l = list(new_excluded_flags)
        l.sort()
        print("%s" % "\n".join(l))
        sys.exit(1)
    if len(new_flags) != 0:
        print("Found flags with character - in golang files not in the list of known flags. Please add these to hack/verify-flags/known-flags.txt")
        l = list(new_flags)
        l.sort()
        print("%s" % "\n".join(l))
        sys.exit(1)
    return list(flags)

def flags_to_re(flags):
    """turn the list of all flags we found into a regex find both - and _ versions"""
    dashRE = re.compile('[-_]')
    flagREs = []
    for flag in flags:
        # turn all flag names into regexs which will find both types
        newre = dashRE.sub('[-_]', flag)
        # only match if there is not a leading or trailing alphanumeric character
        flagREs.append("[^\w${]" + newre + "[^\w]")
    # turn that list of regex strings into a single large RE
    flagRE = "|".join(flagREs)
    flagRE = re.compile(flagRE)
    return flagRE

def load_exceptions(rootdir):
    exceptions = set()
    if args.skip_exceptions:
        return exceptions
    exception_filename = os.path.join(rootdir, "hack/verify-flags/exceptions.txt")
    exception_file = open(exception_filename, 'r')
    for exception in exception_file.read().splitlines():
        out = exception.split(":", 1)
        if len(out) != 2:
            print("Invalid line in exceptions file: %s" % exception)
            continue
        filename = out[0]
        line = out[1]
        exceptions.add((filename, line))
    return exceptions

def main():
    rootdir = os.path.dirname(__file__) + "/../"
    rootdir = os.path.abspath(rootdir)

    exceptions = load_exceptions(rootdir)

    if len(args.filenames) > 0:
        files = args.filenames
    else:
        files = get_all_files(rootdir)
    files = normalize_files(rootdir, files)

    check_known_flags(rootdir)

    flags = get_flags(rootdir, files)
    flagRE = flags_to_re(flags)

    bad_lines = []
    # walk all the file looking for any flag that was declared and now has an _
    for pathname in files:
        relname = os.path.relpath(pathname, rootdir)
        f = open(pathname, 'r')
        for line in f.read().splitlines():
            if line_has_bad_flag(line, flagRE):
                if (relname, line) not in exceptions:
                    bad_lines.append((relname, line))
        f.close()

    if len(bad_lines) != 0:
        if not args.skip_exceptions:
            print("Found illegal 'flag' usage. If these are false negatives you should run `hack/verify-flags-underscore.py -e > hack/verify-flags/exceptions.txt` to update the list.")
        bad_lines.sort()
        for (relname, line) in bad_lines:
            print("%s:%s" % (relname, line))
        return 1

if __name__ == "__main__":
  sys.exit(main())
