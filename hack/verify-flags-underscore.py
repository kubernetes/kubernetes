#!/usr/bin/env python3

# Copyright 2022 The Kubernetes Authors.
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


# Cargo culted from http://stackoverflow.com/questions/898669/how-can-i-detect-if-a-file-is-binary-non-text-in-python
def is_binary(pathname):
    """Return true if the given filename is binary.

    @raise EnvironmentError: if the file does not exist or cannot be accessed.
    @attention: found @ http://bytes.com/topic/python/answers/21222-determine-file-type-binary-text on 6/08/2010
    @author: Trent Mick <TrentM@ActiveState.com>
    @author: Jorge Orpinel <jorge@orpinel.com>
    """
    try:
        with open(pathname, 'r') as file:
            CHUNK_SIZE = 1024
            while True:
                chunk = file.read(CHUNK_SIZE)
                if '\0' in chunk: # found null byte
                    return True
                if len(chunk) < CHUNK_SIZE:
                    break # done
    except Exception:
        return True

    return False

def get_all_files(rootdir: str):
    all_files = []
    for root, dirs, files in os.walk(rootdir):
        # don't visit certain dirs
        ignored_dirs = [
            'vendor', 'staging', '_output',
            '_gopath', 'third_party', '.git',
            '.make',
        ]
        for ignored_dir in ignored_dirs:
            if ignored_dir in dirs:
                dirs.remove(ignored_dir)

        # don't visit certain files
        ignored_files = ['BUILD']
        for ignored_file in ignored_files:
            if ignored_file in files:
                files.remove(ignored_file)

        for name in files:
            pathname = os.path.join(root, name)
            if not is_binary(pathname):
                all_files.append(pathname)

    return all_files

def check_underscore_in_flags(rootdir: str, files: list[str]):
    """Collects all the flags used in golang files and verifies the flags do not contain underscore.

    If any flag needs to be excluded from this check, need to add that flag in hack/verify-flags/excluded-flags.txt.
    """

    # preload the 'known' flags which don't follow the - standard
    excluded_flags_path = os.path.join(rootdir, "hack/verify-flags/excluded-flags.txt")
    excluded_flags_file = open(excluded_flags_path, 'r')
    excluded_flags = set(excluded_flags_file.read().splitlines())
    excluded_flags_file.close()

    regexes = [
        re.compile('Var[P]?\([^,]*, "([^"]*)"'),
        re.compile('.String[P]?\("([^"]*)",[^,]+,[^)]+\)'),
        re.compile('.Int[P]?\("([^"]*)",[^,]+,[^)]+\)'),
        re.compile('.Bool[P]?\("([^"]*)",[^,]+,[^)]+\)'),
        re.compile('.Duration[P]?\("([^"]*)",[^,]+,[^)]+\)'),
        re.compile('.StringSlice[P]?\("([^"]*)",[^,]+,[^)]+\)'),
    ]

    unexpected_excluded_flags = set()
    # walk all the files looking for any flags being declared
    for pathname in files:
        if not pathname.endswith(".go"):
            continue
        file = open(pathname, 'r')
        content = file.read()
        file.close()

        matches = []
        for regex in regexes:
            matches.extend(regex.findall(content))
        for flag in matches:
            if any(x in flag for x in excluded_flags):
                continue
            if "_" in flag:
                unexpected_excluded_flags.add(flag)

    if len(unexpected_excluded_flags) != 0:
        print("Found a flag declared with an _ but which is not explicitly listed as a valid flag name in 'hack/verify-flags/excluded-flags.txt'")
        print("Are you certain this flag should not have been declared with an - instead?")
        excluded_flag_list = sorted(list(unexpected_excluded_flags))
        print("\n".join(excluded_flag_list))
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filenames",
        help="list of files to check, all files if unspecified",
        nargs='*',
    )
    args = parser.parse_args()

    rootdir = os.path.abspath(os.path.dirname(__file__) + "/../")
    if len(args.filenames) > 0:
        files = args.filenames
    else:
        files = get_all_files(rootdir)

    check_underscore_in_flags(rootdir, files)

if __name__ == "__main__":
    main()
