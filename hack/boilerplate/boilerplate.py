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
import datetime
import difflib
import glob
import os
import re
import sys

parser = argparse.ArgumentParser()
parser.add_argument(
    "filenames", help="list of files to check, all files if unspecified", nargs="*"
)

rootdir = os.path.dirname(__file__) + "/../../"
rootdir = os.path.abspath(rootdir)
parser.add_argument("--rootdir", default=rootdir, help="root directory to examine")

default_boilerplate_dir = os.path.join(rootdir, "hack/boilerplate")
parser.add_argument("--boilerplate-dir", default=default_boilerplate_dir)

parser.add_argument(
    "-v",
    "--verbose",
    help="give verbose output regarding why a file does not pass",
    action="store_true",
)

args = parser.parse_args()

verbose_out = sys.stderr if args.verbose else open("/dev/null", "w")


def get_refs():
    refs = {}

    for path in glob.glob(os.path.join(args.boilerplate_dir, "boilerplate.*.txt")):
        extension = os.path.basename(path).split(".")[1]

        with open(path, "r") as ref_file:
            refs[extension] = ref_file.read().splitlines()

    return refs


def is_generated_file(data, regexs):
    return regexs["generated"].search(data)


def file_passes(filename, refs, regexs):
    try:
        with open(filename) as stream:
            data = stream.read()
    except OSError as exc:
        print(f"Unable to open {filename}: {exc}", file=verbose_out)
        return False

    # determine if the file is automatically generated
    generated = is_generated_file(data, regexs)

    basename = os.path.basename(filename)
    extension = file_extension(filename)
    if generated:
        if extension == "go":
            extension = "generatego"

    if extension != "":
        ref = refs[extension]
    else:
        ref = refs[basename]

    # remove extra content from the top of files
    if extension in ("go", "generatego"):
        data, found = regexs["go_build_constraints"].subn("", data, 1)
    elif extension in ["sh", "py"]:
        data, found = regexs["shebang"].subn("", data, 1)

    data = data.splitlines()

    # if our test file is smaller than the reference it surely fails!
    if len(ref) > len(data):
        print(
            f"File {filename} smaller than reference ({len(data)} < {len(ref)})",
            file=verbose_out,
        )
        return False

    # trim our file to the same number of lines as the reference file
    data = data[: len(ref)]

    pattern = regexs["year"]
    for line in data:
        if pattern.search(line):
            if generated:
                print(
                    f"File {filename} has the YEAR field, but it should not be in generated file",
                    file=verbose_out,
                )
            else:
                print(
                    "File {filename} has the YEAR field, but missing the year of date",
                    file=verbose_out,
                )
            return False

    if not generated:
        # Replace all occurrences of the regex "2014|2015|2016|2017|2018" with "YEAR"
        pattern = regexs["date"]
        for i, line in enumerate(data):
            data[i], found = pattern.subn("YEAR", line)
            if found != 0:
                break

    # if we don't match the reference at this point, fail
    if ref != data:
        print(f"Header in {filename} does not match reference, diff:", file=verbose_out)
        if args.verbose:
            print(file=verbose_out)
            for line in difflib.unified_diff(
                ref, data, "reference", filename, lineterm=""
            ):
                print(line, file=verbose_out)
            print(file=verbose_out)
        return False

    return True


def file_extension(filename):
    return os.path.splitext(filename)[1].split(".")[-1].lower()


skipped_names = [
    "third_party",
    "_output",
    ".git",
    "cluster/env.sh",
    "vendor",
    "testdata",
    "test/e2e/generated/bindata.go",
    "hack/boilerplate/test",
    "staging/src/k8s.io/kubectl/pkg/generated/bindata.go",
]


def normalize_files(files):
    newfiles = []
    for pathname in files:
        if any(x in pathname for x in skipped_names):
            continue
        newfiles.append(pathname)
    for i, pathname in enumerate(newfiles):
        if not os.path.isabs(pathname):
            newfiles[i] = os.path.join(args.rootdir, pathname)
    return newfiles


def get_files(extensions):
    files = []
    if len(args.filenames) > 0:
        files = args.filenames
    else:
        for root, dirs, walkfiles in os.walk(args.rootdir):
            # don't visit certain dirs. This is just a performance improvement
            # as we would prune these later in normalize_files(). But doing it
            # cuts down the amount of filesystem walking we do and cuts down
            # the size of the file list
            for dname in skipped_names:
                if dname in dirs:
                    dirs.remove(dname)
            for dname in dirs:
                # dirs that start with __ are ignored
                if dname.startswith("__"):
                    dirs.remove(dname)

            for name in walkfiles:
                pathname = os.path.join(root, name)
                files.append(pathname)

    files = normalize_files(files)
    outfiles = []
    for pathname in files:
        basename = os.path.basename(pathname)
        extension = file_extension(pathname)
        if extension in extensions or basename in extensions:
            outfiles.append(pathname)
    return outfiles


def get_dates():
    years = datetime.datetime.now().year
    return "(%s)" % "|".join(str(year) for year in range(2014, years + 1))


def get_regexs():
    regexs = {}
    # Search for "YEAR" which exists in the boilerplate, but shouldn't in the real thing
    regexs["year"] = re.compile("YEAR")
    # get_dates return 2014, 2015, 2016, 2017, or 2018 until the current year
    # as a regex like: "(2014|2015|2016|2017|2018)";
    # company holder names can be anything
    regexs["date"] = re.compile(get_dates())
    # strip the following build constraints/tags:
    # //go:build
    # // +build \n\n
    regexs["go_build_constraints"] = re.compile(
        r"^(//(go:build| \+build).*\n)+\n", re.MULTILINE
    )
    # strip #!.* from scripts
    regexs["shebang"] = re.compile(r"^(#!.*\n)\n*", re.MULTILINE)
    # Search for generated files
    regexs["generated"] = re.compile(r"^[/*#]+ +.* DO NOT EDIT\.$", re.MULTILINE)
    return regexs


def main():
    regexs = get_regexs()
    refs = get_refs()
    filenames = get_files(refs)

    for filename in filenames:
        if not file_passes(filename, refs, regexs):
            print(filename)

    return 0


if __name__ == "__main__":
    sys.exit(main())
