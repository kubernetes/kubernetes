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

# Script to print out PR info in release note format.

import json
import sys
import urllib2

PULLQUERY=("https://api.github.com/repos/"
           "kubernetes/kubernetes/pulls/{pull}")
LOGIN="login"
TITLE="title"
USER="user"

def print_pulls(pulls):
  for pull in pulls:
    d = json.loads(urllib2.urlopen(PULLQUERY.format(pull=pull)).read())
    print "* {title} #{pull} ({author})".format(
        title=d[TITLE], pull=pull, author=d[USER][LOGIN])

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print ("Usage: {cmd} <pulls>...: Prints out short " +
           "markdown description for PRs appropriate for release notes.")
    sys.exit(1)
  print_pulls(sys.argv[1:])
