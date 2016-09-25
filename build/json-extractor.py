#!/usr/bin/env python

# Copyright 2014 The Kubernetes Authors.
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

# This is a very simple utility that reads a JSON document from stdin, parses it
# and returns the specified value.  The value is described using a simple dot
# notation.  If any errors are encountered along the way, an error is output and
# a failure value is returned.

from __future__ import print_function

import json
import sys

def PrintError(*err):
  print(*err, file=sys.stderr)

def main():
  try:
    obj = json.load(sys.stdin)
  except Exception, e:
    PrintError("Error loading JSON: {0}".format(str(e)))

  if len(sys.argv) == 1:
    # if we don't have a query string, return success
    return 0
  elif len(sys.argv) > 2:
    PrintError("Usage: {0} <json query>".format(sys.args[0]))
    return 1

  query_list = sys.argv[1].split('.')
  for q in query_list:
    if isinstance(obj, dict):
      if q not in obj:
        PrintError("Couldn't find '{0}' in dict".format(q))
        return 1
      obj = obj[q]
    elif isinstance(obj, list):
      try:
        index = int(q)
      except:
        PrintError("Can't use '{0}' to index into array".format(q))
        return 1
      if index >= len(obj):
        PrintError("Index ({0}) is greater than length of list ({1})".format(q, len(obj)))
        return 1
      obj = obj[index]
    else:
      PrintError("Trying to query non-queryable object: {0}".format(q))
      return 1

  if isinstance(obj, str):
    print(obj)
  else:
    print(json.dumps(obj, indent=2))

if __name__ == "__main__":
  sys.exit(main())
