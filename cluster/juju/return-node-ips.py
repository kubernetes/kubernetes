#!/usr/bin/env python
import json
import sys
# This script helps parse out the private IP addreses from the
# `juju run` command's JSON object, see cluster/juju/util.sh

if len(sys.argv) > 1:
    # It takes the JSON output as the first argument.
    nodes = json.loads(sys.argv[1])
    # There can be multiple nodes to print the Stdout.
    for num in nodes:
        print num['Stdout'].rstrip()
else:
    exit(1)
