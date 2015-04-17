#!/usr/bin/env python
import json
import sys
nodes = json.loads(sys.argv[1])
for num in nodes:
    print num['Stdout'].rstrip()

