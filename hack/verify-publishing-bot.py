#!/usr/bin/env python

# Copyright 2019 The Kubernetes Authors.
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

import fnmatch
import os
import sys
import json
import yaml


def get_godeps_dependencies(godeps):
    all_dependencies = {}
    for arg in godeps:
        with open(arg) as f:
            data = json.load(f)
            dependencies = []
            for dep in data["Deps"]:
                if dep['Rev'].startswith('xxxx') and dep['ImportPath'].startswith("k8s.io"):
                    package = "/".join(dep['ImportPath'].split('/')[0:2])
                    if package not in dependencies:
                        dependencies.append(package.split('/')[1])
            all_dependencies[(data["ImportPath"].split('/')[1])] = dependencies
    return all_dependencies


def get_rules_dependencies(rules_file):
    with open(rules_file) as f:
        data = yaml.load(f)
    return data


def main():
    rootdir = os.path.dirname(__file__) + "/../"
    rootdir = os.path.abspath(rootdir)

    godeps = []
    for root, dirnames, filenames in os.walk(rootdir + '/staging/'):
        for filename in fnmatch.filter(filenames, 'Godeps.json'):
            godeps.append(os.path.join(root, filename))

    godep_dependencies = get_godeps_dependencies(godeps)
    rules_dependencies = get_rules_dependencies(rootdir + "/staging/publishing/rules.yaml")

    processed_repos = []
    for rule in rules_dependencies["rules"]:
        branch = rule["branches"][0]
        if branch["name"] != "master":
            raise Exception("cannot find master branch for destination %s" % rule["destination"])
        if branch["source"]["branch"] != "master":
            raise Exception("cannot find master source branch for destination %s" % rule["destination"])

        print("processing : %s" % rule["destination"])
        if rule["destination"] not in godep_dependencies:
            raise Exception("missing Godeps.json for %s" % rule["destination"])
        processed_repos.append(rule["destination"])
        for dep in set(godep_dependencies[rule["destination"]]):
            found = False
            if "dependencies" in branch:
                for dep2 in branch["dependencies"]:
                    if dep2["branch"] != "master":
                        raise Exception("Looking for master branch and found : %s for destination", dep2,
                                        rule["destination"])
                    if dep2["repository"] == dep:
                        found = True
            else:
                raise Exception(
                    "destination %s does not have any dependencies (looking for %s)" % (rule["destination"], dep))
            if not found:
                raise Exception("destination %s does not have dependency %s" % (rule["destination"], dep))
            else:
                print("  found dependency %s" % dep)
    items = set(godep_dependencies.keys()) - set(processed_repos)
    if len(items) > 0:
        raise Exception("missing rules for %s" % ','.join(str(s) for s in items))
    print("Done.")


if __name__ == "__main__":
    sys.exit(main())
