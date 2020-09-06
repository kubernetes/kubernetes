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


def get_gomod_dependencies(rootdir, components):
    all_dependencies = {}
    for component in components:
        with open(os.path.join(rootdir, component, "go.mod")) as f:
            print(component + " dependencies")
            all_dependencies[component] = []
            lines = list(set(f))
            lines.sort()
            for line in lines:
                for dep in components:
                    if dep == component:
                        continue
                    if ("k8s.io/" + dep + " =>") not in line:
                        continue
                    print("\t"+dep)
                    if dep not in all_dependencies[component]:
                        all_dependencies[component].append(dep)
    return all_dependencies


def get_rules_dependencies(rules_file):
    import yaml
    with open(rules_file) as f:
        data = yaml.load(f)
    return data


def main():
    rootdir = os.path.dirname(__file__) + "/../"
    rootdir = os.path.abspath(rootdir)

    components = []
    for component in os.listdir(rootdir + '/staging/src/k8s.io/'):
        components.append(component)
    components.sort()

    rules_file = "/staging/publishing/rules.yaml"
    try:
        import yaml
    except ImportError:
        print("Please install missing pyyaml module and re-run %s" % sys.argv[0])
        sys.exit(1)
    rules_dependencies = get_rules_dependencies(rootdir + rules_file)

    gomod_dependencies = get_gomod_dependencies(rootdir + '/staging/src/k8s.io/', components)

    processed_repos = []
    for rule in rules_dependencies["rules"]:
        branch = rule["branches"][0]

        # If this no longer exists in master
        if rule["destination"] not in gomod_dependencies:
            # Make sure we don't include a rule to publish it from master
            for branch in rule["branches"]:
                if branch["name"] == "master":
                    raise Exception("cannot find master branch for destination %s" % rule["destination"])
            # And skip validation of publishing rules for it
            continue

        if branch["name"] != "master":
            raise Exception("cannot find master branch for destination %s" % rule["destination"])
        if branch["source"]["branch"] != "master":
            raise Exception("cannot find master source branch for destination %s" % rule["destination"])

        # we specify the go version for all master branches through `default-go-version`
        # so ensure we don't specify explicit go version for master branch in rules
        if "go" in branch:
            raise Exception("go version must not be specified for master branch for destination %s" % rule["destination"])

        print("processing : %s" % rule["destination"])
        if rule["destination"] not in gomod_dependencies:
            raise Exception("missing go.mod for %s" % rule["destination"])
        processed_repos.append(rule["destination"])
        processed_deps = []
        for dep in set(gomod_dependencies[rule["destination"]]):
            found = False
            if "dependencies" in branch:
                for dep2 in branch["dependencies"]:
                    processed_deps.append(dep2["repository"])
                    if dep2["branch"] != "master":
                        raise Exception("Looking for master branch and found : %s for destination", dep2,
                                        rule["destination"])
                    if dep2["repository"] == dep:
                        found = True
            else:
                raise Exception(
                    "Please add %s as dependencies under destination %s in %s" % (gomod_dependencies[rule["destination"]], rule["destination"], rules_file))
            if not found:
                raise Exception("Please add %s as a dependency under destination %s in %s" % (dep, rule["destination"], rules_file))
            else:
                print("  found dependency %s" % dep)
        extraDeps = set(processed_deps) - set(gomod_dependencies[rule["destination"]])
        if len(extraDeps) > 0:
            raise Exception("extra dependencies in rules for %s: %s" % (rule["destination"], ','.join(str(s) for s in extraDeps)))
    items = set(gomod_dependencies.keys()) - set(processed_repos)
    if len(items) > 0:
        raise Exception("missing rules for %s" % ','.join(str(s) for s in items))
    print("Done.")


if __name__ == "__main__":
    sys.exit(main())
