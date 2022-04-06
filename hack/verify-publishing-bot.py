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
import os
import sys


def get_gomod_dependencies(rootdir: str, components: list[str]):
    all_dependencies = {}
    for component in components:
        with open(os.path.join(rootdir, component, "go.mod")) as f:
            print(component, "dependencies")
            all_dependencies[component] = []
            lines = sorted(set(f))
            for line in lines:
                for dep in components:
                    if dep == component:
                        continue
                    if f"k8s.io/{dep} =>" not in line:
                        continue
                    print(f"\t{dep}")
                    if dep not in all_dependencies[component]:
                        all_dependencies[component].append(dep)
    return all_dependencies


def get_rules_dependencies(rules_file: str):
    import yaml
    with open(rules_file) as f:
        content = yaml.safe_load(f)
    return content


def main():
    file_dir = os.path.dirname(__file__)
    root_dir = os.path.abspath(os.path.join(file_dir, os.pardir))
    components_dir = os.path.join(root_dir, "staging/src/k8s.io/")
    print(root_dir, components_dir)
    components = sorted(os.listdir(components_dir))
    rules_file = os.path.join(root_dir, "staging/publishing/rules.yaml")
    try:
        import yaml
    except ImportError:
        print("Please install missing pyyaml module and re-run", sys.argv[0])
        sys.exit(1)

    rules_dependencies = get_rules_dependencies(rules_file)
    gomod_dependencies = get_gomod_dependencies(components_dir, components)

    processed_repos = []
    for rule in rules_dependencies["rules"]:
        branch = rule["branches"][0]

        # If this no longer exists in master
        if rule["destination"] not in gomod_dependencies:
            # Make sure we don't include a rule to publish it from master
            for branch in rule["branches"]:
                if branch["name"] == "master":
                    raise Exception("Cannot find master branch for destination", rule["destination"])
            # And skip validation of publishing rules for it
            continue

        if branch["name"] != "master":
            raise Exception("Cannot find master branch for destination", rule["destination"])
        if branch["source"]["branch"] != "master":
            raise Exception("Cannot find master source branch for destination", rule["destination"])

        # we specify the go version for all master branches through `default-go-version`
        # so ensure we don"t specify explicit go version for master branch in rules
        if "go" in branch:
            raise Exception("go version must not be specified for master branch for destination", rule["destination"])

        print("Processing:", rule["destination"])
        if rule["destination"] not in gomod_dependencies:
            raise Exception("missing go.mod for", rule["destination"])

        processed_repos.append(rule["destination"])
        processed_deps = []
        for dep in set(gomod_dependencies[rule["destination"]]):
            found = False
            if "dependencies" in branch:
                for dep2 in branch["dependencies"]:
                    processed_deps.append(dep2["repository"])
                    if dep2["branch"] != "master":
                        raise Exception(
                            f"Looking for master branch and found: {dep2} for destination {rule['destination']}",
                        )
                    if dep2["repository"] == dep:
                        found = True
            else:
                raise Exception(
                    f"Please add {gomod_dependencies[rule['destination']]} as dependencies under destination {rule['destination']} in {rules_file}"
                )

            if found:
                print(f"Found dependency {dep}")
            else:
                raise Exception(
                    f"Please add {dep} as a dependency under destination {rule['destination']} in {rules_file}"
                )

        extraDeps = set(processed_deps) - set(gomod_dependencies[rule["destination"]])
        if len(extraDeps) > 0:
            raise Exception(
                f"Extra dependencies in rules for {rule['destination']}: {','.join(str(s) for s in extraDeps)}"
            )

    items = set(gomod_dependencies.keys()) - set(processed_repos)
    if len(items) > 0:
        raise Exception(f"Missing rules for {','.join(str(s) for s in items)}")
    print("Done")


if __name__ == "__main__":
    main()
