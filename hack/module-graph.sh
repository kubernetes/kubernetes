#!/usr/bin/env bash

# Copyright 2020 The Kubernetes Authors.
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

# This script generates a jpg image of our internal dependency graph.
# It relies on go mod and dot to do so and should be run from K8s root.
# TODO: Containerize the script to remove dependency issues with go mod and dot.

# To generate graph with all Kubernetes modules run
# ./hack/module-graph.sh
# To generate graph with just staging modules run
# ./hack/module-graph.sh staging

error_exit()
{
	echo "$1" 1>&2
	exit 1
}

staging_dependencies()
{
	relevant_modules="(k8s.io/kubernetes|$(find staging/src/k8s.io -maxdepth 1 -mindepth 1 | sed -E 's|staging/src/k8s.io|k8s.io|g' | tr '\n' '|' | sed 's#|$##' ))"
	# Generating lines of the form " k8s_io_kubernetes -> k8s_io_api"
	# Use only the directories in staging
	# Trimming away version info
	# Replacing non DOT (graph description language) characters with underscores
	# Dedupe lines
	# Inserting needed arrow.
	# Indenting the line appropriately
	go mod graph | grep -E "^${relevant_modules}(@.*| )${relevant_modules}@.*$" | sed -E 's|@\S+ | |g' | sed -E 's|@\S+$||g' | sed -E 's/[\.\/\-]/_/g' | sort | uniq | sed -E 's| | -> |g' | sed -E 's|^|    |g' >> _output/module-dependencies.dot || error_exit "Failed to generate staging dependencies in DOT file"
}

kubernetes_dependencies()
{
	# Generating lines of the form " k8s_io_kubernetes -> k8s_io_api"
	# Excluding all non Kubernetes dependencies
	# Trimming away version info
	# Replacing non DOT (graph description language) characters with underscores
	# Dedupe lines
	# Inserting needed arrow.
	# Indenting the line appropriately
	go mod graph | grep -E "^.*k8s.io.*k8s.io.*$" | sed -E 's|@\S+ | |g' | sed -E 's|@\S+$||g' | sed -E 's/[\.\/\-]/_/g' | sort | uniq | sed -E 's| | -> |g' | sed -E 's|^|    |g' >> _output/module-dependencies.dot || error_exit "Failed to generate kubernetes dependencies in DOT file"
}

mkdir -p _output
echo "digraph module_dependencies {" > _output/module-dependencies.dot || error_exit "Failed to open DOT file"
if [[ -n "$1" && $1 == "staging" ]]; then
	echo "Generating just staging modules"
	staging_dependencies
else
	echo "Generating all Kubernetes modules"
	kubernetes_dependencies
fi
echo "}" >> _output/module-dependencies.dot || error_exit "Failed to close DOT file"
dot -Gratio=1,1 -Nwidth=2 -Nheight=2 -Nfontsize=48 -Earrowsize=8 -Tjpg _output/module-dependencies.dot -o _output/module-dependencies.jpg || error_exit "Failed to generate graph from DOT file"
