#!/usr/bin/env bash

# Copyright 2018 The Kubernetes Authors.
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

set -o errexit
set -o nounset
set -o pipefail

run_plugins_tests() {
  set -o nounset
  set -o errexit

  kube::log::status "Testing kubectl plugins"

  # top-level plugin command
  output_message=$(KUBECTL_PLUGINS_PATH=test/fixtures/pkg/kubectl/plugins kubectl -h 2>&1)
  kube::test::if_has_string "${output_message}" 'plugin\s\+Runs a command-line plugin'

  # no plugins
  output_message=$(! kubectl plugin 2>&1)
  kube::test::if_has_string "${output_message}" 'no plugins installed'

  # single plugins path
  output_message=$(! KUBECTL_PLUGINS_PATH=test/fixtures/pkg/kubectl/plugins kubectl plugin 2>&1)
  kube::test::if_has_string "${output_message}" 'echo\s\+Echoes for test-cmd'
  kube::test::if_has_string "${output_message}" 'get\s\+The wonderful new plugin-based get!'
  kube::test::if_has_string "${output_message}" 'error\s\+The tremendous plugin that always fails!'
  kube::test::if_has_not_string "${output_message}" 'The hello plugin'
  kube::test::if_has_not_string "${output_message}" 'Incomplete plugin'
  kube::test::if_has_not_string "${output_message}" 'no plugins installed'

  # multiple plugins path
  output_message=$(KUBECTL_PLUGINS_PATH=test/fixtures/pkg/kubectl/plugins/:test/fixtures/pkg/kubectl/plugins2/ kubectl plugin -h 2>&1)
  kube::test::if_has_string "${output_message}" 'echo\s\+Echoes for test-cmd'
  kube::test::if_has_string "${output_message}" 'get\s\+The wonderful new plugin-based get!'
  kube::test::if_has_string "${output_message}" 'error\s\+The tremendous plugin that always fails!'
  kube::test::if_has_string "${output_message}" 'hello\s\+The hello plugin'
  kube::test::if_has_not_string "${output_message}" 'Incomplete plugin'

  # don't override existing commands
  output_message=$(KUBECTL_PLUGINS_PATH=test/fixtures/pkg/kubectl/plugins/:test/fixtures/pkg/kubectl/plugins2/ kubectl get -h 2>&1)
  kube::test::if_has_string "${output_message}" 'Display one or many resources'
  kube::test::if_has_not_string "$output_message{output_message}" 'The wonderful new plugin-based get'

  # plugin help
  output_message=$(KUBECTL_PLUGINS_PATH=test/fixtures/pkg/kubectl/plugins/:test/fixtures/pkg/kubectl/plugins2/ kubectl plugin hello -h 2>&1)
  kube::test::if_has_string "${output_message}" 'The hello plugin is a new plugin used by test-cmd to test multiple plugin locations.'
  kube::test::if_has_string "${output_message}" 'Usage:'

  # run plugin
  output_message=$(KUBECTL_PLUGINS_PATH=test/fixtures/pkg/kubectl/plugins/:test/fixtures/pkg/kubectl/plugins2/ kubectl plugin hello 2>&1)
  kube::test::if_has_string "${output_message}" '#hello#'
  output_message=$(KUBECTL_PLUGINS_PATH=test/fixtures/pkg/kubectl/plugins/:test/fixtures/pkg/kubectl/plugins2/ kubectl plugin echo 2>&1)
  kube::test::if_has_string "${output_message}" 'This plugin works!'
  output_message=$(! KUBECTL_PLUGINS_PATH=test/fixtures/pkg/kubectl/plugins/ kubectl plugin hello 2>&1)
  kube::test::if_has_string "${output_message}" 'unknown command'
  output_message=$(! KUBECTL_PLUGINS_PATH=test/fixtures/pkg/kubectl/plugins/ kubectl plugin error 2>&1)
  kube::test::if_has_string "${output_message}" 'error: exit status 1'

  # plugin tree
  output_message=$(! KUBECTL_PLUGINS_PATH=test/fixtures/pkg/kubectl/plugins kubectl plugin tree 2>&1)
  kube::test::if_has_string "${output_message}" 'Plugin with a tree of commands'
  kube::test::if_has_string "${output_message}" 'child1\s\+The first child of a tree'
  kube::test::if_has_string "${output_message}" 'child2\s\+The second child of a tree'
  kube::test::if_has_string "${output_message}" 'child3\s\+The third child of a tree'
  output_message=$(KUBECTL_PLUGINS_PATH=test/fixtures/pkg/kubectl/plugins kubectl plugin tree child1 --help 2>&1)
  kube::test::if_has_string "${output_message}" 'The first child of a tree'
  kube::test::if_has_not_string "${output_message}" 'The second child'
  kube::test::if_has_not_string "${output_message}" 'child2'
  output_message=$(KUBECTL_PLUGINS_PATH=test/fixtures/pkg/kubectl/plugins kubectl plugin tree child1 2>&1)
  kube::test::if_has_string "${output_message}" 'child one'
  kube::test::if_has_not_string "${output_message}" 'child1'
  kube::test::if_has_not_string "${output_message}" 'The first child'

  # plugin env
  output_message=$(KUBECTL_PLUGINS_PATH=test/fixtures/pkg/kubectl/plugins kubectl plugin env -h 2>&1)
  kube::test::if_has_string "${output_message}" "This is a flag 1"
  kube::test::if_has_string "${output_message}" "This is a flag 2"
  kube::test::if_has_string "${output_message}" "This is a flag 3"
  output_message=$(KUBECTL_PLUGINS_PATH=test/fixtures/pkg/kubectl/plugins kubectl plugin env --test1=value1 -t value2 2>&1)
  kube::test::if_has_string "${output_message}" 'KUBECTL_PLUGINS_CURRENT_NAMESPACE'
  kube::test::if_has_string "${output_message}" 'KUBECTL_PLUGINS_CALLER'
  kube::test::if_has_string "${output_message}" 'KUBECTL_PLUGINS_DESCRIPTOR_COMMAND=./env.sh'
  kube::test::if_has_string "${output_message}" 'KUBECTL_PLUGINS_DESCRIPTOR_SHORT_DESC=The plugin envs plugin'
  kube::test::if_has_string "${output_message}" 'KUBECTL_PLUGINS_GLOBAL_FLAG_KUBECONFIG'
  kube::test::if_has_string "${output_message}" 'KUBECTL_PLUGINS_GLOBAL_FLAG_REQUEST_TIMEOUT=0'
  kube::test::if_has_string "${output_message}" 'KUBECTL_PLUGINS_LOCAL_FLAG_TEST1=value1'
  kube::test::if_has_string "${output_message}" 'KUBECTL_PLUGINS_LOCAL_FLAG_TEST2=value2'
  kube::test::if_has_string "${output_message}" 'KUBECTL_PLUGINS_LOCAL_FLAG_TEST3=default'

  set +o nounset
  set +o errexit
}
