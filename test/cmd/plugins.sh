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

  # Create a folder which only contains our kubectl executable
  TEMP_PATH=$(mktemp -d /tmp/tmp-kubectl-path-XXXXX)
  ln -s "$(which kubectl)" "${TEMP_PATH}/kubectl"

  # test plugins that overwrite existing kubectl commands
  output_message=$(! PATH=${TEMP_PATH}:"test/fixtures/pkg/kubectl/plugins/version" kubectl plugin list 2>&1)
  kube::test::if_has_string "${output_message}" 'kubectl-version overwrites existing command: "kubectl version"'

  # test plugins that overwrite similarly-named plugins
  output_message=$(! PATH=${TEMP_PATH}:"test/fixtures/pkg/kubectl/plugins:test/fixtures/pkg/kubectl/plugins/foo" kubectl plugin list 2>&1)
  kube::test::if_has_string "${output_message}" 'test/fixtures/pkg/kubectl/plugins/foo/kubectl-foo is overshadowed by a similarly named plugin'

  # test plugins with no warnings
  output_message=$(PATH=${TEMP_PATH}:"test/fixtures/pkg/kubectl/plugins" kubectl plugin list 2>&1)
  kube::test::if_has_string "${output_message}" 'plugins are available'

  # no plugins
  output_message=$(! PATH=${TEMP_PATH}:"test/fixtures/pkg/kubectl/plugins/empty" kubectl plugin list 2>&1)
  kube::test::if_has_string "${output_message}" 'unable to find any kubectl plugins in your PATH'

  # attempt to run a plugin in the user's PATH
  output_message=$(PATH=${TEMP_PATH}:"test/fixtures/pkg/kubectl/plugins" kubectl foo)
  kube::test::if_has_string "${output_message}" 'plugin foo'

  # check arguments passed to the plugin
  output_message=$(PATH=${PATH}:"test/fixtures/pkg/kubectl/plugins/bar" kubectl bar arg1)
  kube::test::if_has_string "${output_message}" 'test/fixtures/pkg/kubectl/plugins/bar/kubectl-bar arg1'

  # ensure that a kubectl command supersedes a plugin that overshadows it
  output_message=$(PATH=${TEMP_PATH}:"test/fixtures/pkg/kubectl/plugins/version" kubectl version --client)
  kube::test::if_has_string "${output_message}" 'Client Version'
  kube::test::if_has_not_string "${output_message}" 'overshadows an existing plugin'

  # attempt to run a plugin as a subcommand of kubectl create in the user's PATH
  output_message=$(PATH=${TEMP_PATH}:"test/fixtures/pkg/kubectl/plugins/create" kubectl create foo)
  kube::test::if_has_string "${output_message}" 'plugin foo as a subcommand of kubectl create command'

  # ensure that a kubectl create cronjob builtin command supersedes a plugin that overshadows it
  output_message=$(PATH=${TEMP_PATH}:"test/fixtures/pkg/kubectl/plugins/create" kubectl create cronjob --help)
  kube::test::if_has_not_string "${output_message}" 'plugin cronjob as a subcommand of kubectl create command'

  rm -fr "${TEMP_PATH}"

  set +o nounset
  set +o errexit
}
