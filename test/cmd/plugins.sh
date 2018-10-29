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

  # test plugins that overwrite existing kubectl commands
  output_message=$(! PATH=${PATH}:"test/fixtures/pkg/kubectl/plugins/version" kubectl plugin list 2>&1)
  kube::test::if_has_string "${output_message}" 'kubectl-version overwrites existing command: "kubectl version"'

  # test plugins that overwrite similarly-named plugins
  output_message=$(! PATH=${PATH}:"test/fixtures/pkg/kubectl/plugins:test/fixtures/pkg/kubectl/plugins/foo" kubectl plugin list 2>&1)
  kube::test::if_has_string "${output_message}" 'test/fixtures/pkg/kubectl/plugins/foo/kubectl-foo is overshadowed by a similarly named plugin'

  # test plugins with no warnings
  output_message=$(PATH=${PATH}:"test/fixtures/pkg/kubectl/plugins" kubectl plugin list 2>&1)
  kube::test::if_has_string "${output_message}" 'plugins are available'

  # no plugins
  output_message=$(! PATH=${PATH}:"test/fixtures/pkg/kubectl/plugins/empty" kubectl plugin list 2>&1)
  kube::test::if_has_string "${output_message}" 'unable to find any kubectl plugins in your PATH'

  # attempt to run a plugin in the user's PATH
  output_message=$(PATH=${PATH}:"test/fixtures/pkg/kubectl/plugins" kubectl foo)
  kube::test::if_has_string "${output_message}" 'plugin foo'

  # ensure that a kubectl command supersedes a plugin that overshadows it
  output_message=$(PATH=${PATH}:"test/fixtures/pkg/kubectl/plugins/version" kubectl version)
  kube::test::if_has_string "${output_message}" 'Client Version'
  kube::test::if_has_not_string "${output_message}" 'overshadows an existing plugin'

  set +o nounset
  set +o errexit
}
