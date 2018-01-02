# Copyright (c) 2015 VMware, Inc. All Rights Reserved.
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

# Provide a simple shell extension to save and load govc
# environments to disk. No more running `export GOVC_ABC=xyz`
# in different shells over and over again. Loading the right
# govc environment variables is now only one short and
# autocompleted command away!
#
# Usage:
# * Source this file from your `~/.bashrc` or running shell.
# * Execute `govc-env` to print GOVC_* variables.
# * Execute `govc-env --save <name>` to save GOVC_* variables.
# * Execute `govc-env <name>` to load GOVC_* variables.
#

_govc_env_dir=$HOME/.govmomi/env
mkdir -p "${_govc_env_dir}"

_govc-env-complete() {
  local w="${COMP_WORDS[COMP_CWORD]}"
  local c="$(find ${_govc_env_dir} -mindepth 1 -maxdepth 1 -type f  | sort | xargs -r -L1 basename | xargs echo)"

  # Only allow completion if preceding argument if the function itself
  if [ "$3" == "govc-env" ]; then
    COMPREPLY=( $(compgen -W "${c}" -- "${w}") )
  fi
}

govc-env() {
  # Print current environment
  if [ -z "$1" ]; then
    for VAR in $(env | grep ^GOVC_ | cut -d= -f1); do
      echo "export ${VAR}='${!VAR}'"
    done

    return
  fi

  # Save current environment
  if [ "$1" == "--save" ]; then
    govc-env > ${_govc_env_dir}/$2
    return
  fi

  # Load specified environment
  source ${_govc_env_dir}/$1
}

complete -F _govc-env-complete govc-env

