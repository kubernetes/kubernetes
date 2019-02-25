#!/usr/bin/env bash

# Copyright 2014 The Kubernetes Authors.
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

# Generate Docker daemon options based on flannel env file.

# exit on any error
set -e

usage() {
  echo "$0 [-f FLANNEL-ENV-FILE] [-d DOCKER-ENV-FILE] [-i] [-c] [-m] [-k COMBINED-KEY]

Generate Docker daemon options based on flannel env file
OPTIONS:
    -f  Path to flannel env file. Defaults to /run/flannel/subnet.env
    -d  Path to Docker env file to write to. Defaults to /run/docker_opts.env
    -i  Output each Docker option as individual var. e.g. DOCKER_OPT_MTU=1500
    -c  Output combined Docker options into DOCKER_OPTS var
    -k  Set the combined options key to this value (default DOCKER_OPTS=)
    -m  Do not output --ip-masq (useful for older Docker version)
" >/dev/stderr
  exit 1
}

flannel_env="/run/flannel/subnet.env"
docker_env="/run/docker_opts.env"
combined_opts_key="DOCKER_OPTS"
indiv_opts=false
combined_opts=false
ipmasq=true
val=""

while getopts "f:d:icmk:" opt; do
  case $opt in
    f)
      flannel_env=$OPTARG
      ;;
    d)
      docker_env=$OPTARG
      ;;
    i)
      indiv_opts=true
      ;;
    c)
      combined_opts=true
      ;;
    m)
      ipmasq=false
      ;;
    k)
      combined_opts_key=$OPTARG
      ;;
    \?)
      usage
      ;;
  esac
done

if [[ $indiv_opts = false ]] && [[ $combined_opts = false ]]; then
  indiv_opts=true
  combined_opts=true
fi

if [[ -f "${flannel_env}" ]]; then
  source "${flannel_env}"
fi

if [[ -n "$FLANNEL_SUBNET" ]]; then
  # shellcheck disable=SC2034  # Variable name referenced in OPT_LOOP below
  DOCKER_OPT_BIP="--bip=$FLANNEL_SUBNET"
fi

if [[ -n "$FLANNEL_MTU" ]]; then
  # shellcheck disable=SC2034  # Variable name referenced in OPT_LOOP below
  DOCKER_OPT_MTU="--mtu=$FLANNEL_MTU"
fi

if [[ "$FLANNEL_IPMASQ" = true ]] && [[ $ipmasq = true ]]; then
  # shellcheck disable=SC2034  # Variable name referenced in OPT_LOOP below
  DOCKER_OPT_IPMASQ="--ip-masq=false"
fi

eval docker_opts="\$${combined_opts_key}"
docker_opts+=" "

echo -n "" >"${docker_env}"

# OPT_LOOP
for opt in $(compgen -v DOCKER_OPT_); do
  eval val=\$"${opt}"

  if [[ "$indiv_opts" = true ]]; then
    echo "$opt=\"$val\"" >>"${docker_env}"
  fi

  docker_opts+="$val "
done

if [[ "$combined_opts" = true ]]; then
  echo "${combined_opts_key}=\"${docker_opts}\"" >>"${docker_env}"
fi
