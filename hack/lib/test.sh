#!/bin/bash

# Copyright 2015 The Kubernetes Authors.
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

# A set of helpers for tests

readonly reset=$(tput sgr0)
readonly  bold=$(tput bold)
readonly black=$(tput setaf 0)
readonly   red=$(tput setaf 1)
readonly green=$(tput setaf 2)

kube::test::clear_all() {
  if kube::test::if_supports_resource "rc" ; then
    kubectl delete "${kube_flags[@]}" rc --all --grace-period=0 --force
  fi
  if kube::test::if_supports_resource "pods" ; then
    kubectl delete "${kube_flags[@]}" pods --all --grace-period=0 --force
  fi
}

# Prints the calling file and line number $1 levels deep
# Defaults to 2 levels so you can call this to find your own caller
kube::test::get_caller() {
  local levels=${1:-2}
  local caller_file="${BASH_SOURCE[$levels]}"
  local caller_line="${BASH_LINENO[$levels-1]}"
  echo "$(basename "${caller_file}"):${caller_line}"
}

# Force exact match of a returned result for a object query.  Wrap this with || to support multiple
# valid return types.
# This runs `kubectl get` once and asserts that the result is as expected.
## $1: Object on which get should be run
# $2: The go-template to run on the result
# $3: The expected output
# $4: Additional args to be passed to kubectl
kube::test::get_object_assert() {
  kube::test::object_assert 1 "$@"
}

# Asserts that the output of a given get query is as expected.
# Runs the query multiple times before failing it.
# $1: Object on which get should be run
# $2: The go-template to run on the result
# $3: The expected output
# $4: Additional args to be passed to kubectl
kube::test::wait_object_assert() {
  kube::test::object_assert 10 "$@"
}

# Asserts that the output of a given get query is as expected.
# Can run the query multiple times before failing it.
# $1: Number of times the query should be run before failing it.
# $2: Object on which get should be run
# $3: The go-template to run on the result
# $4: The expected output
# $5: Additional args to be passed to kubectl
kube::test::object_assert() {
  local tries=$1
  local object=$2
  local request=$3
  local expected=$4
  local args=${5:-}

  for j in $(seq 1 ${tries}); do
    res=$(eval kubectl get -a "${kube_flags[@]}" ${args} $object -o go-template=\"$request\")
    if [[ "$res" =~ ^$expected$ ]]; then
        echo -n ${green}
        echo "$(kube::test::get_caller 3): Successful get $object $request: $res"
        echo -n ${reset}
        return 0
    fi
    echo "Waiting for Get $object $request $args: expected: $expected, got: $res"
    sleep $((${j}-1))
  done

  echo ${bold}${red}
  echo "$(kube::test::get_caller 3): FAIL!"
  echo "Get $object $request"
  echo "  Expected: $expected"
  echo "  Got:      $res"
  echo ${reset}${red}
  caller
  echo ${reset}
  return 1
}

kube::test::get_object_jsonpath_assert() {
  local object=$1
  local request=$2
  local expected=$3

  res=$(eval kubectl get -a "${kube_flags[@]}" $object -o jsonpath=\"$request\")

  if [[ "$res" =~ ^$expected$ ]]; then
      echo -n ${green}
      echo "$(kube::test::get_caller): Successful get $object $request: $res"
      echo -n ${reset}
      return 0
  else
      echo ${bold}${red}
      echo "$(kube::test::get_caller): FAIL!"
      echo "Get $object $request"
      echo "  Expected: $expected"
      echo "  Got:      $res"
      echo ${reset}${red}
      caller
      echo ${reset}
      return 1
  fi
}

kube::test::describe_object_assert() {
  local resource=$1
  local object=$2
  local matches=${@:3}

  result=$(eval kubectl describe "${kube_flags[@]}" $resource $object)

  for match in ${matches}; do
    if [[ ! $(echo "$result" | grep ${match}) ]]; then
      echo ${bold}${red}
      echo "$(kube::test::get_caller): FAIL!"
      echo "Describe $resource $object"
      echo "  Expected Match: $match"
      echo "  Not found in:"
      echo "$result"
      echo ${reset}${red}
      caller
      echo ${reset}
      return 1
    fi
  done

  echo -n ${green}
  echo "$(kube::test::get_caller): Successful describe $resource $object:"
  echo "$result"
  echo -n ${reset}
  return 0
}

kube::test::describe_object_events_assert() {
    local resource=$1
    local object=$2
    local showevents=${3:-"true"}

    if [[ -z "${3:-}" ]]; then
        result=$(eval kubectl describe "${kube_flags[@]}" $resource $object)
    else
        result=$(eval kubectl describe "${kube_flags[@]}" "--show-events=$showevents" $resource $object)
    fi

    if [[ -n $(echo "$result" | grep "No events.\|Events:") ]]; then
        local has_events="true"
    else
        local has_events="false"
    fi
    if [[ $showevents == $has_events ]]; then
        echo -n ${green}
        echo "$(kube::test::get_caller): Successful describe"
        echo "$result"
        echo ${reset}
        return 0
    else
        echo ${bold}${red}
        echo "$(kube::test::get_caller): FAIL"
        if [[ $showevents == "false" ]]; then
            echo "  Events information should not be described in:"
        else
            echo "  Events information not found in:"
        fi
        echo $result
        echo ${reset}${red}
        caller
        echo ${reset}
        return 1
    fi
}

kube::test::describe_resource_assert() {
  local resource=$1
  local matches=${@:2}

  result=$(eval kubectl describe "${kube_flags[@]}" $resource)

  for match in ${matches}; do
    if [[ ! $(echo "$result" | grep ${match}) ]]; then
      echo ${bold}${red}
      echo "FAIL!"
      echo "Describe $resource"
      echo "  Expected Match: $match"
      echo "  Not found in:"
      echo "$result"
      echo ${reset}${red}
      caller
      echo ${reset}
      return 1
    fi
  done

  echo -n ${green}
  echo "Successful describe $resource:"
  echo "$result"
  echo -n ${reset}
  return 0
}

kube::test::describe_resource_events_assert() {
    local resource=$1
    local showevents=${2:-"true"}

    result=$(eval kubectl describe "${kube_flags[@]}" "--show-events=$showevents" $resource)

    if [[ $(echo "$result" | grep "No events.\|Events:") ]]; then
        local has_events="true"
    else
        local has_events="false"
    fi
    if [[ $showevents == $has_events ]]; then
        echo -n ${green}
        echo "Successful describe"
        echo "$result"
        echo -n ${reset}
        return 0
    else
        echo ${bold}${red}
        echo "FAIL"
        if [[ $showevents == "false" ]]; then
            echo "  Events information should not be described in:"
        else
            echo "  Events information not found in:"
        fi
        echo $result
        caller
        echo ${reset}
        return 1
    fi
}

kube::test::if_has_string() {
  local message=$1
  local match=$2

  if echo "$message" | grep -q "$match"; then
    echo "Successful"
    echo "message:$message"
    echo "has:$match"
    return 0
  else
    echo "FAIL!"
    echo "message:$message"
    echo "has not:$match"
    caller
    return 1
  fi
}

kube::test::if_has_not_string() {
  local message=$1
  local match=$2

  if echo "$message" | grep -q "$match"; then
    echo "FAIL!"
    echo "message:$message"
    echo "has:$match"
    caller
    return 1
  else
    echo "Successful"
    echo "message:$message"
    echo "has not:$match"
    return 0
  fi
}

# Returns true if the required resource is part of supported resources.
# Expects env vars:
#   SUPPORTED_RESOURCES: Array of all resources supported by the apiserver. "*"
#   means it supports all resources. For ex: ("*") or ("rc" "*") both mean that
#   all resources are supported.
#   $1: Name of the resource to be tested.
kube::test::if_supports_resource() {
  SUPPORTED_RESOURCES=${SUPPORTED_RESOURCES:-""}
  REQUIRED_RESOURCE=${1:-""}

  for r in "${SUPPORTED_RESOURCES[@]}"; do
    if [[ "${r}" == "*" || "${r}" == "${REQUIRED_RESOURCE}" ]]; then
      return 0
    fi
  done
  return 1
}
