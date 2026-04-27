#!/usr/bin/env bash

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

# shellcheck disable=SC2034 # Variables sourced in other scripts.

# A set of helpers for tests

reset=$(tput sgr0)
bold=$(tput bold)
black=$(tput setaf 0)
red=$(tput setaf 1)
green=$(tput setaf 2)
readonly reset bold black red green

kube::test::clear_all() {
  if kube::test::if_supports_resource "rc" ; then
    # shellcheck disable=SC2154
    # Disabling because "kube_flags" is set in a parent script
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
  local caller_file="${BASH_SOURCE[${levels}]}"
  local caller_line="${BASH_LINENO[${levels}-1]}"
  echo "$(basename "${caller_file}"):${caller_line}"
}

# Force exact match of a returned result for a object query.  Wrap this with || to support multiple
# valid return types.
# This runs `kubectl get` once and asserts that the result is as expected.
# $1: Object on which get should be run
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

  for j in $(seq 1 "${tries}"); do
    # shellcheck disable=SC2086
    # Disabling because to allow for expansion here
    res=$(kubectl get "${kube_flags[@]}" ${args} ${object} -o go-template="${request}")
    if [[ "${res}" =~ ^$expected$ ]]; then
        echo -n "${green}"
        echo "$(kube::test::get_caller 3): Successful get ${object} ${request}: ${res}"
        echo -n "${reset}"
        return 0
    fi
    echo "Waiting for Get ${object} ${request} ${args}: expected: ${expected}, got: ${res}"
    sleep $((j-1))
  done

  echo "${bold}${red}"
  echo "$(kube::test::get_caller 3): FAIL!"
  echo "Get ${object} ${request}"
  echo "  Expected: ${expected}"
  echo "  Got:      ${res}"
  echo "${reset}${red}"
  caller
  echo "${reset}"
  return 1
}

kube::test::get_object_jsonpath_assert() {
  local object=$1
  local request=$2
  local expected=$3

  # shellcheck disable=SC2086
  # Disabling to allow for expansion here
  res=$(kubectl get "${kube_flags[@]}" ${object} -o jsonpath=${request})

  if [[ "${res}" =~ ^$expected$ ]]; then
      echo -n "${green}"
      echo "$(kube::test::get_caller): Successful get ${object} ${request}: ${res}"
      echo -n "${reset}"
      return 0
  else
      echo "${bold}${red}"
      echo "$(kube::test::get_caller): FAIL!"
      echo "Get ${object} ${request}"
      echo "  Expected: ${expected}"
      echo "  Got:      ${res}"
      echo "${reset}${red}"
      caller
      echo "${reset}"
      return 1
  fi
}

kube::test::describe_object_assert() {
  local resource=$1
  local object=$2
  local matches=( "${@:3}" )

  # shellcheck disable=SC2086
  # Disabling to allow for expansion here
  result=$(kubectl describe "${kube_flags[@]}" ${resource} ${object})

  for match in "${matches[@]}"; do
    if grep -q "${match}" <<< "${result}"; then
      echo "matched ${match}"
    else
      echo "${bold}${red}"
      echo "$(kube::test::get_caller): FAIL!"
      echo "Describe ${resource} ${object}"
      echo "  Expected Match: ${match}"
      echo "  Not found in:"
      echo "${result}"
      echo "${reset}${red}"
      caller
      echo "${reset}"
      return 1
    fi
  done

  echo -n "${green}"
  echo "$(kube::test::get_caller): Successful describe ${resource} ${object}:"
  echo "${result}"
  echo -n "${reset}"
  return 0
}

kube::test::describe_object_events_assert() {
    local resource=$1
    local object=$2
    local showevents=${3:-"true"}

  # shellcheck disable=SC2086
  # Disabling to allow for expansion here
    if [[ -z "${3:-}" ]]; then
        result=$(kubectl describe "${kube_flags[@]}" ${resource} ${object})
    else
        result=$(kubectl describe "${kube_flags[@]}" "--show-events=${showevents}" ${resource} ${object})
    fi

    if grep -q "No events.\|Events:" <<< "${result}"; then
        local has_events="true"
    else
        local has_events="false"
    fi
    if [[ "${showevents}" == "${has_events}" ]]; then
        echo -n "${green}"
        echo "$(kube::test::get_caller): Successful describe"
        echo "${result}"
        echo "${reset}"
        return 0
    else
        echo "${bold}${red}"
        echo "$(kube::test::get_caller): FAIL"
        if [[ "${showevents}" == "false" ]]; then
            echo "  Events information should not be described in:"
        else
            echo "  Events information not found in:"
        fi
        echo "${result}"
        echo "${reset}${red}"
        caller
        echo "${reset}"
        return 1
    fi
}

kube::test::describe_resource_assert() {
  local resource=$1
  local matches=( "${@:2}" )

  # shellcheck disable=SC2086
  # Disabling to allow for expansion here
  result=$(kubectl describe "${kube_flags[@]}" ${resource})

  for match in "${matches[@]}"; do
    if grep -q "${match}" <<< "${result}"; then
      echo "matched ${match}"
    else
      echo "${bold}${red}"
      echo "FAIL!"
      echo "Describe ${resource}"
      echo "  Expected Match: ${match}"
      echo "  Not found in:"
      echo "${result}"
      echo "${reset}${red}"
      caller
      echo "${reset}"
      return 1
    fi
  done

  echo -n "${green}"
  echo "Successful describe ${resource}:"
  echo "${result}"
  echo -n "${reset}"
  return 0
}

kube::test::describe_resource_events_assert() {
    local resource=$1
    local showevents=${2:-"true"}

    # shellcheck disable=SC2086
    # Disabling to allow for expansion here
    result=$(kubectl describe "${kube_flags[@]}" "--show-events=${showevents}" ${resource})

    if grep -q "No events.\|Events:" <<< "${result}"; then
        local has_events="true"
    else
        local has_events="false"
    fi
    if [[ "${showevents}" == "${has_events}" ]]; then
        echo -n "${green}"
        echo "Successful describe"
        echo "${result}"
        echo -n "${reset}"
        return 0
    else
        echo "${bold}${red}"
        echo "FAIL"
        if [[ "${showevents}" == "false" ]]; then
            echo "  Events information should not be described in:"
        else
            echo "  Events information not found in:"
        fi
        echo "${result}"
        caller
        echo "${reset}"
        return 1
    fi
}

kube::test::describe_resource_chunk_size_assert() {
  # $1: the target resource
  local resource=$1
  # $2: comma-separated list of additional resources that will be listed
  local additionalResources=${2:-}
  # Remaining args are flags to pass to kubectl
  local args=${3:-}

  # Expect list requests for the target resource and the additional resources
  local expectLists
  IFS="," read -r -a expectLists <<< "${resource},${additionalResources}"

  # shellcheck disable=SC2086
  # Disabling to allow for expansion here
  defaultResult=$(kubectl describe ${resource} --show-events=true -v=6 ${args} "${kube_flags[@]}" 2>&1 >/dev/null)
  for r in "${expectLists[@]}"; do
    if grep -q "${r}?.*limit=500" <<< "${defaultResult}"; then
      echo "query for ${r} had limit param"
    else
      echo "${bold}${red}"
      echo "FAIL!"
      echo "Describe ${resource}"
      echo "  Expected limit param on request for: ${r}"
      echo "  Not found in:"
      echo "${defaultResult}"
      echo "${reset}${red}"
      caller
      echo "${reset}"
      return 1
    fi
  done

  # shellcheck disable=SC2086
  # Disabling to allow for expansion here
  # Try a non-default chunk size
  customResult=$(kubectl describe ${resource} --show-events=false --chunk-size=10 -v=6 ${args} "${kube_flags[@]}" 2>&1 >/dev/null)
  if grep -q "${resource}?limit=10" <<< "${customResult}"; then
    echo "query for ${resource} had user-specified limit param"
  else
    echo "${bold}${red}"
    echo "FAIL!"
    echo "Describe ${resource}"
    echo "  Expected limit param on request for: ${r}"
    echo "  Not found in:"
    echo "${customResult}"
    echo "${reset}${red}"
    caller
    echo "${reset}"
    return 1
  fi

  echo -n "${green}"
  echo "Successful describe ${resource} verbose logs:"
  echo "${defaultResult}"
  echo -n "${reset}"

  return 0
}

# Compare sort-by resource name output (first column, skipping first line) with expected order specify in the last parameter
kube::test::if_sort_by_has_correct_order() {
  local var
  var="$(echo "$1" | awk '{if(NR!=1) print $1}' | tr '\n' ':')"
  kube::test::if_has_string "${var}" "${@:$#}"
}

kube::test::if_has_string() {
  local message=$1
  local match=$2

  if grep -q "${match}" <<< "${message}"; then
    echo -n "${green}"
    echo "Successful"
    echo -n "${reset}"
    echo "message:${message}"
    echo "has:${match}"
    return 0
  else
    echo -n "${bold}${red}"
    echo "FAIL!"
    echo -n "${reset}"
    echo "message:${message}"
    echo "has not:${match}"
    caller
    return 1
  fi
}

kube::test::if_has_not_string() {
  local message=$1
  local match=$2

  if grep -q "${match}" <<< "${message}"; then
    echo -n "${bold}${red}"
    echo "FAIL!"
    echo -n "${reset}"
    echo "message:${message}"
    echo "has:${match}"
    caller
    return 1
  else
    echo -n "${green}"
    echo "Successful"
    echo -n "${reset}"
    echo "message:${message}"
    echo "has not:${match}"
    return 0
  fi
}

kube::test::if_empty_string() {
  local match=$1
  if [ -n "${match}" ]; then
    echo -n "${bold}${red}"
    echo "FAIL!"
    echo "${match} is not empty"
    echo -n "${reset}"
    caller
    return 1
  else
    echo -n "${green}"
    echo "Successful"
    echo -n "${reset}"
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

kube::test::version::object_to_file() {
  name=$1
  flags=${2:-""}
  file=$3
  # shellcheck disable=SC2086
  # Disabling because "flags" needs to allow for expansion here
  kubectl version ${flags} | grep "${name} Version:" | sed -e s/"${name} Version: "/""/g > "${file}"
}

kube::test::version::json_object_to_file() {
  flags=$1
  file=$2
  # shellcheck disable=SC2086
  # Disabling because "flags" needs to allow for expansion here
  kubectl version ${flags} --output json | sed -e s/' '/''/g -e s/'\"'/''/g -e s/'}'/''/g -e s/'{'/''/g -e s/'clientVersion:'/'clientVersion:,'/ -e s/'serverVersion:'/'serverVersion:,'/ | tr , '\n' > "${file}"
}

kube::test::version::json_client_server_object_to_file() {
  flags=$1
  name=$2
  file=$3
  # shellcheck disable=SC2086
  # Disabling because "flags" needs to allow for expansion here
  kubectl version ${flags} --output json | jq -r ".${name}" | sed -e s/'\"'/''/g -e s/'}'/''/g -e s/'{'/''/g -e /^$/d -e s/','/''/g  -e s/':'/'='/g > "${file}"
}

kube::test::version::yaml_object_to_file() {
  flags=$1
  file=$2
  # shellcheck disable=SC2086
  # Disabling because "flags" needs to allow for expansion here
  kubectl version ${flags} --output yaml | sed -e s/' '/''/g -e s/'\"'/''/g -e /^$/d > "${file}"
}

kube::test::version::diff_assert() {
  local original=$1
  local comparator=${2:-"eq"}
  local latest=$3
  local diff_msg=${4:-""}
  local res=""

  if [ ! -f "${original}" ]; then
        echo "${bold}${red}"
        echo "FAIL! ${diff_msg}"
        echo "the file '${original}' does not exit"
        echo "${reset}${red}"
        caller
        echo "${reset}"
        return 1
  fi

  if [ ! -f "${latest}" ]; then
        echo "${bold}${red}"
        echo "FAIL! ${diff_msg}"
        echo "the file '${latest}' does not exit"
        echo "${reset}${red}"
        caller
        echo "${reset}"
        return 1
  fi

  if [ "${comparator}" == "exact" ]; then
      # Skip sorting of file content for exact comparison.
      cp "${original}" "${original}.sorted"
      cp "${latest}" "${latest}.sorted"
  else
      sort "${original}" > "${original}.sorted"
      sort "${latest}" > "${latest}.sorted"
  fi

  if [ "${comparator}" == "eq" ] || [ "${comparator}" == "exact" ]; then
    if [ "$(diff -iwB "${original}".sorted "${latest}".sorted)" == "" ] ; then
        echo -n "${green}"
        echo "Successful: ${diff_msg}"
        echo -n "${reset}"
        return 0
    else
        echo "${bold}${red}"
        echo "FAIL! ${diff_msg}"
        echo "  Expected: "
        cat "${original}"
        echo "  Got: "
        cat "${latest}"
        echo "${reset}${red}"
        caller
        echo "${reset}"
        return 1
    fi
  else
    if [ -n "$(diff -iwB "${original}".sorted "${latest}".sorted)" ] ; then
        echo -n "${green}"
        echo "Successful: ${diff_msg}"
        echo -n "${reset}"
        return 0
    else
        echo "${bold}${red}"
        echo "FAIL! ${diff_msg}"
        echo "  Expected: "
        cat "${original}"
        echo "  Got: "
        cat "${latest}"
        echo "${reset}${red}"
        caller
        echo "${reset}"
        return 1
      fi
  fi
}

# Force exact match of kubectl stdout, stderr, and return code.
# $1: file with actual stdout
# $2: file with actual stderr
# $3: the actual return code
# $4: file with expected stdout
# $5: file with expected stderr
# $6: expected return code
# $7: additional message describing the invocation
kube::test::results::diff() {
  local actualstdout=$1
  local actualstderr=$2
  local actualcode=$3
  local expectedstdout=$4
  local expectedstderr=$5
  local expectedcode=$6
  local message=$7
  local result=0

  if ! kube::test::version::diff_assert "${expectedstdout}" "exact" "${actualstdout}" "stdout for ${message}"; then
      result=1
  fi
  if ! kube::test::version::diff_assert "${expectedstderr}" "exact" "${actualstderr}" "stderr for ${message}"; then
      result=1
  fi
  if [ "${actualcode}" -ne "${expectedcode}" ]; then
      echo "${bold}${red}"
      echo "$(kube::test::get_caller): FAIL!"
      echo "Return code for ${message}"
      echo "  Expected: ${expectedcode}"
      echo "  Got:      ${actualcode}"
      echo "${reset}${red}"
      caller
      echo "${reset}"
      result=1
  fi

  if [ "${result}" -eq 0 ]; then
     echo -n "${green}"
     echo "$(kube::test::get_caller): Successful: ${message}"
     echo -n "${reset}"
  fi

  return "$result"
}
