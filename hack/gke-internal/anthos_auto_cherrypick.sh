#!/bin/bash

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

set -o errexit
set -o nounset
set -o pipefail

usage="The script automatically cherrypicks a kubernetes patch for an anthos version.

Usage: $(basename $0) [-dhkv] COMMIT_NUMBER ANTHOS_MAJOR_MINOR_VERSION|K8S_MAJOR_MINOR_PATCH_VERSION

Options:
  -d  Dry run without actual change pushed.
  -h  Print help message.
  -k  Use Kubernetes major minor patch version instead.
  -v  Print verbose output."


while getopts 'dhkv' o; do
  case "$o" in
    d) echo "Dry run mode, no actual change pushed"
      dryrun="true"
      ;;
    h) echo "$usage"
      exit
      ;;
    k) echo "Use Kubernetes version"
      use_k8s_version="true"
      ;;
    v) echo "Verbose output mode"
      set -o xtrace
      verbose="true"
      ;;
    \?) echo "$usage" >&2
      exit 1
      ;;
  esac
done
shift $(($OPTIND-1))

commit="$1"
major_minor_pattern="[0-9]+\.[0-9]+"
major_minor_patch_pattern="[0-9]+\.[0-9]+\.[0-9]+"
anthos_branch_pattern="pre-release-${major_minor_pattern}"
gke_branch_pattern="release-${major_minor_patch_pattern}-gke"
oss_branch_pattern="release-${major_minor_pattern}"

# compare_versions returns:
# * 0 if $1 >= $2;
# * 1 if $1 < $2.
compare_versions() {
  local -r v1="$1"
  local -r v2="$2"
  local s=("$v1" "$v2")
  IFS=$'\n'; local sorted=($(sort -Vr <<<"${s[*]}")); unset IFS
  if [[ "${s[0]}" == "${sorted[0]}" && "${s[1]}" == "${sorted[1]}" ]]; then
    return 0
  fi
  return 1
}

# run_git runs git and generates git output based on verbose mode.
run_git() {
  if [[ ${verbose:-"false"} == "true" ]]; then
    git "$@"
  else
    git "$@" &>/dev/null
  fi
}

target_branches=()
if [[ "${use_k8s_version:-false}" == "true" ]]; then
  if [[ $# -lt 2 ]]; then
    echo "Missing arguments!" >&2
    echo "$usage" >&2
    exit
  fi
  shift
  versions=($*)

  for v in "${versions[@]}"; do
    if [[ ! "$v" =~ ^$major_minor_patch_pattern$ ]];then
      echo "Kubernetes version \"$v\" doesn't match pattern \"^$major_minor_patch_pattern$\"" >&2
      echo "$usage" >&2
      exit 1
    fi
    target_branches+=("release-$v-gke")
  done

  echo "Generated target branches:"
  printf "\t%s\n" "${target_branches[@]}"
else
  if [[ $# -ne 2 ]]; then
    echo "Missing arguments!" >&2
    echo "$usage" >&2
    exit
  fi

  version="$2"

  if [[ ! "$version" =~ ^$major_minor_pattern$ ]];then
    echo "Anthos version \"$version\" doesn't match pattern \"^$major_minor_pattern$\"" >&2
    echo "$usage" >&2
    exit 1
  fi

  anthos_repo=$(mktemp -d)
  trap "rm -rf $anthos_repo" EXIT
  run_git clone sso://gke-internal/syllogi/cluster-management "$anthos_repo"
  pushd "$anthos_repo" &>/dev/null
  anthos_branches=($(git branch -r | grep -oP "(?<=origin/)${anthos_branch_pattern}$"))
  anthos_branches+=("master")

  # Collect target Kubernetes branches from corresponding anthos branches.
  output=()
  for b in "${anthos_branches[@]}"; do
    # Skip branches older than the target.
    if ! compare_versions "$b" "pre-release-$version" && [[ "$b" != "master" ]]; then
      continue
    fi
    run_git checkout "origin/$b"
    # TODO(random-liu): Update this after upgrade simplification.
    k8s_version=$(cat on_prem_bundle/bundles/gkeonprem/bundler.yaml | grep -oP "(?<=KubeAPIServerImageTag: \"v)${major_minor_patch_pattern}(?=-gke\.[0-9]+\")")
    target_branches+=("release-${k8s_version}-gke")
    output+=("$b: release-${k8s_version}-gke")
  done
  popd &>/dev/null

  echo "Target branches generated from the anthos repo:"
  printf "\t%s\n" "${output[@]}"
fi

oss_branches=($(git branch -r | grep -oP "(?<=origin/)${oss_branch_pattern}$"))
gke_branches=($(git branch -r | grep -oP "(?<=origin/)${gke_branch_pattern}$"))
oss_branch_suffix=".9999999"
# Add a big suffix to oss branches, so that it will be greater than gke branches
# for sorting.
for i in "${!oss_branches[@]}"; do
  oss_branches[i]="${oss_branches[$i]}${oss_branch_suffix}"
done
branches=("${oss_branches[@]}" "${gke_branches[@]}")
# Sort branches and target branches from newer version to older version.
IFS=$'\n' branches=($(sort -Vru <<<"${branches[*]}")); unset IFS
IFS=$'\n' target_branches=($(sort -Vru <<<"${target_branches[*]}")); unset IFS

# Remove all branches lower than the oldest target branch.
oldest_target_branch="${target_branches[-1]}"
new_branches=()
for b in "${branches[@]}"; do
  if compare_versions "$b" "$oldest_target_branch"; then
    new_branches+=("$b")
  fi
done
b=( "${a[@]}" )
branches=( "${new_branches[@]}" )

# Remove branches with the same oss patch version, but lower than target
# branches.
for t in "${target_branches[@]}"; do
  prefix=$(echo "$t" | grep -oP "release-[0-9]+\.[0-9]+\.")
  new_branches=()
  for b in "${branches[@]}"; do
    if [[ "$b" != $prefix* ]]; then
      new_branches+=("$b")
    elif compare_versions "$b" "$t"; then
      new_branches+=("$b")
    fi
  done
  branches=( "${new_branches[@]}" )
done

# Remove suffix from oss branches.
for i in "${!branches[@]}"; do
  branches[i]="${branches[$i]%"${oss_branch_suffix}"}"
done
echo "Branches to cherrypick:"
printf "\t%s\n" "${branches[@]}"

run_git fetch origin
checkout_num=0
for b in "${branches[@]}"; do
  remote_branch=origin/$b
  echo "Cherrypicking $commit to branch $remote_branch..."
  run_git checkout "$remote_branch"
  (( checkout_num+=1 ))
  if ! run_git cherry-pick "$commit"; then
    while true; do
      echo "Please resolve the conflicts in another window (and remember to 'git add / git cherry-pick --continue')"
      read -p "Proceed? [y/s/n] ('s' skips the cherry-pick, anything but 'y' or 's' aborts the cherry-pick)" reply
      if [[ "${reply}" =~ ^[sS]$ ]]; then
        run_git cherry-pick --skip
      elif ! [[ "${reply}" =~ ^[yY]$ ]]; then
        run_git cherry-pick --abort
        echo "Aborting."
        exit 1
      fi
      if ! git status | grep "You are currently cherry-picking commit" &> /dev/null; then
        break
      fi
      echo "Please finish the onging cherrypick"
    done
  fi
  if git diff "$remote_branch" --exit-code &> /dev/null; then
    echo "Nothing to push to branch $remote_branch."
    continue
  fi
  if [[ "${dryrun:-"false"}" != "true" ]]; then
    echo "Pushing to branch $remote_branch..."
    # Amend the commit to make sure it has a change ID.
    run_git commit --amend --no-edit
    run_git push origin "HEAD:refs/for/$b"
  else
    echo "Push skipped in dry run mode"
  fi
  echo "Cherrypicked $commit to branch $remote_branch..."
  # Use the latest commit for future cherrypick to avoid
  # unnecessary conflict.
  commit=$(git rev-parse HEAD)
done
# Checkout back to the initial state
run_git checkout @{-$checkout_num}

echo "Done!"
