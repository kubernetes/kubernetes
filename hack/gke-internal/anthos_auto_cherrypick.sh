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

usage="The script automatically cherrypicks a commit into target branches and all branches newer than them.

Usage: $(basename $0) [-hd] COMMIT_NUMBER TARGET_BRANCH1 [TARGET_BRANCH2 ...]

Options:
  -h  Print help message.
  -d  Dry run without actual change pushed."


while getopts 'hd' o; do
  case "$o" in
    h) echo "$usage"
      exit
      ;;
    d) echo "Dry run mode, no actual change pushed"
      dryrun="true"
      shift
      ;;
    \?) echo "$usage" >&2
      exit 1
      ;;
  esac
done

if [[ $# -lt 2 ]]; then
  echo "Missing arguments!" >&2
  echo "$usage" >&2
  exit
fi

commit="$1"
shift
target_branches=($*)
gke_branch_pattern="release-[0-9]+\.[0-9]+\.[0-9]+-gke"
oss_branch_pattern="release-[0-9]+\.[0-9]+"

# Make sure target branches are all gke patch branches.
for b in "${target_branches[@]}"; do
  if [[ ! "$b" =~ ^${gke_branch_pattern}$ ]]; then
    echo "Version \"$b\" doesn't match pattern \"${gke_branch_pattern}\"" >&2
    echo "$usage" >&2
    exit 1
  fi
done

oss_branches=($(git branch -r | grep -oP "(?<=origin/)${oss_branch_pattern}$"))
gke_branches=($(git branch -r | grep -oP "(?<=origin/)${gke_branch_pattern}$"))
oss_branch_suffix=".9999999"
# Add a big suffix to oss branches, so that it will be greater than gke branches
# for sorting.
for i in "${!oss_branches[@]}"; do
  oss_branches[i]="${oss_branches[$i]}${oss_branch_suffix}"
done
branches=("${oss_branches[@]}" "${gke_branches[@]}")

# Sort both branches and target branches from newer version to older version.
IFS=$'\n' branches=($(sort -Vr <<<"${branches[*]}")); unset IFS
IFS=$'\n' target_branches=($(sort -Vr <<<"${target_branches[*]}")); unset IFS

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

git fetch origin > /dev/null
for b in "${branches[@]}"; do
  remote_branch=origin/$b
  echo "Cherrypicking $commit to branch $remote_branch..."
  git checkout "$remote_branch" > /dev/null
  if ! git cherry-pick "$commit" > /dev/null; then
    while true; do
      echo "Please resolve the conflicts in another window (and remember to 'git add / git cherry-pick --continue')"
      read -p "Proceed (anything but 'y' aborts the cherry-pick)? [y/n] " reply
      if ! [[ "${reply}" =~ ^[yY]$ ]]; then
        git cherry-pick --abort > /dev/null
        echo "Aborting."
        exit 1
      fi
      if ! git status | grep "You are currently cherry-picking commit" &> /dev/null; then
        break
      fi
      echo "Please finish the onging cherrypick"
    done
  fi
  if git diff "$remote_branch" --exit-code > /dev/null; then
    echo "Nothing to push to branch $remote_branch. Skipping older branches..."
    break
  fi
  if [[ "${dryrun:-"false"}" != "true" ]]; then
    echo "Pushing to branch $remote_branch..."
    git push origin "HEAD:refs/for/$b"
  else
    echo "Push skipped in dry run mode"
  fi
  echo "Cherrypicked $commit to branch $remote_branch..."
  # Use the latest commit for future cherrypick to avoid
  # unnecessary conflict.
  commit=$(git rev-parse HEAD)
done

echo "Done!"
