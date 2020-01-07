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

Usage: $(basename $0) [-cdehkrv] COMMIT_ID master|ANTHOS_MAJOR_MINOR_VERSION|[K8S_MAJOR_MINOR_PATCH_VERSION]...

Options:
  -c  List of emails to CC (comma separated).
  -d  Dry run without actual change pushed.
  -e  Do not skip empty cherrypicks. By default empty cherrypicks are skipped.
  -h  Print help message.
  -k  Use Kubernetes major minor patch version instead.
  -r  List of reviewer emails (comma separated).
  -v  Print verbose output.

Example:
$ $(basename $0) -c ldap1@google.com -r ldap2@google.com f61f2f22ab1e2fc038eb43b37753505fb1f162e7 1.1
Target branches generated from the anthos repo:
    pre-release-1.1: release-1.13.7-gke
    pre-release-1.2: release-1.14.7-gke
    master: release-1.14.7-gke
Branches to cherrypick:
    release-1.17
    release-1.16
    release-1.16.4-gke
    ..."

while getopts 'c:dhkr:ev' o; do
  case "$o" in
    c) cc="$OPTARG"
      ;;
    d) echo "Dry run mode, no actual change pushed"
      dryrun="true"
      ;;
    h) echo "$usage"
      exit
      ;;
    k) echo "Use Kubernetes version"
      use_k8s_version="true"
      ;;
    r) reviewers="$OPTARG"
      ;;
    e) echo "Do not skip empty cherrypicks"
      skip_empty="false"
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

if [[ $# -lt 2 ]]; then
  echo "Missing arguments!" >&2
  echo "$usage" >&2
  exit
fi
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

# run_git runs git and generates git output based on verbose mode and exit code.
run_git() {
  set +o errexit
  local output
  local exitcode
  output=$(git "$@" 2>&1)
  exitcode=$?
  if [[ $exitcode -ne 0 ]] || [[ "${verbose:-}" == "true" ]]; then
    echo "$output" >&2
  fi
  set -o errexit
  return $exitcode
}

if git show --no-patch --format="%P" "$commit" | grep " " &>/dev/null; then
  echo "$commit is a merge commit! please use the corresponding regular commit instead." >&2
  exit 1
fi

if ! git diff-index HEAD --quiet; then
  echo "Working directory is dirty! Please make the working directory clean." >&2
  exit 1
fi

# Creating global resources and cleanup on EXIT.
starting_branch="$(git symbolic-ref --short HEAD)"
working_directory="$(mktemp -d)"
cleanup() {
  rm -rf "$working_directory"
  run_git checkout -f "${starting_branch}"
}
trap cleanup EXIT

target_branches=()
if [[ "${use_k8s_version:-}" == "true" ]]; then
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
  if [[ $# -gt 2 ]]; then
    echo "Too many arguments!" >&2
    echo "$usage" >&2
    exit
  fi

  version="$2"

  if [[ ! "$version" =~ ^$major_minor_pattern$ ]] && [[ "$version" != "master" ]];then
    echo "Anthos version \"$version\" is not master and doesn't match pattern \"^$major_minor_pattern$\"" >&2
    echo "$usage" >&2
    exit 1
  fi

  anthos_repo="${working_directory}/anthos_repo"
  run_git clone sso://gke-internal/syllogi/cluster-management "$anthos_repo"
  pushd "$anthos_repo" &>/dev/null
  anthos_branches=("master")
  if [[ "$version" != "master" ]]; then
    anthos_branches+=($(git branch -r | grep -oP "(?<=origin/)${anthos_branch_pattern}$"))
  fi

  # Collect target Kubernetes branches from corresponding anthos branches.
  output=()
  for b in "${anthos_branches[@]}"; do
    # Skip branches older than the target.
    if [[ "$b" != "master" ]] && ! compare_versions "$b" "pre-release-$version"; then
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

# Remove all branches smaller than the smallest target branch.
# (The order is based on `sort -V` version sort result)
# Example:
# * Target branch: release-1.16.2-gke release-1.17.1-gke
# * Original branches:
#     release-1.14.0-gke
#     release-1.15.0-gke
#     release-1.16.0-gke
#     release-1.16.1-gke
#     release-1.16.2-gke
#     release-1.17.0-gke
#     release-1.17.1-gke
#     release-1.17.2-gke
#     release-1.18.0-gke
# * New branches:
#     release-1.16.2-gke
#     release-1.17.0-gke
#     release-1.17.1-gke
#     release-1.17.2-gke
#     release-1.18.0-gke
oldest_target_branch="${target_branches[-1]}"
new_branches=()
for b in "${branches[@]}"; do
  if compare_versions "$b" "$oldest_target_branch"; then
    new_branches+=("$b")
  fi
done
b=( "${a[@]}" )
branches=( "${new_branches[@]}" )

# Remove branches with the same oss patch version, but smaller than target
# branches.
# (The order is based on `sort -V` version sort result)
# Example:
# * Target branch: release-1.16.2-gke release-1.17.1-gke
# * Original branches:
#     release-1.16.2-gke
#     release-1.17.0-gke
#     release-1.17.1-gke
#     release-1.17.2-gke
#     release-1.18.0-gke
# * New branches:
#     release-1.16.2-gke
#     release-1.17.1-gke
#     release-1.17.2-gke
#     release-1.18.0-gke
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
read -p "Proceed? (anything but 'y' aborts the cherry-pick) [y/n]" reply
if ! [[ "${reply}" =~ ^[yY]$ ]]; then
  echo "Aborting."
  exit 1
fi

# Get push options.
push_options=""
if [[ -n "${cc:-}" ]]; then
  IFS=',' read -ra CC <<< "$cc"
  for i in "${CC[@]}"; do
    push_options+="-o cc=$i "
  done
fi

if [[ -n "${reviewers:-}" ]]; then
  IFS=',' read -ra REVIEWERS <<< "$reviewers"
  for i in "${REVIEWERS[@]}"; do
    push_options+="-o r=$i "
  done
fi

# Only add the first 10 characters of the commit hash into the topic.
push_options+="-o topic=cherrypick-$(git rev-parse --short $commit)"

# Do actual cherrypick.
run_git fetch origin
for b in "${branches[@]}"; do
  remote_branch=origin/$b
  echo "Cherrypicking $commit to branch $remote_branch..."
  run_git checkout "$remote_branch"
  if ! run_git cherry-pick "$commit"; then
    if git diff "$remote_branch" --exit-code &> /dev/null && [[ "${skip_empty:-}" != "false" ]]; then
      echo "Empty commit. The commit may already exist in $remote_branch."
      run_git cherry-pick --skip
      continue
    fi
    while true; do
      echo "Please resolve the conflicts in another window (and remember to 'git add / git cherry-pick --continue')"
      read -p "Proceed (anything but 'y' aborts the cherry-pick)? [y/n] " reply
      if ! [[ "${reply}" =~ ^[yY]$ ]]; then
        run_git cherry-pick --abort
        echo "Aborting."
        exit 1
      fi
      if ! git status | grep "You are currently cherry-picking commit" &> /dev/null; then
        break
      fi
      echo "Please finish the ongoing cherrypick"
    done
  fi
  if git diff "$remote_branch" --exit-code &> /dev/null; then
    echo "Nothing to push to branch $remote_branch."
    continue
  fi
  if [[ "${dryrun:-}" != "true" ]]; then
    echo "Pushing to branch $remote_branch..."
    # Amend the commit to make sure it has a change ID.
    run_git commit --amend --no-edit
    run_git push origin "HEAD:refs/for/$b" ${push_options}
  else
    echo "Push skipped in dry run mode"
  fi
  echo "Cherrypicked $commit to branch $remote_branch..."
  # Use the latest commit for future cherrypick to avoid
  # unnecessary conflict.
  commit=$(git rev-parse HEAD)
done

echo "Done!"
