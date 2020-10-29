#!/usr/bin/env bash

set -o nounset
set -o errexit
set -o pipefail

# This script is intended to simplify the maintaining a rebase branch for
# openshift/kubernetes.
#
# - If the branch named by REBASE_BRANCH does not exist, it will be created by
# branching from UPSTREAM_TAG and merging in TARGET_BRANCH with strategy
# 'ours'.
#
# - If the branch named by REBASE_BRANCH exists, it will be renamed to
# <branch-name>-<timestamp>, a new branch will be created as per above, and
# carries from the renamed branch will be cherry-picked.

UPSTREAM_TAG="${UPSTREAM_TAG:-}"
if [[ -z "${UPSTREAM_TAG}" ]]; then
  echo >&2 "UPSTREAM_TAG is required"
  exit 1
fi

REBASE_BRANCH="${REBASE_BRANCH:-}"
if [[ -z "${REBASE_BRANCH}" ]]; then
  echo >&2 "REBASE_BRANCH is required"
  exit 1
fi

TARGET_BRANCH="${TARGET_BRANCH:-master}"
if [[ -z "${TARGET_BRANCH}" ]]; then
  echo >&2 "TARGET_BRANCH is required"
  exit 1
fi

echo "Ensuring target branch '${TARGET_BRANCH} is updated"
git co "${TARGET_BRANCH}"
git pull

echo "Checking if '${REBASE_BRANCH}' exists"
REBASE_IN_PROGRESS=
if git show-ref --verify --quiet "refs/heads/${REBASE_BRANCH}"; then
  REBASE_IN_PROGRESS=y
fi

# If a rebase is in progress, rename the existing branch
if [[ "${REBASE_IN_PROGRESS}" ]]; then
  TIMESTAMP="$(date +"%Y-%m-%d_%H-%M-%S")"
  PREVIOUS_REBASE_BRANCH="${REBASE_BRANCH}.${TIMESTAMP}"
  echo "Renaming rebase branch '${REBASE_BRANCH}' to '${PREVIOUS_REBASE_BRANCH}'"
  git br -m "${REBASE_BRANCH}" "${PREVIOUS_REBASE_BRANCH}"
fi

echo "Branching upstream tag '${UPSTREAM_TAG}' to rebase branch '${REBASE_BRANCH}'"
git co -b "${REBASE_BRANCH}" "${UPSTREAM_TAG}"

echo "Merging target branch '${TARGET_BRANCH}' to rebase branch '${REBASE_BRANCH}'"
git merge -s ours --no-edit "${TARGET_BRANCH}"

if [[ "${REBASE_IN_PROGRESS}" ]]; then
  echo "Cherry-picking carried commits from previous rebase branch '${PREVIOUS_REBASE_BRANCH}'"
  # The first merge in the previous rebase branch should be the point at which
  # the target branch was merged with the upstream tag. Any commits since this
  # merge should be cherry-picked.
  MERGE_SHA="$(git log --pretty=%H --merges --max-count=1 "${PREVIOUS_REBASE_BRANCH}" )"
  git cherry-pick "${MERGE_SHA}..${PREVIOUS_REBASE_BRANCH}"
fi
