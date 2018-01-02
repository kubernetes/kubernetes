#!/usr/bin/env bash
set -e

cd "$(dirname "$BASH_SOURCE")/../.."

targets_from() {
       git fetch -q https://github.com/docker/docker.git "$1"
       git ls-tree -r --name-only "$(git rev-parse FETCH_HEAD)" contrib/builder/deb/ | grep '/Dockerfile$' | sed -r 's!^contrib/builder/deb/|^contrib/builder/deb/amd64/|-debootstrap|/Dockerfile$!!g' | grep -v /
}

release_branch=$(git ls-remote --heads https://github.com/docker/docker.git | awk -F 'refs/heads/' '$2 ~ /^release/ { print $2 }' | sort -V | tail -1)
{ targets_from master; targets_from "$release_branch"; } | sort -u
