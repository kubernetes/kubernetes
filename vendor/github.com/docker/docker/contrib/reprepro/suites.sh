#!/bin/bash
set -e

cd "$(dirname "$BASH_SOURCE")/../.."

targets_from() {
       git fetch -q https://github.com/docker/docker.git "$1"
       git ls-tree -r --name-only origin/master contrib/builder/deb | grep '/Dockerfile$' | sed -r 's!^contrib/builder/deb/|-debootstrap|/Dockerfile$!!g'
}

{ targets_from master; targets_from release; } | sort -u
