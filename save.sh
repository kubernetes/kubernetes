#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail

git add -f convert-federation.sh
git add -f save.sh
git commit --amend --no-edit
git reset --hard HEAD
