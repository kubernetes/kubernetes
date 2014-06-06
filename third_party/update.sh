#!/bin/bash

set -e

if (( $(git status --porcelain 2>/dev/null | grep "^M" | wc -l) > 0 )); then
  echo "You can't have any staged files in git when updating packages."
  echo "Either commit them or unstage them to continue."
  exit 1
fi

THIRD_PARTY_DIR=$(dirname $0)
cd $THIRD_PARTY_DIR

. ./deps.sh

# Create a temp GOPATH root.  It must be an absolute path
mkdir -p ../target/go_dep_update
cd ../target/go_dep_update
TMP_GO_ROOT=$PWD
cd -
export GOPATH=${TMP_GO_ROOT}

for p in $PACKAGES; do
  echo "Fetching $p"

  # this is the target directory
  mkdir -p src/$p

  # This will checkout the project into src
  go get -u -d $p

  # The go get path
  gp=$TMP_GO_ROOT/src/$p

  # Attempt to find the commit hash of the repo
  cd $gp

  HEAD=
  REL_PATH=$(git rev-parse --show-prefix 2>/dev/null)
  if [[ -z "$HEAD" && $REL_PATH != *target/go_dep_update* ]]; then
    # Grab the head if it is git
    HEAD=$(git rev-parse HEAD)
  fi

  # Grab the head if it is mercurial
  if [[ -z "$HEAD" ]] && hg root >/dev/null 2>&1; then
    HEAD=$(hg id -i)
  fi

  cd -

  # Copy the code into the final directory
  rsync -a -z -r --exclude '.git/' --exclude '.hg/' $TMP_GO_ROOT/src/$p/ $p

  # Make a nice commit about what everything bumped to
  git add $p
  if ! git diff --cached --exit-code > /dev/null 2>&1; then
    git commit -m "bump($p): $HEAD"
  fi
done
